# ============================================================
#  DISCLAIMER:
#  Whilst this model was originally written without the use of a LLM, 
#  a LLM code review was used. During this review, the model was slightly
#  refactored for clarity, and comments/docstrings were added.
#  No changes were made to the model architecture or logic.
# ============================================================

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


MODEL_NAME: str = "masked_gnn_attention_model"


def _build_mlp(sizes: list[int], Act: type[nn.Module] | None) -> nn.Sequential:
    """
    Build a feedforward MLP with optional activation between all but last layer.

    Args:
        sizes(list[int]): List of layer sizes, including input and output.
        Act(type[nn.Module] | None): Activation class (e.g. nn.ReLU) or None for linear.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(Act()) if Act is not None else None
    return nn.Sequential(*layers)


class _LaneAttentionEncoder(nn.Module):
    """
    Produce a single embedding from multiple lane features via:
      1) Per-lane MLP to produce lane embeddings
      2) Additive attention across lanes to produce single embedding
    """

    def __init__(
        self,
        in_dim: int,
        lane_mlp_hiddens: list[int],
        out_dim: int,
        act_name: str,
        attn_hidden: int = 64,
    ):
        super().__init__()
        Act = getattr(nn, act_name, nn.Tanh)
        self.mlp = _build_mlp(
            [in_dim, *lane_mlp_hiddens, out_dim], Act
        )  # Per-lane MLP to produce embeddings
        self.W = nn.Linear(
            out_dim, attn_hidden, bias=True
        )  # Attention feature transform
        self.v = nn.Linear(
            attn_hidden, 1, bias=False
        )  # Attention scoring vector - collapses attention features to a single score per lane
        self.tanh = nn.Tanh()

    def forward(self, lanes: torch.Tensor, lane_mask: torch.Tensor) -> torch.Tensor:
        lane_embeddings = self.mlp(lanes)  # [B, L, D]
        attn_scores = self.v(self.tanh(self.W(lane_embeddings))).squeeze(
            -1
        )  # [B, L]
        attn_scores = attn_scores.masked_fill(lane_mask <= 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L]
        attn_weights = torch.where(
            torch.isfinite(attn_weights),
            attn_weights,
            torch.zeros_like(attn_weights),
        )  # all-masked safety
        encoded = (lane_embeddings * attn_weights.unsqueeze(-1)).sum(
            dim=1
        )  # [B, D]
        return encoded


class _NeighbourToEgoGATLayer(nn.Module):
    """
    Neighbour → Ego GAT update.

    h_ego' = act( self_loop_proj(h_ego) + Σ_k α_k * msg_proj(h_k) )

    where:
      - msg_proj: shared linear that projects embeddings before attention and messaging
      - attn_pair_scorer: additive scorer on [proj(h_ego) || proj(h_k)]
      - α_k: masked softmax over neighbours (nbr_mask == 0 removes k)
      - act: output nonlinearity for the updated ego embedding
    Shapes:
      h_ego: [B, D], h_nei: [B, K, D], nbr_mask: [B, K] in {0,1}
    """

    def __init__(self, dim: int, act_name: str, use_self_loop: bool = True):
        super().__init__()
        # Projects ego and neighbour embeddings into an attention/message space
        self.msg_proj = nn.Linear(dim, dim, bias=False)

        # Scores a neighbour given the ego context
        self.attn_pair_scorer = nn.Linear(2 * dim, 1, bias=False)
        self.attn_activation = nn.LeakyReLU(0.2)

        self.use_self_loop = use_self_loop
        if use_self_loop:
            # Optional self contribution to keep ego context each layer
            self.self_loop_proj = nn.Linear(dim, dim, bias=True)

        Act = getattr(nn, act_name, nn.Tanh)
        self.out_activation = Act()

    def forward(
        self,
        h_ego: torch.Tensor,  # [B, D]
        h_nei: torch.Tensor,  # [B, K, D]
        nbr_mask: torch.Tensor,  # [B, K] in {0,1}
    ) -> torch.Tensor:
        # Shared projection for both ego and neighbours
        nbr_proj = self.msg_proj(h_nei)  # [B, K, D]
        ego_proj = self.msg_proj(h_ego).unsqueeze(1).expand_as(nbr_proj)  # [B, K, D]

        # Additive attention logits per neighbour
        attn_logits = self.attn_pair_scorer(
            torch.cat([ego_proj, nbr_proj], dim=-1)
        ).squeeze(
            -1
        )  # [B, K]
        attn_logits = self.attn_activation(attn_logits)

        # Mask absent neighbours before softmax
        attn_logits = attn_logits.masked_fill(nbr_mask <= 0, float("-inf"))

        # Attention weights
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, K]
        attn_weights = torch.where(
            torch.isfinite(attn_weights), attn_weights, torch.zeros_like(attn_weights)
        )

        # Neighbour message and optional self loop
        nbr_message = (nbr_proj * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        updated = nbr_message + (
            self.self_loop_proj(h_ego) if self.use_self_loop else 0.0
        )

        return self.out_activation(updated)  # [B, D]


class MaskedNeighbourGNN(TorchModelV2, nn.Module):
    """
    Obs Dict:
      lane_features:  [B, L_max, F]
      lane_mask:      [B, L_max]
      action_mask:    [B, A]
      nbr_features:   [B, K_max, L_max, F]
      nbr_lane_mask:  [B, K_max, L_max]
      nbr_mask:       [B, K_max]   (1 if neighbour present else 0)
      nbr_discount:   [B, K_max]   (present but ignored)

    Pipeline:
      1) Lane -> node via MLP + attention (shared for ego and neighbours)
      2) Repeat GAT neighbour->ego message passing (ignores discounts)
      3) Policy/Value heads on final ego embedding
      4) Safe action masking
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Shapes from obs space
        L_max, F = obs_space.original_space["lane_features"].shape
        K_max, Ln, Fn = obs_space.original_space["nbr_features"].shape
        assert Ln == L_max and Fn == F, "Neighbour L/F must match ego L/F"

        self.K = int(K_max)
        self.A = int(action_space.n)

        c = model_config.get("custom_model_config", {}) or {}

        lane_mlp_hiddens = list(c.get("lane_mlp_hiddens"))
        lane_act = str(c.get("lane_activation"))
        lane_attn_hiddens = int(c.get("lane_attn_hiddens"))

        self.D = int(c.get("node_dim"))
        g_layers = int(c.get("gnn_layers"))
        g_act = str(c.get("gnn_activation"))
        use_self = bool(c.get("use_self_loop"))

        actor_hiddens = list(c.get("actor_hiddens"))
        critic_hiddens = list(c.get("critic_hiddens"))

        # Lane -> node encoder (shared)
        self.lane_encoder = _LaneAttentionEncoder(
            in_dim=F,
            lane_mlp_hiddens=lane_mlp_hiddens,
            out_dim=self.D,
            act_name=lane_act,
            attn_hidden=lane_attn_hiddens,
        )

        # GAT layers (neighbour -> ego)
        self.gnn = nn.ModuleList(
            [
                _NeighbourToEgoGATLayer(self.D, act_name=g_act, use_self_loop=use_self)
                for _ in range(max(1, g_layers))
            ]
        )

        # Heads on ego only
        self.policy_actor = _build_mlp([self.D, *actor_hiddens, self.A], None)
        self.value_critic = _build_mlp([self.D, *critic_hiddens, 1], None)

        self._value_out: torch.Tensor | None = None

    def _encode_ego(
        self, lane_features: torch.Tensor, lane_mask: torch.Tensor
    ) -> torch.Tensor:
        # [B, L, F], [B, L] -> [B, D]
        return self.lane_encoder(lane_features, lane_mask)

    def _encode_nei(
        self, nbr_features: torch.Tensor, nbr_lane_mask: torch.Tensor
    ) -> torch.Tensor:
        # [B, K, L, F] -> [B*K, L, F] via reshape
        B, K, L, F = nbr_features.shape
        x = nbr_features.reshape(B * K, L, F)
        m = nbr_lane_mask.reshape(B * K, L)
        h = self.lane_encoder(x, m)  # [B*K, D]

        return h.reshape(B, K, self.D)  # [B, K, D]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        lane_features = obs["lane_features"].float()  # [B, L, F]
        lane_mask = obs["lane_mask"].float()  # [B, L]
        action_mask = obs["action_mask"].float()  # [B, A]

        nbr_features = obs["nbr_features"].float()  # [B, K, L, F]
        nbr_lane_mask = obs["nbr_lane_mask"].float()  # [B, K, L]
        nbr_mask = obs["nbr_mask"].float()  # [B, K]

        h_ego = self._encode_ego(lane_features, lane_mask)  # [B, D]
        H_nei = self._encode_nei(nbr_features, nbr_lane_mask)  # [B, K, D]

        # GAT neighbour -> ego message passing
        for layer in self.gnn:
            h_ego = layer(h_ego, H_nei, nbr_mask)  # [B, D]

        logits = self.policy_actor(h_ego)  # [B, A]

        # Safe action masking to -inf with all-invalid fallback
        invalid = action_mask <= 0
        masked_logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]

        self._value_out = self.value_critic(h_ego).squeeze(-1)  # [B]
        return masked_logits, state

    def value_function(self):
        return self._value_out

def register_attention_gnn_model():
    ModelCatalog.register_custom_model(MODEL_NAME, MaskedNeighbourGNN)
