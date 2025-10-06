# models/masked_gnn_neighbour_model.py
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

NEIGHBOUR_GNN_CUSTOM_CONFIG: dict[str, object] = {
    # lane encoder
    "lane_mlp_hiddens": [64],
    "lane_activation": "Tanh",
    # node dim after lane pooling
    "node_dim": 128,
    # GNN
    "gnn_layers": 2,  # number of neighbour->ego message-passing layers
    "gnn_activation": "Tanh",
    "use_self_loop": True,  # include a transformed ego term each layer
    # heads
    "pi_hiddens": [128, 128],
    "vf_hiddens": [128, 128],
    "head_activation": "Tanh",
}
MODEL_NAME: str = "masked_gnn_neighbour_model"


def _mlp(sizes: list[int], act: nn.Module) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class _LaneEncoder(nn.Module):
    """Lane-wise MLP then masked mean over lanes -> node embedding."""

    def __init__(
        self, in_dim: int, lane_mlp_hiddens: list[int], out_dim: int, act_name: str
    ):
        super().__init__()
        Act = getattr(nn, act_name, nn.Tanh)
        if lane_mlp_hiddens:
            self.mlp = _mlp([in_dim, *lane_mlp_hiddens, out_dim], Act)
        else:
            self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, lanes: torch.Tensor, lane_mask: torch.Tensor) -> torch.Tensor:
        """
        lanes:     [B, L, F] or [B*K, L, F]
        lane_mask: [B, L]     or [B*K, L]
        returns:   [B, D]     or [B*K, D]
        """
        B, L, F = lanes.shape
        x = self.mlp(lanes)  # [B, L, D]
        m = lane_mask.unsqueeze(-1)  # [B, L, 1]
        x = x * m  # zero padded
        denom = lane_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        x = x.sum(dim=1) / denom  # [B, D] masked mean
        return x


class _NeighbourToEgoLayer(nn.Module):
    """
    One-hop neighbour -> ego message passing:
        h_ego' = act( W_self h_ego + sum_k w_k * W_nbr h_k )
    where w_k = nbr_mask * nbr_discount (normalised per batch).
    """

    def __init__(self, dim: int, act_name: str, use_self_loop: bool = True):
        super().__init__()
        self.W_nbr = nn.Linear(dim, dim, bias=True)
        self.use_self = use_self_loop
        if use_self_loop:
            self.W_self = nn.Linear(dim, dim, bias=True)
        Act = getattr(nn, act_name, nn.Tanh)
        self.act = Act()

    def forward(
        self, h_ego: torch.Tensor, h_nei: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """
        h_ego: [B, D]
        h_nei: [B, K, D]
        w:     [B, K]      (mask*discount, will be row-normalised inside)
        """
        # normalise weights per batch item; if all zeros, keep zeros
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B,1]
        w_norm = w / w_sum  # [B,K]

        msg = self.W_nbr(h_nei)  # [B,K,D]
        msg = (msg * w_norm.unsqueeze(-1)).sum(dim=1)  # [B,D]

        out = msg
        if self.use_self:
            out = out + self.W_self(h_ego)
        return self.act(out)  # [B,D]


class MaskedNeighbourGNN(TorchModelV2, nn.Module):
    """
    Obs Dict expected:
      - lane_features:  [B, L_max, F]
      - lane_mask:      [B, L_max]
      - action_mask:    [B, A]
      - nbr_features:   [B, K_max, L_max, F]
      - nbr_lane_mask:  [B, K_max, L_max]
      - nbr_mask:       [B, K_max]
      - nbr_discount:   [B, K_max]

    Pipeline:
      1) LaneEncoder -> ego node h_e: [B,D], neighbour nodes H_n: [B,K,D]
      2) Repeat GNN layers neighbour->ego
      3) Policy/Value from final ego embedding
      4) Mask invalid actions to -inf (with all-invalid fallback)
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

        c = model_config.get("custom_model_config", {})
        lane_mlp_hiddens = list(c.get("lane_mlp_hiddens", [64]))
        lane_act = str(c.get("lane_activation", "Tanh"))
        self.D = int(c.get("node_dim", 128))

        g_layers = int(c.get("gnn_layers", 1))
        g_act = str(c.get("gnn_activation", "Tanh"))
        use_self = bool(c.get("use_self_loop", True))

        head_act = str(c.get("head_activation", "Tanh"))
        pi_h = list(c.get("pi_hiddens", [128, 128]))
        vf_h = list(c.get("vf_hiddens", [128, 128]))
        ActHead = getattr(nn, head_act, nn.Tanh)

        # Lane encoders (shared weights for ego and neighbours)
        self.lane_encoder = _LaneEncoder(F, lane_mlp_hiddens, self.D, act_name=lane_act)

        # GNN layers (neighbour -> ego)
        self.gnn = nn.ModuleList(
            [
                _NeighbourToEgoLayer(self.D, act_name=g_act, use_self_loop=use_self)
                for _ in range(max(1, g_layers))
            ]
        )

        # Heads on ego only
        self.pi = _mlp([self.D, *pi_h, self.A], ActHead)
        self.vf = _mlp([self.D, *vf_h, 1], ActHead)

        self._value_out = None

    def _encode_ego(self, lane_features, lane_mask):
        # [B,L,F], [B,L] -> [B,D]
        return self.lane_encoder(lane_features, lane_mask)

    def _encode_nei(self, nbr_features, nbr_lane_mask):
        # [B,K,L,F] -> [B*K, L, F] without assuming contiguity
        B, K, L, F = nbr_features.shape

        # Use reshape (handles non-contiguous) instead of view
        x = nbr_features.reshape(B * K, L, F)  # [B*K, L, F]
        m = nbr_lane_mask.reshape(B * K, L)  # [B*K, L]

        h = self.lane_encoder(x, m)  # [B*K, D]
        return h.reshape(B, K, self.D)  # [B, K, D]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        lane_features = obs["lane_features"].float()  # [B,L,F]
        lane_mask = obs["lane_mask"].float()  # [B,L]
        action_mask = obs["action_mask"].float()  # [B,A]
        nbr_features = obs["nbr_features"].float()  # [B,K,L,F]
        nbr_lane_mask = obs["nbr_lane_mask"].float()  # [B,K,L]
        nbr_mask = obs["nbr_mask"].float()  # [B,K]
        nbr_discount = obs["nbr_discount"].float()  # [B,K]

        h_ego = self._encode_ego(lane_features, lane_mask)  # [B,D]
        H_nei = self._encode_nei(nbr_features, nbr_lane_mask)  # [B,K,D]

        # edge weights from env (mask * discount)
        w = nbr_mask * nbr_discount  # [B,K]

        # message passing: neighbours -> ego (repeat)
        for layer in self.gnn:
            h_ego = layer(h_ego, H_nei, w)  # [B,D]

        logits = self.pi(h_ego)  # [B,A]

        # mask invalid actions to -inf, with fallback if all invalid
        invalid = action_mask <= 0
        masked_logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]

        self._value_out = self.vf(h_ego).squeeze(-1)  # [B]
        return masked_logits, state

    def value_function(self):
        return self._value_out


def register_neighbour_gnn_model():
    ModelCatalog.register_custom_model(MODEL_NAME, MaskedNeighbourGNN)
