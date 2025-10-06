import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

# Optional external embedder (your existing implementation)
# If you don’t want this path, keep use_simple_attention=True in config.
from modules.intersection.feature_extractor.embedder import (
    LaneSetAttentionEmbedder,
    EmbedderHyperparameters,
)


MASKED_EMBED_CUSTOM_CONFIG: dict[str, object] = {
    # embedder choice
    "use_simple_attention": True,  # False to use LaneSetAttentionEmbedder
    "embed_out_dim": 64,  # pooled embedding size (simple attention)
    # MLP heads
    "fcnet_hiddens": [128, 128],
    "vf_hiddens": [128, 128],
    "fcnet_activation": "Tanh",
    # If you switch to LaneSetAttentionEmbedder (use_simple_attention=False),
    # these hparams are passed into your EmbedderHyperparameters(**...):
    "embedder_hparams": {
        "intermediate_vector_length": 64,
        "post_aggregation_hidden_length": 64,
        "output_vector_length": 64,  # keep equal to embed_out_dim
        "num_seeds": 1,
        "dropout": 0.0,
        "layernorm": True,
    },
}

EMBED_MODEL_NAME = "masked_embed_model"


# ---------- Simple single-head attention pool over lanes ----------


class SingleHeadAttentionPool(nn.Module):
    """
    Additive attention over rows of X: [L_i, F] -> pooled [E_out]

        score_i = v^T tanh(W x_i)
        w = softmax(score)
        z = sum_i w_i * x_i
        out = proj(z) ∈ R^{E_out}
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, in_dim, bias=True)
        self.v = nn.Linear(in_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self._out_dim = int(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [L_i, F] (may be empty if no active lanes)
        returns: [E_out]
        """
        if X.numel() == 0:
            return torch.zeros(self._out_dim, device=X.device, dtype=X.dtype)

        H = self.tanh(self.W(X))  # [L_i, F]
        scores = self.v(H).squeeze(-1)  # [L_i]
        w = torch.softmax(scores, dim=-1)  # [L_i]
        pooled = (w.unsqueeze(0) @ X).squeeze(0)  # [F]
        out = self.proj(pooled)
        return out.view(-1)  # [E_out]


# -------------------------- RLlib model ---------------------------


class MaskedEmbedNet(TorchModelV2, nn.Module):
    """
    Observation dict (per RLlib):
      - lane_features: float32 [B, L_max, F]
      - lane_mask:     float32 [B, L_max]      (1 real, 0 padded)
      - action_mask:   float32 [B, A]          (1 valid, 0 invalid)

    Pipeline (configurable via custom_model_config):
      1) Slice active lanes by lane_mask → X_b: [L_i, F]
      2a) If use_simple_attention=True: SingleHeadAttentionPool(X_b) → [E]
      2b) Else: LaneSetAttentionEmbedder(X_b)                       → [E]
      3) Heads: policy logits [A], value scalar
      4) Hard-mask invalid actions by setting logits to -inf
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Shapes
        L_max, F = obs_space.original_space["lane_features"].shape
        self.A = int(action_space.n)

        # Custom config
        cconf = model_config.get("custom_model_config", {})
        self._use_simple_attention: bool = bool(cconf.get("use_simple_attention", True))
        embed_out_dim: int = int(cconf.get("embed_out_dim", 64))

        # Embedder choice
        if self._use_simple_attention:
            self.embedder = SingleHeadAttentionPool(in_dim=F, out_dim=embed_out_dim)
            embed_dim = embed_out_dim
            self._uses_lane_set_embedder = False
        else:
            hp = cconf.get(
                "embedder_hparams",
                {
                    "intermediate_vector_length": 64,
                    "post_aggregation_hidden_length": 64,
                    "output_vector_length": embed_out_dim,
                    "num_seeds": 1,
                    "dropout": 0.0,
                    "layernorm": True,
                },
            )
            self.embedder = LaneSetAttentionEmbedder(
                num_of_features=F, hyperparams=EmbedderHyperparameters(**hp)
            )
            embed_dim = int(hp.get("output_vector_length", embed_out_dim))
            self._uses_lane_set_embedder = True

        # Heads
        act_name = model_config.get("fcnet_activation", "Tanh")
        Act = getattr(nn, act_name, nn.Tanh)
        pi_hiddens = list(model_config.get("fcnet_hiddens", [128, 128]))
        vf_hiddens = list(model_config.get("vf_hiddens", [128, 128]))

        self.pi = self._mlp(embed_dim, self.A, pi_hiddens, Act)
        self.vf = self._mlp(embed_dim, 1, vf_hiddens, Act)

        # Stable init for Tanh/Relu
        self._init_mlp(self.pi, act_name)
        self._init_mlp(self.vf, act_name)

        self._value_out: torch.Tensor | None = None
        self._embed_dim = embed_dim

    # ---- utils ----
    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hiddens: list[int], Act) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = in_dim
        for h in hiddens:
            layers += [nn.Linear(dim, h), Act()]
            dim = h
        layers += [nn.Linear(dim, out_dim)]
        return nn.Sequential(*layers)

    @staticmethod
    def _init_mlp(seq: nn.Sequential, act_name: str) -> None:
        gain = nn.init.calculate_gain("tanh" if act_name.lower() == "tanh" else "relu")
        for m in seq.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    # ---- RLlib API ----
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        lane_features: torch.Tensor = obs["lane_features"].float()  # [B, L_max, F]
        lane_mask: torch.Tensor = obs["lane_mask"].float()  # [B, L_max]
        action_mask: torch.Tensor = obs["action_mask"].float()  # [B, A]

        B = lane_features.size(0)

        # Embed per sample (simple and robust; can be batched later if needed)
        embeds = []
        for b in range(B):
            Xb = lane_features[b]  # [L_max, F]
            mb = lane_mask[b] > 0.5  # [L_max] -> bool
            if mb.any():
                X = Xb[mb]  # [L_i, F]
                if self._uses_lane_set_embedder:
                    out = self.embedder(X)
                    # Support (z, aux) or z
                    z = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    z = self.embedder(X)
                z = z.view(-1)  # ensure [E]
            else:
                z = torch.zeros(self._embed_dim, device=Xb.device, dtype=Xb.dtype)
            embeds.append(z)

        z = torch.stack(embeds, dim=0)  # [B, E]

        logits = self.pi(z)  # [B, A]

        # Hard action mask: remove probability mass from invalid actions
        invalid = action_mask <= 0
        masked_logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        # Fallback if a row is entirely invalid (should not happen, but safe)
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]

        self._value_out = self.vf(z).squeeze(-1)  # [B]
        return masked_logits, state

    def value_function(self):
        return self._value_out


def register_embedder_model() -> None:
    """Register this model under the name 'masked_embed_model' for RLlib."""
    ModelCatalog.register_custom_model(EMBED_MODEL_NAME, MaskedEmbedNet)
