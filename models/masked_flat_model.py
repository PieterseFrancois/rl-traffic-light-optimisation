import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

MASKED_FLAT_CUSTOM_CONFIG: dict[str, object] = {
    "policy_head_hidden_sizes": [128, 128],
    "value_head_hidden_sizes": [128, 128],
    "hidden_activation": "Tanh",
}
MODEL_NAME: str = "masked_flat_model"


class MaskedFlatNet(TorchModelV2, nn.Module):
    """
    Obs Dict:
      - lane_features: float32 [L_max, F]
      - lane_mask:     float32 [L_max]      (1 real, 0 padded)
      - action_mask:   float32 [A]          (1 valid, 0 invalid)
    No pooling. We just mask padded rows to zero, flatten, and pass through MLPs.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        L_max, F = obs_space.original_space["lane_features"].shape
        A = int(action_space.n)
        flat_in = int(L_max * F)

        act_name = model_config.get("hidden_activation", "Tanh")
        Act = getattr(nn, act_name, nn.Tanh)
        pi_hiddens = list(model_config.get("policy_head_hidden_sizes", [128, 128]))
        vf_hiddens = list(model_config.get("value_head_hidden_sizes", [128, 128]))

        # Policy head
        pi_layers = []
        dim = flat_in
        for h in pi_hiddens:
            pi_layers += [nn.Linear(dim, h), Act()]
            dim = h
        pi_layers += [nn.Linear(dim, A)]
        self.pi = nn.Sequential(*pi_layers)

        # Value head
        vf_layers = []
        dim = flat_in
        for h in vf_hiddens:
            vf_layers += [nn.Linear(dim, h), Act()]
            dim = h
        vf_layers += [nn.Linear(dim, 1)]
        self.vf = nn.Sequential(*vf_layers)

        self._value_out = None
        self._eps = 1e-8

    def forward(self, input_dict, state, seq_lens):
        lane_features = input_dict["obs"]["lane_features"].float()  # [B, L, F]
        lane_mask = input_dict["obs"]["lane_mask"].float()  # [B, L]
        action_mask = input_dict["obs"]["action_mask"].float()  # [B, A]

        # Zero out padded lanes, then flatten to [B, L*F]
        masked = lane_features * lane_mask.unsqueeze(-1)  # [B, L, F]
        flat = masked.flatten(start_dim=1)  # [B, L*F]

        logits = self.pi(flat)  # [B, A]

        invalid = action_mask <= 0
        masked_logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        # if an entire row is invalid, fall back to raw logits to avoid NaNs
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]

        self._value_out = self.vf(flat).squeeze(-1)
        return masked_logits, state

    def value_function(self):
        return self._value_out


def register_flat_model():
    ModelCatalog.register_custom_model(MODEL_NAME, MaskedFlatNet)
