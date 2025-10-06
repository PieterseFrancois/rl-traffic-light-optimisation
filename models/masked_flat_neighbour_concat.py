# models/masked_flat_neighbour_concat.py
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

NEIGHBOUR_FLAT_CUSTOM_CONFIG: dict[str, object] = {
    "fcnet_hiddens": [128, 128],
    "vf_hiddens": [128, 128],
    "fcnet_activation": "Tanh",
}
MODEL_NAME: str = "masked_flat_neighbour_concat"


class MaskedFlatNeighbourConcat(TorchModelV2, nn.Module):
    """
    Obs Dict:
      - lane_features:  [B, L_max, F]
      - lane_mask:      [B, L_max]         (1 real, 0 padded)
      - action_mask:    [B, A]             (1 valid, 0 invalid)
      - nbr_features:   [B, K_max, L_max, F]
      - nbr_lane_mask:  [B, K_max, L_max]
      - nbr_mask:       [B, K_max]         (1 neighbour slot used, else 0)
      - nbr_discount:   [B, K_max]         (per-neighbour scalar weight)

    Behaviour:
      - Ego: mask padded lanes → flatten → [B, L_max*F]
      - Neighbours: mask lanes, multiply each neighbour block by (nbr_mask * nbr_discount),
        then flatten K blocks in fixed order and concatenate to ego: [B, (1+K_max)*L_max*F]
      - Simple MLP heads. Invalid actions are masked to -inf (with fallback).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        L_max, F = obs_space.original_space["lane_features"].shape
        K_max, Ln, Fn = obs_space.original_space["nbr_features"].shape
        assert Ln == L_max and Fn == F, "Neighbour L/F must match ego L/F"

        A = int(action_space.n)
        flat_in = int((1 + K_max) * L_max * F)

        act_name = model_config.get("fcnet_activation", "Tanh")
        Act = getattr(nn, act_name, nn.Tanh)
        pi_hiddens = list(model_config.get("fcnet_hiddens", [128, 128]))
        vf_hiddens = list(model_config.get("vf_hiddens", [128, 128]))

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

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        lane_features = obs["lane_features"].float()  # [B, L, F]
        lane_mask = obs["lane_mask"].float()  # [B, L]
        action_mask = obs["action_mask"].float()  # [B, A]

        # Ego: zero padded lanes, then flatten
        ego = (lane_features * lane_mask.unsqueeze(-1)).flatten(start_dim=1)  # [B, L*F]

        # Neighbours
        nbr_feat = obs["nbr_features"].float()  # [B, K, L, F]
        nbr_lmask = obs["nbr_lane_mask"].float()  # [B, K, L]
        nbr_mask = obs["nbr_mask"].float()  # [B, K]
        nbr_disc = obs["nbr_discount"].float()  # [B, K]

        # Mask padded lanes per neighbour
        nbr = nbr_feat * nbr_lmask.unsqueeze(-1)  # [B, K, L, F]

        # Apply per-neighbour scalar weight = nbr_mask * nbr_discount
        w = (nbr_mask * nbr_disc).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        nbr = nbr * w  # [B, K, L, F]

        # Flatten neighbours in fixed order (no pooling)
        nbr_flat = nbr.flatten(start_dim=2)  # [B, K, L*F]
        nbr_flat = nbr_flat.flatten(start_dim=1)  # [B, K*L*F]

        # Concatenate ego + neighbours
        x = torch.cat([ego, nbr_flat], dim=1)  # [B, (1+K)*L*F]

        # Heads
        logits = self.pi(x)  # [B, A]

        # Mask invalid actions to -inf, but fall back if all invalid
        invalid = action_mask <= 0
        masked_logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        all_invalid = invalid.all(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]

        self._value_out = self.vf(x).squeeze(-1)  # [B]
        return masked_logits, state

    def value_function(self):
        return self._value_out


def register_neighbour_flat_model():
    ModelCatalog.register_custom_model(MODEL_NAME, MaskedFlatNeighbourConcat)
