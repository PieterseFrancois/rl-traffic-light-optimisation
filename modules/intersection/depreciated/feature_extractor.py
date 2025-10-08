import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dataclasses import dataclass

from .embedder import LaneSetAttentionEmbedder, EmbedderHyperparameters


@dataclass
class FeatureExtractorConfig:
    """
    Configuration for the SB3-compatible feature extractor.

    Attributes:
        number_of_features (int): Number of features per lane (F).
        embedder_hyperparams (EmbedderHyperparameters): Hyperparameters for the LaneSetAttentionEmbedder.
    """

    number_of_features: int
    embedder_hyperparams: EmbedderHyperparameters
    neighbourhood_aggregator: None = None  # Placeholder for future use


class FeatureExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor that:
      - Accepts a FLAT observation: [L * F]
      - Reshapes to [L, F]
      - Runs LaneSetAttentionEmbedder
      - Returns a fixed-size vector of length `config.out_dim`

    IMPORTANT: This class lives inside the policy, so its parameters are trained end-to-end via PPO.
    """

    def __init__(self, observation_space: spaces.Box, config: FeatureExtractorConfig):
        # Must call super with the output vector length (policy expects this)
        super().__init__(
            observation_space=observation_space,
            features_dim=config.embedder_hyperparams.output_vector_length,
        )

        self.number_of_features = config.number_of_features

        # Sanity check on the observation space shape (flat vector)
        if observation_space.shape is None or len(observation_space.shape) != 1:
            raise ValueError(
                f"FeatureExtractor expects a 1D flattened observation space, got shape={observation_space.shape}"
            )

        # Core learnable pooling block
        self.embedder = LaneSetAttentionEmbedder(
            num_of_features=self.number_of_features,
            hyperparams=config.embedder_hyperparams,
        )

        if config.neighbourhood_aggregator:
            pass  # Placeholder for future use - update forward as well

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [B, L*F] (SB3 provides batches)
        Returns:
          features: [B, out_dim]
        """
        if observations.ndim != 2:
            raise ValueError(
                f"Expected batched obs tensor [B, L*F], got {tuple(observations.shape)}"
            )

        B, flat_dim = observations.shape
        if flat_dim % self.number_of_features != 0:
            raise ValueError(
                f"Obs length {flat_dim} is not divisible by per-lane dim F={self.number_of_features}. "
                "Ensure your Preprocessor row width matches FeatureExtractorConfig.in_dim_per_lane."
            )

        L = flat_dim // self.number_of_features  # number of lanes in this intersection

        # Loop over observation space
        outputs = []
        for b in range(B):
            x_flat = observations[b]  # [L*F]
            x = x_flat.view(L, self.number_of_features)  # [L, F]
            z, _weights = self.embedder(x)  # z: [out_dim]
            outputs.append(z)

        return torch.stack(outputs, dim=0)  # [B, out_dim]
