# ============================================================
#  DISCLAIMER:
#  Whilst this module was not coded using an LLM, the docstrings and explanatory comments
#  were written with the assistance of an LLM (GPT 5)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class EmbedderHyperparameters:
    """Configuration for the LaneSetAttentionEmbedder.

    Attributes:
        intermediate_vector_length (int): Dimensionality of the intermediate lane embeddings (E)
        post_aggregation_hidden_length (int): Dimensionality of the hidden layer in the post-aggregation multi-layer perceptron (MLP)
        output_vector_length (int): Dimensionality of the output embedding
        num_seeds (int): Number of learnable seed vectors (queries) - how many different "views" to take of the lane set
        dropout (float): Dropout probability [training only] (used to prevent overfitting by randomly zeroing some lane embeddings)
        layernorm (bool): Whether to apply LayerNorm to the lane embeddings
    """

    intermediate_vector_length: int = 64
    post_aggregation_hidden_length: int = 64
    output_vector_length: int = 64
    num_seeds: int = 1
    dropout: float = 0.0
    layernorm: bool = True


class LaneSetAttentionEmbedder(nn.Module):
    """
    Permutation-invariant lane embedding module.
    Takes a lane-by-feature state matrix [L, F] and produces a fixed-size vector [output_vector_length],
    regardless of the number of lanes L.

    High-level:
    1) Per-lane MLP encodes each lane's features -> lane embeddings (size = intermediate_vector_length).
    2) A set of learnable 'seed' queries attends over lanes to form K pooled summaries.
    3) The K summaries are concatenated and passed through a post-MLP to the final output size.

    Shapes:
        Input:
            X: [L, F]  where L = num_lanes, F = num_of_features
        Output:
            z: [output_vector_length]
            weights: [num_seeds, L]  softmax weights over lanes for each seed (useful for inspection)
    """

    def __init__(
        self,
        num_of_features: int,
        hyperparams: EmbedderHyperparameters = EmbedderHyperparameters(),
    ):
        # Initialise parent module, namely, nn.Module
        super().__init__()

        # Minor validation of hyperparameters before using
        self._validate_hyperparameters(hyperparams)
        if num_of_features < 1:
            raise ValueError("Embedder Module: num_of_features must be at least 1")

        # Phi: per-lane encoder
        self.phi = nn.Sequential(
            nn.Linear(num_of_features, hyperparams.intermediate_vector_length),
            nn.ReLU(inplace=True),
            nn.Linear(
                hyperparams.intermediate_vector_length,
                hyperparams.intermediate_vector_length,
            ),
        )
        self.ln = (
            nn.LayerNorm(hyperparams.intermediate_vector_length)
            if hyperparams.layernorm
            else nn.Identity()
        )
        self.do = (
            nn.Dropout(hyperparams.dropout)
            if hyperparams.dropout > 0
            else nn.Identity()
        )

        # Attention parameters
        self.W = nn.Linear(
            hyperparams.intermediate_vector_length,
            hyperparams.intermediate_vector_length,
            bias=True,
        )  # mix lane embeddings
        self.U = nn.Linear(
            hyperparams.intermediate_vector_length,
            hyperparams.intermediate_vector_length,
            bias=False,
        )  # mix seed/query
        self.v = nn.Linear(
            hyperparams.intermediate_vector_length, 1, bias=False
        )  # to score lanes

        # Learnable seed vectors (queries) q_k
        STABLE_ATTENTION_INIT: float = 0.02
        self.seeds = nn.Parameter(
            torch.randn(hyperparams.num_seeds, hyperparams.intermediate_vector_length)
            * STABLE_ATTENTION_INIT
        )

        # p: post-aggregation MLP to final output embedding
        self.rho = nn.Sequential(
            nn.Linear(
                hyperparams.num_seeds * hyperparams.intermediate_vector_length,
                hyperparams.post_aggregation_hidden_length,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hyperparams.post_aggregation_hidden_length,
                hyperparams.output_vector_length,
            ),
        )

    def _validate_hyperparameters(self, hyperparams: EmbedderHyperparameters) -> None:
        """Validate hyperparameters for the embedder."""
        if hyperparams.num_seeds < 1:
            raise ValueError("Invalid Embedder Hyperparameter: num_seeds must be at least 1")
        if hyperparams.intermediate_vector_length < 1:
            raise ValueError("Invalid Embedder Hyperparameter: intermediate_vector_length must be at least 1")
        if hyperparams.output_vector_length < 1:
            raise ValueError("Invalid Embedder Hyperparameter: output_vector_length must be at least 1")
        if hyperparams.post_aggregation_hidden_length < 1:
            raise ValueError("Invalid Embedder Hyperparameter: post_aggregation_hidden_length must be at least 1")
        if not (0.0 <= hyperparams.dropout < 1.0):
            raise ValueError("Invalid Embedder Hyperparameter: dropout must be in the range [0.0, 1.0)")

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
          X: [L, F] float tensor of lane features in any order (permutation invariance is handled by attention).
             L = number of lanes, F = num_of_features.

        Returns:
          z: [output_vector_length] fixed-size embedding of the whole intersection state.
          weights: [num_seeds, L] attention weights over lanes (softmax per seed). Useful for inspection.
        """
        # Ensure input is a 2D (matrix) tensor
        if X.ndim != 2:
            raise ValueError(f"Expected X of shape [L, F], got {tuple(X.shape)}")

        # 1) Encode each lane into an embedding h_i
        #    φ: [L, F] -> [L, E]
        H = self.phi(X)  # [L, E]
        H = self.ln(H)  # optional LayerNorm
        H = self.do(H)  # optional Dropout
        _, E = H.shape  # E should equal intermediate_vector_length
        K = self.seeds.size(0)  # number of seed queries

        # 2) Attention scores per seed over lanes:
        #    For each seed k, score lane i with: v^T tanh( W h_i + U q_k )
        Wh = self.W(H)  # [L, E]
        Uq = self.U(self.seeds)  # [K, E]

        # Broadcast add to combine per-lane and per-seed terms → [K, L, E]
        SEED_AXIS: int = 0
        LANE_AXIS: int = 1
        joint = torch.tanh(
            Wh.unsqueeze(SEED_AXIS) + Uq.unsqueeze(LANE_AXIS)
        )  # [K, L, E]

        LAST_SINGLETON_DIM: int = -1
        scores = self.v(joint).squeeze(LAST_SINGLETON_DIM)  # [K, L]

        LAST_DIM_OF_SOFTMAX: int = -1
        weights = F.softmax(
            scores, dim=LAST_DIM_OF_SOFTMAX
        )  # [K, L], sum over lanes = 1 per seed

        # 3) Weighted sums per seed: z_k = Σ_i α_{k,i} h_i
        Zk = torch.matmul(weights, H)  # [K, E]

        # 4) Concatenate seed summaries and map to final size
        Z = Zk.reshape(K * E)  # [K*E]
        z = self.rho(Z)  # [output_vector_length]
        return z, weights
