"""
SB3 Features Extractor for a single intersection agent.

It embeds the current intersection's *raw* state with a small MLP, then mixes
that self embedding with neighbour embeddings using a GATv2 GNN over a fixed
local graph declared at initialisation. The output (self node's GNN embedding)
feeds the SB3 policy/value heads.

Expected observation from the env (per agent):
  obs["self_raw"]   : Tensor (B, F_raw)     -- raw features for *this* intersection
  obs["nbr_embed"]  : Tensor (B, K, D_emb)  -- cached neighbour embeddings in a fixed order

Notes:
- K can differ across agents, but must be fixed for a given agent/env.
- If a neighbour is temporarily missing, fill its row with zeros on the env side.
- Node index 0 is always the self node. Neighbours occupy indices 1..K.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.nn import GATv2Conv
except Exception as e:
    raise RuntimeError(
        "torch-geometric is required: pip install torch-geometric"
    ) from e


class StateEmbedder(nn.Module):
    """
    Small MLP that turns raw per-intersection features into a fixed-length embedding.
    This is where the network can learn to weight certain lanes/features more strongly.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)


class GATv2Aggregator(nn.Module):
    """
    Two-layer GATv2 that mixes a node with its neighbours.
    Works on one local, fixed graph with N = 1 + K nodes (self + K neighbours).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, heads: int = 2):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hidden, heads=heads, concat=True)
        self.g2 = GATv2Conv(hidden * heads, out_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D), edge_index: (2, E)
        returns: (N, out_dim)
        """
        h = F.elu(self.g1(x, edge_index))
        h = F.elu(self.g2(h, edge_index))
        return h


class NeighbourGNNFeatures(BaseFeaturesExtractor):
    """
    SB3 FeaturesExtractor that:
      1) embeds the current intersection's raw features with StateEmbedder,
      2) stacks [self_embed] + neighbour_embeds into node features,
      3) runs a fixed GATv2 graph and returns the self node's final embedding.

    Observation keys:
      - "self_raw": (B, F_raw)
      - "nbr_embed": (B, K, D_emb) with fixed K for this agent
    """

    def __init__(
        self,
        observation_space,  # gym.spaces.Dict with "self_raw" and "nbr_embed"
        self_raw_dim: int,  # F_raw
        embed_dim: int,  # D_emb (also GNN in/out)
        gat_hidden: int = 64,
        gat_heads: int = 2,
        edge_index: Optional[
            torch.Tensor
        ] = None,  # (2, E) over N = 1+K nodes; 0 = self
        device: Optional[torch.device] = None,
    ):
        super().__init__(observation_space, features_dim=embed_dim)

        self.device = device or torch.device("cpu")
        self.self_embed = StateEmbedder(self_raw_dim, embed_dim).to(self.device)
        self.gnn = GATv2Aggregator(
            embed_dim, embed_dim, hidden=gat_hidden, heads=gat_heads
        ).to(self.device)

        # Edge index for the fixed local graph of this agent.
        # If not provided, default to a star: self connected to each neighbour both ways.
        self.register_buffer("edge_index", None, persistent=False)
        if edge_index is not None:
            self.edge_index = edge_index.long().to(self.device)

        # Cache last self embedding to publish via getter
        self._last_self_embedding: Optional[torch.Tensor] = None

    @torch.no_grad()
    def get_last_self_embedding(self) -> Optional[torch.Tensor]:
        """
        Returns the last self-node embedding produced by forward(), shape (D_emb,).
        Use this to publish to neighbours.
        """
        if self._last_self_embedding is None:
            return None
        return self._last_self_embedding.detach().clone()

    def _ensure_edge_index(self, N: int) -> torch.Tensor:
        """
        Build a default star graph if no edge_index was provided:
        self (0) <-> each neighbour (i), for i in [1..N-1].
        """
        if self.edge_index is not None:
            return self.edge_index
        if N <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        src = []
        dst = []
        for i in range(1, N):
            src.extend([0, i])
            dst.extend([i, 0])
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def _gnn_one(self, self_emb: torch.Tensor, nbr_emb: torch.Tensor) -> torch.Tensor:
        """
        Run the GNN for a single sample.
        self_emb: (D_emb,)
        nbr_emb : (K, D_emb)
        returns : (D_emb,) for node 0
        """
        if nbr_emb.ndim == 1:
            # If K == 1 and SB3 squeezed it
            nbr_emb = nbr_emb.unsqueeze(0)
        x = torch.cat([self_emb.unsqueeze(0), nbr_emb], dim=0)  # (N, D_emb)
        eidx = self._ensure_edge_index(x.size(0))  # (2, E)
        h = self.gnn(x, eidx)  # (N, D_emb)
        return h[0]

    def forward(self, obs) -> torch.Tensor:
        """
        obs is a dict-like with Tensors on SB3's device:
          obs["self_raw"]: (B, F_raw)
          obs["nbr_embed"]: (B, K, D_emb)  (K fixed for this agent)
        """
        self_raw: torch.Tensor = obs["self_raw"].to(self.device)
        nbr_embed: torch.Tensor = obs["nbr_embed"].to(self.device)

        # 1) embed the self raw features
        self_emb = self.self_embed(self_raw)  # (B, D_emb)

        # 2) run per-sample GNN (B small in single-agent setups)
        B = self_emb.size(0)
        out_rows: List[torch.Tensor] = []
        last_self = None
        for b in range(B):
            h0 = self._gnn_one(self_emb[b], nbr_embed[b])  # (D_emb,)
            out_rows.append(h0)
            last_self = h0

        out = torch.stack(out_rows, dim=0)  # (B, D_emb)

        # Cache last self embedding for publishing
        self._last_self_embedding = last_self

        return out  # SB3 expects (B, features_dim)
