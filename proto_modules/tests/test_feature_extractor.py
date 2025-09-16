# tests/test_features_extractor.py
"""
Features Extractor Test Suite (Independent Agent)

This suite validates the SB3 FeaturesExtractor that embeds the
current intersection's raw features and aggregates neighbour
embeddings via a GATv2 GNN to produce the policy input.

What it tests
-------------
1) Forward shape & determinism
2) Permutation invariance to neighbour ordering (star graph)
3) Graph propagation: 1-hop vs 2-hop influence via custom edge_index
4) Robustness to missing neighbours (zero vectors as placeholders)
5) Gradient flow through the self embedder and GNN
6) Getter for publishing the latest self embedding

Assumptions
-----------
- The extractor class is `NeighbourGNNFeatures` in `modules.features_extractor`.
- Observation format follows:
    obs["self_raw"]:  (B, F_raw)
    obs["nbr_embed"]: (B, K, D_emb)   # fixed K per agent/env
- Node index 0 is the self node; neighbours occupy 1..K
"""

import pytest
import torch
import numpy as np

import gymnasium as gym

# Import the actual extractor under test
from modules.intersection.features_extractor import NeighbourGNNFeatures

from torch_geometric.nn import GATv2Conv  # noqa: F401

# -------------------------
# Helper fixtures/functions
# -------------------------


@pytest.fixture(autouse=True)
def set_seed():
    """Make tests deterministic."""
    torch.manual_seed(1234)
    np.random.seed(1234)


def make_obs_space(F_raw: int, K: int, D_emb: int):
    """
    Build a minimal Dict observation space compatible with the extractor.
    Shapes must match the tensors passed in tests.
    """
    return gym.spaces.Dict(
        {
            "self_raw": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(F_raw,), dtype=np.float32
            ),
            "nbr_embed": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(K, D_emb), dtype=np.float32
            ),
        }
    )


def make_extractor(F_raw=6, K=3, D_emb=8, edge_index=None, device="cpu"):
    """
    Construct a NeighbourGNNFeatures with configurable shapes and an optional edge list.
    By default, uses a star graph (0 <-> i) if edge_index is None.
    """
    obs_space = make_obs_space(F_raw=F_raw, K=K, D_emb=D_emb)
    return NeighbourGNNFeatures(
        observation_space=obs_space,
        self_raw_dim=F_raw,
        embed_dim=D_emb,
        gat_hidden=16,
        gat_heads=2,
        edge_index=edge_index,
        device=torch.device(device),
    )


def make_obs(B=1, F_raw=6, K=3, D_emb=8):
    """
    Create a random observation dict with the required shapes.
    """
    self_raw = torch.randn(B, F_raw)
    nbr_embed = torch.randn(B, K, D_emb)
    return {"self_raw": self_raw, "nbr_embed": nbr_embed}


# -------------
# The test cases
# -------------


def test_forward_shape_and_determinism():
    """
    1) Forward shape & determinism
       - Output shape must be (B, D_emb)
       - Running twice with identical inputs and weights yields identical outputs
    """
    F_raw, K, D = 6, 3, 8
    extr = make_extractor(F_raw=F_raw, K=K, D_emb=D)
    obs = make_obs(B=4, F_raw=F_raw, K=K, D_emb=D)

    out1 = extr(obs)
    out2 = extr(obs)

    assert out1.shape == (4, D)
    assert torch.allclose(
        out1, out2, atol=0.0
    ), "Extractor should be deterministic for identical inputs"


def test_neighbour_permutation_invariance_star_graph():
    """
    2) Permutation invariance to neighbour ordering (star graph)
       With a star graph (self <-> each neighbour), reordering the K neighbours
       should not change the self node's aggregated embedding (up to numerical noise),
       because attention aggregates over the *set* of neighbour messages.
    """
    F_raw, K, D = 5, 4, 8
    B = 2

    # Default extractor uses a star if edge_index is None
    extr = make_extractor(F_raw=F_raw, K=K, D_emb=D)
    obs = make_obs(B=B, F_raw=F_raw, K=K, D_emb=D)

    out_orig = extr(obs)

    # Permute neighbour order consistently in the obs
    perm = torch.randperm(K)
    obs_perm = {
        "self_raw": obs["self_raw"].clone(),
        "nbr_embed": obs["nbr_embed"][:, perm, :].clone(),
    }
    out_perm = extr(obs_perm)

    assert torch.allclose(
        out_orig, out_perm, atol=1e-6
    ), "Self embedding should be invariant to neighbour order (star)"


def test_two_hop_propagation_vs_one_hop():
    """
    3) Graph propagation
       Build two extractors with identical weights:
         A) 2-hop chain: 0 <-> 1 <-> 2 (no 0 <-> 2)
         B) 1-hop star:  0 <-> 1   (node 2 disconnected from 0)
       Changing neighbour-2 embedding should affect A's output (two GAT layers propagate over 2 hops),
       but should NOT affect B's output (neighbour-2 is disconnected from self).
    """
    F_raw, K, D = 4, 2, 8
    B = 1

    # Edge list for (0<->1) and (1<->2) only: a two-hop chain
    e_chain = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # Extractor A: chain graph
    extr_chain = make_extractor(F_raw=F_raw, K=K, D_emb=D, edge_index=e_chain)

    # Extractor B: star with only 0<->1 (leave 2 disconnected from 0)
    e_onehop = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long
    )  # effectively 0<->1 only
    extr_onehop = make_extractor(F_raw=F_raw, K=K, D_emb=D, edge_index=e_onehop)

    # Sync weights so we can compare responses fairly
    extr_onehop.load_state_dict(extr_chain.state_dict())

    # Base obs
    obs = make_obs(B=B, F_raw=F_raw, K=K, D_emb=D)
    base_chain = extr_chain(obs).clone()
    base_onehop = extr_onehop(obs).clone()

    # Perturb neighbour 2 only
    obs_pert = {
        "self_raw": obs["self_raw"].clone(),
        "nbr_embed": obs["nbr_embed"].clone(),
    }
    obs_pert["nbr_embed"][
        :, 1, :
    ] += 0.25  # neighbour index 2 is position 1 (0=self, 1..K neighbours)

    out_chain = extr_chain(obs_pert)
    out_onehop = extr_onehop(obs_pert)

    # In chain, neighbour-2 should influence self via node 1 within two GAT layers
    delta_chain = torch.norm(out_chain - base_chain).item()
    # In one-hop star (with node 2 disconnected from 0), no effect expected
    delta_onehop = torch.norm(out_onehop - base_onehop).item()

    assert (
        delta_chain > 1e-6
    ), "Two-hop change should affect the self embedding in a 2-layer GAT over the chain"
    assert (
        delta_onehop < 1e-8
    ), "Disconnected neighbour should not affect the self embedding"


def test_missing_neighbour_zero_placeholder():
    """
    4) Missing neighbour robustness
       Replacing one neighbour embedding with an all-zeros vector should change the
       output less than replacing it with a random vector (star graph).
    """
    F_raw, K, D = 6, 3, 8
    B = 1
    extr = make_extractor(F_raw=F_raw, K=K, D_emb=D)
    obs = make_obs(B=B, F_raw=F_raw, K=K, D_emb=D)

    base = extr(obs).clone()

    # Replace neighbour 1 with zeros
    obs_zero = {
        "self_raw": obs["self_raw"].clone(),
        "nbr_embed": obs["nbr_embed"].clone(),
    }
    obs_zero["nbr_embed"][:, 0, :] = 0.0
    out_zero = extr(obs_zero)

    # Replace neighbour 1 with a different random vector
    obs_rand = {
        "self_raw": obs["self_raw"].clone(),
        "nbr_embed": obs["nbr_embed"].clone(),
    }
    obs_rand["nbr_embed"][:, 0, :] = torch.randn_like(obs_rand["nbr_embed"][:, 0, :])
    out_rand = extr(obs_rand)

    d_zero = torch.norm(out_zero - base).item()
    d_rand = torch.norm(out_rand - base).item()

    assert (
        d_zero <= d_rand + 1e-6
    ), "Zero placeholder should perturb the embedding no more than a random replacement"


def test_gradient_flow_through_embedder_and_gnn():
    """
    5) Gradient flow
       A dummy scalar loss on the extractor output should backpropagate into both:
         - the self embedder MLP parameters,
         - the GATv2 parameters.
    """
    F_raw, K, D = 5, 2, 8
    extr = make_extractor(F_raw=F_raw, K=K, D_emb=D)
    obs = make_obs(B=3, F_raw=F_raw, K=K, D_emb=D)

    out = extr(obs)  # (B, D)
    loss = out.pow(2).mean()  # simple scalar loss
    loss.backward()

    got_embed_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in extr.self_embed.parameters()
    )
    got_gnn_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in extr.gnn.parameters()
    )

    assert got_embed_grad, "Self embedder did not receive gradients"
    assert got_gnn_grad, "GNN did not receive gradients"


def test_cached_self_embedding_getter():
    """
    6) Getter for latest self embedding
       After a forward pass with batch size 1, the cached self embedding
       should match the returned output row.
    """
    F_raw, K, D = 6, 3, 8
    extr = make_extractor(F_raw=F_raw, K=K, D_emb=D)
    obs = make_obs(B=1, F_raw=F_raw, K=K, D_emb=D)

    out = extr(obs)  # (1, D)
    cached = extr.get_last_self_embedding()
    assert cached is not None
    assert cached.shape == (D,)
    assert torch.allclose(
        out.squeeze(0), cached, atol=0.0
    ), "Cached embedding must equal the last forward output for B=1"


def test_output_dim_consistency_across_self_raw_lengths():
    """
    7) Output dim consistency across different self_raw lengths
       Two agents can have different numbers of lanes (thus different F_raw),
       but the extractor's output embedding size (D_emb) must remain identical.
    """
    D = 8
    # Agent A: fewer lanes -> smaller F_raw
    extr_a = make_extractor(F_raw=4, K=2, D_emb=D, edge_index=None)
    obs_a = make_obs(B=2, F_raw=4, K=2, D_emb=D)

    # Agent B: more lanes -> larger F_raw
    extr_b = make_extractor(F_raw=10, K=2, D_emb=D, edge_index=None)
    obs_b = make_obs(B=3, F_raw=10, K=2, D_emb=D)

    out_a = extr_a(obs_a)
    out_b = extr_b(obs_b)

    assert (
        out_a.shape[-1] == D and out_b.shape[-1] == D
    ), "Final embedding dim must equal D_emb"
    assert out_a.ndim == 2 and out_b.ndim == 2
    # Numerical sanity
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()


def test_output_dim_consistency_across_neighbour_counts():
    """
    8) Output dim consistency across different neighbour counts (K)
       Per-agent K is fixed, but can differ across agents. The final embedding
       size used by the policy must remain D_emb regardless of K.
    """
    F_raw, D = 6, 8

    # Agent A: K=1
    extr_k1 = make_extractor(F_raw=F_raw, K=1, D_emb=D, edge_index=None)
    obs_k1 = make_obs(B=2, F_raw=F_raw, K=1, D_emb=D)
    out_k1 = extr_k1(obs_k1)
    assert out_k1.shape == (2, D)

    # Agent B: K=5
    extr_k5 = make_extractor(F_raw=F_raw, K=5, D_emb=D, edge_index=None)
    obs_k5 = make_obs(B=3, F_raw=F_raw, K=5, D_emb=D)
    out_k5 = extr_k5(obs_k5)
    assert out_k5.shape == (3, D)

    # Numerical sanity
    assert torch.isfinite(out_k1).all() and torch.isfinite(out_k5).all()


def test_different_agents_independent_weights_and_shapes():
    """
    9) Independent agents: differing F_raw and K should not interfere.
       Construct two extractors with different shapes and ensure each forward
       works with its own obs and produces stable outputs.
    """
    # Agent X
    extr_x = make_extractor(F_raw=5, K=3, D_emb=8)
    obs_x = make_obs(B=1, F_raw=5, K=3, D_emb=8)
    out_x = extr_x(obs_x)
    assert out_x.shape == (1, 8)
    assert torch.isfinite(out_x).all()

    # Agent Y
    extr_y = make_extractor(F_raw=9, K=2, D_emb=8)
    obs_y = make_obs(B=4, F_raw=9, K=2, D_emb=8)
    out_y = extr_y(obs_y)
    assert out_y.shape == (4, 8)
    assert torch.isfinite(out_y).all()
