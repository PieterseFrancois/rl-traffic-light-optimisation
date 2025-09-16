from typing import Sequence, Tuple
import torch

Edge = Tuple[str, str]


def edge_index_from_edge_list(
    self_id: str,
    neighbour_ids: Sequence[str],
    edges: Sequence[Edge],
    *,
    add_self_loops: bool = True,
    symmetric: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    idx = {self_id: 0, **{nid: i + 1 for i, nid in enumerate(neighbour_ids)}}
    src, dst = [], []

    def add(u, v):
        if u in idx and v in idx:
            src.append(idx[u])
            dst.append(idx[v])

    for u, v in edges:
        add(u, v)
        if symmetric:
            add(v, u)
    if add_self_loops:
        for i in range(len(idx)):
            src.append(i)
            dst.append(i)
    if not src:  # fallback to star if you passed an empty list
        for j in range(1, len(idx)):
            src += [0, j]
            dst += [j, 0]
    return torch.tensor([src, dst], dtype=torch.long, device=device)
