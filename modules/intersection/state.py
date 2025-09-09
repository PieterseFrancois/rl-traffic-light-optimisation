# modules/intersection/state.py
"""
Raw state assembly (no learnable embedding).

Design
------
- Reads E2 (laneArea) detectors via a reader.
- If a logical detector maps to multiple physical E2s, we first AGGREGATE RAW
  counts/speeds across those E2s, then NORMALISE the aggregated row.
- Normalisation modes (toggle per-intersection or globally):
    "none"  : keep raw counts
    "per_m" : divide counts by total covered length (veh/m)
    "to_ref": scale counts to a reference length L_ref (veh per L_ref)
- Speed is always a length-weighted mean.

Extractor expects:
{
  "self_raw":  (F_raw,),
  "nbr_embed": (K, D_emb)
}
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
import traci  # E2 length + metrics

FEATURE_KEYS: Tuple[str, ...] = ("halt", "count", "speed")

# -------------------- module-level normalisation defaults --------------------

_DEFAULT_NORM = {"mode": "none", "ref_length_m": 30.0}


def set_detector_reader_norm(norm_mode: str, ref_length_m: float = 30.0) -> None:
    """
    Global switch for E2 normalisation used by build_self_raw/build_matrices
    if an intersection does not provide e2_norm_mode / e2_ref_length_m.
    """
    _DEFAULT_NORM["mode"] = str(norm_mode).lower()
    _DEFAULT_NORM["ref_length_m"] = float(ref_length_m)


# -------------------- E2 helper: length cache --------------------

def _e2_length(det_id: str) -> float:
    cache: Dict[str, float] = _e2_length.__dict__.setdefault("_cache", {})
    if det_id not in cache:
        try:
            cache[det_id] = float(traci.lanearea.getLength(det_id))  # type: ignore
        except Exception:
            cache[det_id] = 1.0
    return cache[det_id]


# -------------------- reader (RAW values, no normalisation) ------------------

def detector_reader(det_id: str) -> Dict[str, float]:
    """
    Returns RAW E2 metrics for 'det_id' (no normalisation):
      - halt  = halting vehicles (veh)
      - count = vehicles present (veh)
      - speed = mean speed (m/s)
    Normalisation is applied AFTER group aggregation.
    """
    try:
        return {
            "halt": float(traci.lanearea.getLastStepHaltingNumber(det_id)),   # type: ignore
            "count": float(traci.lanearea.getLastStepVehicleNumber(det_id)),  # type: ignore
            "speed": float(traci.lanearea.getLastStepMeanSpeed(det_id)),      # type: ignore
        }
    except Exception:
        return {"halt": 0.0, "count": 0.0, "speed": 0.0}


# -------------------- grouping and normalisation --------------------

def _read_one(det_id: str, reader) -> Tuple[Tensor, float]:
    """One physical E2 → (raw metrics tensor, length metres)."""
    raw = reader(det_id) or {}
    vals = [float(raw.get(k, 0.0)) for k in FEATURE_KEYS]
    return torch.tensor(vals, dtype=torch.float32), _e2_length(det_id)


def _aggregate_raw(items: Sequence[Tuple[Tensor, float]]) -> Tuple[Tensor, float]:
    """
    Aggregate physical E2s into one logical RAW row:
      - halt,count: sums
      - speed     : length-weighted mean
      - returns (row_tensor, total_length)
    """
    if not items:
        return torch.zeros(len(FEATURE_KEYS), dtype=torch.float32), 0.0
    mats = torch.stack([t for (t, _) in items], dim=0)  # (N, 3)
    lens = torch.tensor([max(1e-6, L) for (_, L) in items], dtype=torch.float32)  # (N,)
    Ltot = float(lens.sum().item())
    w = lens / lens.sum()
    speed = (mats[:, 2] * w).sum()
    halt = mats[:, 0].sum()
    cnt = mats[:, 1].sum()
    return torch.stack((halt, cnt, speed), dim=0), Ltot


def _normalise_row(row_raw: Tensor, Ltot: float, *, mode: str, L_ref: float) -> Tensor:
    """
    Apply normalisation to an aggregated logical RAW row.
      - "none"  : passthrough
      - "per_m" : divide halt,count by Ltot
      - "to_ref": scale halt,count by L_ref / Ltot
    Speed is unchanged (already length-weighted).
    """
    mode = mode.lower()
    out = row_raw.clone()
    if mode == "per_m":
        scale = 1.0 / max(Ltot, 1e-6)
        out[0] = out[0] * scale
        out[1] = out[1] * scale
    elif mode == "to_ref":
        scale = float(L_ref) / max(Ltot, 1e-6)
        out[0] = out[0] * scale
        out[1] = out[1] * scale
    # "none": unchanged
    return out


def _stack_logical(
    logical_ids: Iterable[str],
    reader,
    groups: Mapping[str, Sequence[str]] | None,
    *,
    mode: str,
    L_ref: float,
) -> Tensor:
    """
    Preserve the order of 'logical_ids'.
    For each logical id:
      - read all physical E2s,
      - aggregate RAW,
      - then normalise the aggregated row with (mode, L_ref).
    """
    ids: List[str] = list(logical_ids)
    if not ids:
        return torch.zeros((0, len(FEATURE_KEYS)), dtype=torch.float32)
    groups = groups or {}
    rows: List[Tensor] = []
    for lid in ids:
        phys = list(groups.get(lid, ()))
        if phys:
            row_raw, Ltot = _aggregate_raw([_read_one(p, reader) for p in phys])
        else:
            row_raw, Ltot = _aggregate_raw([_read_one(lid, reader)])
        rows.append(_normalise_row(row_raw, Ltot, mode=mode, L_ref=L_ref))
    return torch.stack(rows, dim=0)


# -------------------- public API --------------------

def build_self_raw(
    intersection,
    detector_reader,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Build 'self_raw' by flattening logical Din/Dout rows.
    Normalisation mode is chosen as:
      intersection.e2_norm_mode / e2_ref_length_m if present,
      else module defaults configured via set_detector_reader_norm(...).
    """
    cfg = getattr(intersection, "cfg", None)
    groups = getattr(intersection, "detector_groups", None) or (getattr(cfg, "detector_groups", None) if cfg else None)
    mode = str(
        getattr(intersection, "e2_norm_mode", None)
        or (getattr(cfg, "e2_norm_mode", None) if cfg else None)
        or _DEFAULT_NORM["mode"]
    )
    L_ref = float(
        getattr(intersection, "e2_ref_length_m", None)
        or (getattr(cfg, "e2_ref_length_m", None) if cfg else None)
        or _DEFAULT_NORM["ref_length_m"]
    )

    Din = _stack_logical(intersection.detectors_in,  detector_reader, groups, mode=mode, L_ref=L_ref).to(device)
    Dout = _stack_logical(intersection.detectors_out, detector_reader, groups, mode=mode, L_ref=L_ref).to(device)

    parts: List[Tensor] = []
    if Din.numel():
        parts.append(Din.flatten())
    if Dout.numel():
        parts.append(Dout.flatten())

    if isinstance(getattr(intersection, "ctx_vec", None), torch.Tensor):
        parts.append(intersection.ctx_vec.to(device).flatten())
    if isinstance(getattr(intersection, "static_vec", None), torch.Tensor):
        parts.append(intersection.static_vec.to(device).flatten())

    if not parts:
        return torch.zeros(1, dtype=torch.float32, device=device)
    return torch.cat(parts, dim=0)


def build_matrices(
    intersection,
    detector_reader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """Convenience for debugging: return (Din, Dout) matrices on 'device'."""
    cfg = getattr(intersection, "cfg", None)
    groups = getattr(intersection, "detector_groups", None) or (getattr(cfg, "detector_groups", None) if cfg else None)
    mode = str(getattr(intersection, "e2_norm_mode", None) or (getattr(cfg, "e2_norm_mode", None) if cfg else None) or _DEFAULT_NORM["mode"])
    L_ref = float(getattr(intersection, "e2_ref_length_m", None) or (getattr(cfg, "e2_ref_length_m", None) if cfg else None) or _DEFAULT_NORM["ref_length_m"])
    Din  = _stack_logical(intersection.detectors_in,  detector_reader, groups, mode=mode, L_ref=L_ref).to(device)
    Dout = _stack_logical(intersection.detectors_out, detector_reader, groups, mode=mode, L_ref=L_ref).to(device)
    return Din, Dout    
