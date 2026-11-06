"""Heuristic per-tooth frame used by CROWN-base root completion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_APICAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def prior_key(jaw: str, tooth_id: int) -> str:
    return f"{jaw.lower()}_t{int(tooth_id):02d}"


def load_prior_table(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scale_for_crown(g: dict, crown_mm: np.ndarray, *, scale_mode: str = "median") -> float:
    if scale_mode == "median":
        return float(g["scale_median_mm"])
    centroid = np.asarray(crown_mm, dtype=np.float64).mean(axis=0)
    crown_r = float(
        np.linalg.norm(np.asarray(crown_mm, dtype=np.float64) - centroid, axis=1).max()
    )
    if crown_r < 1e-6:
        return float(g["scale_median_mm"])
    ratio = g.get("scale_over_crown_r_mean")
    if ratio is None:
        crown_r_med = float(g.get("crown_r_median_mm", crown_r))
        ratio = float(g["scale_median_mm"]) / max(crown_r_med, 1e-6)
    return float(ratio) * crown_r


def build_root_proxy(crown: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    proj = crown @ _APICAL
    cervical = crown[proj <= np.quantile(proj, 0.35)]
    base = cervical.mean(axis=0) if len(cervical) else crown.mean(axis=0)
    tip = center + _APICAL * scale * 0.98
    t = np.linspace(0.15, 1.0, 512)[:, None]
    return base * (1.0 - t) + tip * t


def normalize_pertooth_unit_ball(
    crown: np.ndarray,
    *,
    jaw: str,
    tooth_id: int,
    prior_table: dict,
    scale_mode: str = "crown_r_ratio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    crown = np.asarray(crown, dtype=np.float64)
    if len(crown) < 4:
        return None
    groups = prior_table.get("groups", prior_table)
    g = groups.get(prior_key(jaw, tooth_id))
    if g is None:
        return None
    offset = np.asarray(g["offset_mean_mm"], dtype=np.float64).reshape(3)
    centroid = crown.mean(axis=0)
    center = centroid + offset
    scale = scale_for_crown(g, crown, scale_mode=scale_mode)
    if scale < 1e-7:
        return None
    root_pts = build_root_proxy(crown, center, scale)
    full_pts = np.vstack([crown, root_pts])
    inv = 1.0 / scale
    crown_n = ((crown - center) * inv).astype(np.float32)
    full_n = ((full_pts - center) * inv).astype(np.float32)
    return crown_n, full_n, center.astype(np.float32), float(scale)
