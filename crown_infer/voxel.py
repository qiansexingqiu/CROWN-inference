"""7-channel voxel helpers matching the paper IOS-only volume assembler.

Channel layout (feat, 7×G×G×G):
  0  IOS log-density occupancy
  1–3 RGB (paper checkpoints use norgb: filled with 0)
  4  RGSP predicted-root log-density
  5  dominant sequential tooth id / 14
  6  inter-tooth boundary

Tooth-id voxelization applies the same FDI→sequential map as the training
assembler, including the second map inside ``accumulate_tooth_channels``.
"""
from __future__ import annotations

import numpy as np


def fdi_to_sequential(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.int32)
    for i, lab in enumerate(labels):
        if lab == 0:
            continue
        q, p = divmod(int(lab), 10)
        if p < 1 or p > 7:
            continue
        out[i] = p if q in (1, 4) else p + 7
    return out


def mask_points_in_box(
    pts: np.ndarray, bounds_min: np.ndarray, extent: np.ndarray
) -> np.ndarray:
    hi = bounds_min + extent
    return np.all((pts >= bounds_min) & (pts <= hi), axis=1)


def accumulate_ios_to_grid(
    points: np.ndarray,
    colors: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    g = grid_size
    idx = np.floor((points - bounds_min) / (extent + 1e-9) * g).astype(np.int64)
    idx = np.clip(idx, 0, g - 1)
    linear = idx[:, 0] * (g * g) + idx[:, 1] * g + idx[:, 2]
    count = np.zeros((g, g, g), dtype=np.float32)
    np.add.at(count.ravel(), linear, 1.0)
    rgb_sum = np.zeros((g, g, g, 3), dtype=np.float32)
    for c in range(3):
        np.add.at(rgb_sum[..., c].ravel(), linear, colors[:, c])
    return count, rgb_sum


def accumulate_tooth_channels(
    ddc_verts: np.ndarray,
    labels_raw: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tooth_id, tooth_density, tooth_boundary), each (G,G,G)."""
    g = grid_size
    total = g * g * g
    tooth_mask = labels_raw > 0
    pts = ddc_verts[tooth_mask]
    labs_raw = labels_raw[tooth_mask]
    if len(pts) == 0:
        z = np.zeros((g, g, g), dtype=np.float32)
        return z.copy(), z.copy(), z.copy()

    idx = np.floor((pts - bounds_min) / (extent + 1e-9) * g).astype(np.int64)
    idx = np.clip(idx, 0, g - 1)
    linear = idx[:, 0] * (g * g) + idx[:, 1] * g + idx[:, 2]

    count = np.zeros(total, dtype=np.float32)
    np.add.at(count, linear, 1.0)
    cmax = float(count.max()) + 1e-6
    tooth_density = (np.log1p(count) / np.log1p(cmax)).reshape(g, g, g)

    seq = fdi_to_sequential(labs_raw)
    unique_seq = np.unique(seq)
    counts_per_cls = np.zeros((len(unique_seq), total), dtype=np.int32)
    for i, sl in enumerate(unique_seq):
        np.add.at(counts_per_cls[i], linear[seq == sl], 1)

    best_idx = counts_per_cls.argmax(axis=0)
    best_cnt = counts_per_cls.max(axis=0)
    dom = np.zeros(total, dtype=np.int32)
    for i, sl in enumerate(unique_seq):
        dom[best_idx == i] = sl
    dom[best_cnt == 0] = 0
    dom = dom.reshape(g, g, g)
    tooth_id = dom.astype(np.float32) / 14.0

    boundary = np.zeros((g, g, g), dtype=np.float32)
    for ax in range(3):
        fwd = np.roll(dom, -1, axis=ax)
        bwd = np.roll(dom, 1, axis=ax)
        boundary = np.maximum(
            boundary, ((dom > 0) & (fwd > 0) & (dom != fwd)).astype(np.float32)
        )
        boundary = np.maximum(
            boundary, ((dom > 0) & (bwd > 0) & (dom != bwd)).astype(np.float32)
        )
    return tooth_id, tooth_density, boundary


def accumulate_density_channel(
    points: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    colors = np.ones((len(points), 3), dtype=np.float64) * 0.5
    count, _ = accumulate_ios_to_grid(points, colors, bounds_min, extent, grid_size)
    cmax = float(count.max()) + 1e-6
    return (np.log1p(count) / np.log1p(cmax)).astype(np.float32)
