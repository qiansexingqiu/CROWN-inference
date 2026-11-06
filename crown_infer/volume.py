"""Assemble the 7-channel IOS-only input volume (no CT, no GT bone)."""
from __future__ import annotations

import numpy as np

from .voxel import (
    accumulate_density_channel,
    accumulate_ios_to_grid,
    accumulate_tooth_channels,
    mask_points_in_box,
)


def jaw_bounds(
    ios_pts: np.ndarray,
    ddc_pts: np.ndarray,
    *,
    margin_mm: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    all_pts = np.vstack(
        [np.asarray(ios_pts, dtype=np.float64), np.asarray(ddc_pts, dtype=np.float64)]
    )
    m = float(margin_mm)
    bounds_min = all_pts.min(axis=0) - m
    extent = (all_pts.max(axis=0) + m) - bounds_min
    return bounds_min.astype(np.float64), extent.astype(np.float64)


def assemble_feat(
    *,
    ios_pts: np.ndarray,
    ios_colors: np.ndarray | None,
    ddc_pts: np.ndarray,
    ddc_seq_labels: np.ndarray,
    root_pred_pts_mm: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    grid_size: int = 128,
    rgb_fill: float | None = 0.0,
) -> np.ndarray:
    """Return feat (7, G, G, G). rgb_fill=0 matches the paper norgb checkpoints."""
    g = int(grid_size)
    bmin = np.asarray(bounds_min, dtype=np.float64)
    ext = np.asarray(extent, dtype=np.float64)

    m_ios = mask_points_in_box(ios_pts, bmin, ext)
    pts_i = np.asarray(ios_pts[m_ios], dtype=np.float64)
    if ios_colors is not None and len(ios_colors) == len(ios_pts):
        col_i = np.asarray(ios_colors[m_ios], dtype=np.float64)
    else:
        col_i = np.ones((len(pts_i), 3), dtype=np.float64) * 0.5

    count, rgb_sum = accumulate_ios_to_grid(pts_i, col_i, bmin, ext, g)
    mask = count > 0
    cmax = float(count.max()) + 1e-6
    ios_density = (np.log1p(count) / np.log1p(cmax)).astype(np.float32)
    rgb = np.zeros((3, g, g, g), dtype=np.float32)
    for c in range(3):
        rgb[c] = np.where(mask, rgb_sum[..., c] / (count + 1e-6), 0.0).astype(
            np.float32
        )

    m_ddc = mask_points_in_box(ddc_pts, bmin, ext)
    tooth_id_vol, _dens, boundary_vol = accumulate_tooth_channels(
        np.asarray(ddc_pts[m_ddc], dtype=np.float64),
        np.asarray(ddc_seq_labels[m_ddc], dtype=np.int32),
        bmin,
        ext,
        g,
    )

    m_r = mask_points_in_box(root_pred_pts_mm, bmin, ext)
    root_ch4 = accumulate_density_channel(
        np.asarray(root_pred_pts_mm[m_r], dtype=np.float64), bmin, ext, g
    )

    feat = np.zeros((7, g, g, g), dtype=np.float32)
    feat[0] = ios_density
    feat[1:4] = rgb
    feat[4] = root_ch4
    feat[5] = tooth_id_vol
    feat[6] = boundary_vol
    if rgb_fill is not None:
        feat[1:4] = float(rgb_fill)
    return feat
