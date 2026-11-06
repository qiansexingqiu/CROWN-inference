#!/usr/bin/env python3
"""CPU-only checks that do not load checkpoints."""
from __future__ import annotations

import numpy as np

from crown_infer.voxel import fdi_to_sequential
from crown_infer.volume import assemble_feat, jaw_bounds


def test_fdi() -> None:
    labs = np.array([11, 21, 0, 47], dtype=np.int32)
    seq = fdi_to_sequential(labs)
    assert seq.tolist() == [1, 8, 0, 7], seq


def test_feat_shape() -> None:
    rng = np.random.default_rng(0)
    ios = rng.normal(size=(200, 3))
    ddc = ios.copy()
    labs = np.zeros(len(ddc), dtype=np.int32)
    labs[:80] = 11
    labs[80:140] = 21
    from crown_infer.io_mesh import to_sequential

    seq = to_sequential(labs)
    roots = ios[:50] + np.array([0.0, 0.0, -2.0])
    bmin, ext = jaw_bounds(ios, ddc, margin_mm=4.0)
    feat = assemble_feat(
        ios_pts=ios,
        ios_colors=None,
        ddc_pts=ddc,
        ddc_seq_labels=seq,
        root_pred_pts_mm=roots,
        bounds_min=bmin,
        extent=ext,
        grid_size=16,
        rgb_fill=0.0,
    )
    assert feat.shape == (7, 16, 16, 16)
    assert float(feat[1:4].max()) == 0.0
    assert float(feat[0].max()) > 0.0
    assert float(feat[4].max()) > 0.0


if __name__ == "__main__":
    test_fdi()
    test_feat_shape()
    print("selftest ok")
