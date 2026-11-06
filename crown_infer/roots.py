"""Phase-2 crown → full-tooth completion and chimeric root extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .syspath import add_vendor_mint


def sample_pts(pts: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    idx = rng.integers(0, len(pts), size=n)
    return pts[idx]


def _min_dist_to_set(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(ref, dtype=np.float64))
        d, _ = tree.query(np.asarray(query, dtype=np.float64), k=1)
        return np.asarray(d, dtype=np.float64)
    except ImportError:
        q = np.asarray(query, dtype=np.float64)
        r = np.asarray(ref, dtype=np.float64)
        dmin = np.full(len(q), np.inf, dtype=np.float64)
        bs = 512
        for i in range(0, len(q), bs):
            diff = q[i : i + bs, None, :] - r[None, :, :]
            dmin[i : i + len(diff)] = np.linalg.norm(diff, axis=2).min(axis=1)
        return dmin


def chimeric_crown_plus_pred_root(
    crown_np: np.ndarray,
    pred_np: np.ndarray,
    scale_radius_mm: float,
    crown_margin_mm: float,
) -> np.ndarray:
    eps = crown_margin_mm / max(float(scale_radius_mm), 1e-6)
    d = _min_dist_to_set(pred_np, crown_np)
    root_m = d > eps
    if root_m.sum() == 0:
        return np.asarray(crown_np).copy()
    return np.vstack([crown_np, pred_np[root_m]])


def denormalize(pts_n: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(pts_n, dtype=np.float64) * float(scale) + np.asarray(
        center, dtype=np.float64
    )


def forward_fine_batched(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def build_phase2_model(state: dict, sd_args: dict) -> nn.Module:
    add_vendor_mint()
    from model_point_completion import CoarseToFineFoldingNet, FoldingNetCompletion, PointCompletionPCN

    num_out = int(sd_args.get("num_out_points", 4096))
    dh = sd_args.get("decoder_hidden")
    if dh is not None:
        decoder_hidden = int(dh)
    elif "coarse_mlp.0.weight" in state:
        decoder_hidden = int(state["coarse_mlp.0.weight"].shape[0])
    else:
        decoder_hidden = 2048

    arch = str(sd_args.get("arch", "auto"))
    if arch == "auto" or arch not in ("pcn", "folding", "coarse_fine"):
        if "coarse_mlp.4.weight" in state:
            arch = "coarse_fine"
        elif "fold1.0.weight" in state:
            arch = "folding"
        else:
            arch = "pcn"

    if arch == "coarse_fine":
        w4 = state.get("coarse_mlp.4.weight")
        nc = int(sd_args.get("coarse_num", w4.shape[0] // 3 if w4 is not None else 256))
        k = num_out // max(nc, 1)
        pg = int(sd_args.get("patch_grid", int(round(k**0.5))))
        if nc * pg * pg != num_out:
            pg = int(round(k**0.5))
        return CoarseToFineFoldingNet(
            latent_dim=1024,
            num_output_points=num_out,
            decoder_hidden=decoder_hidden,
            num_coarse=nc,
            patch_grid_size=pg,
        )
    if arch == "folding":
        return FoldingNetCompletion(
            latent_dim=1024,
            num_output_points=num_out,
            decoder_hidden=decoder_hidden,
        )
    return PointCompletionPCN(
        latent_dim=1024,
        num_output_points=num_out,
        decoder_hidden=decoder_hidden,
    )


def load_phase2(checkpoint: Path, *, device: torch.device) -> tuple[nn.Module, dict]:
    try:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(checkpoint, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd_args = ck.get("args") if isinstance(ck, dict) else {}
    if not isinstance(sd_args, dict):
        sd_args = {}
    model = build_phase2_model(state, sd_args).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, sd_args


@torch.no_grad()
def complete_tooth(
    model: nn.Module,
    crown_n: np.ndarray,
    *,
    device: torch.device,
    num_in: int,
    seed: int,
    scale_mm: float,
    crown_margin_mm: float = 1.5,
) -> np.ndarray:
    """Return chimeric full-tooth points in the unit-ball frame."""
    crown = sample_pts(crown_n, num_in, seed)
    x = torch.from_numpy(crown.T.copy()).unsqueeze(0).to(device)
    pred = forward_fine_batched(model, x)[0].detach().cpu().numpy()
    return chimeric_crown_plus_pred_root(crown, pred, scale_mm, crown_margin_mm)
