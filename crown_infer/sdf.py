"""Phase-3 nnU-Net SDF inference + marching cubes. Plans path is rewritten locally."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .syspath import BUNDLED_PLANS, add_vendor_nnunet


def load_phase3(checkpoint: Path, *, device: torch.device):
    add_vendor_nnunet()
    from backbone_nnunet_sdf import build_nnunet_sdf_backbone

    try:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(checkpoint, map_location=device)
    hparams = ck.get("args") or {}
    if not isinstance(hparams, dict):
        hparams = {}
    # Checkpoints store an absolute development-machine plans path; always
    # rebuild from the bundled copy in this pack.
    model = build_nnunet_sdf_backbone(BUNDLED_PLANS).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, hparams


@torch.no_grad()
def predict_sdf(
    model,
    feat: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(feat)).float().unsqueeze(0).to(device)
    amp = use_amp and device.type == "cuda"
    if amp:
        try:
            ctx = torch.amp.autocast("cuda")
        except (TypeError, AttributeError):
            ctx = torch.cuda.amp.autocast()
        with ctx:
            pred = model(x)
    else:
        pred = model(x)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    return pred[0, 0].detach().float().cpu().numpy()


def volume_to_world_points(
    verts_idx: np.ndarray, bounds_min: np.ndarray, extent: np.ndarray, grid_size: int
) -> np.ndarray:
    sp = extent / float(grid_size)
    return bounds_min + verts_idx * sp


def export_marching_cubes_obj(
    vol: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    obj_path: Path,
    *,
    level: float = 0.0,
) -> float:
    from skimage.measure import marching_cubes

    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    iso = float(level)
    if not (vmin <= iso <= vmax):
        if not (vmin < vmax):
            raise ValueError(f"SDF is approximately constant (min=max={vmin:.6g})")
        iso = 0.5 * (vmin + vmax)
    verts, faces, _n, _v = marching_cubes(vol, level=iso, spacing=(1, 1, 1))
    g = int(vol.shape[0])
    verts_mm = volume_to_world_points(
        verts,
        np.asarray(bounds_min, dtype=np.float64),
        np.asarray(extent, dtype=np.float64),
        g,
    )
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(obj_path, "w", encoding="utf-8") as f:
        for x, y, z in verts_mm:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return iso
