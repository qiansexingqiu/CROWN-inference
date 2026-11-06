"""RotationNet canonicalization of a segmented jaw (crown_web ios_only_canonicalize)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from .syspath import add_vendor_dms, dms_model_root


def _rotation_matrix_torch(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if torch.sqrt(torch.dot(axis, axis)) < 1e-8:
        return torch.eye(3, device=axis.device, dtype=torch.float32)
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        device=axis.device,
        dtype=torch.float32,
    )


def load_rotation_model(model_root: Path, device: torch.device) -> torch.nn.Module:
    add_vendor_dms()
    from src.models.pct_models import Pct  # noqa: E402

    ckpt = Path(model_root) / "model_rotation"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = Pct({}).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def predict_stage0_rotation(
    ios_ply: Path, model: torch.nn.Module, device: torch.device
) -> dict:
    add_vendor_dms()
    from src.data.TeethInferDataset import TeethInferDataset  # noqa: E402

    infer_set = TeethInferDataset(str(ios_ply), device=device)
    with torch.no_grad():
        data_tensor = infer_set.get_data_tensor(False, four_k=True).to(device)
        pred_axis_up, pred_axis_forward = model(data_tensor[:, :, 0:6].permute(0, 2, 1))
        pred_axis_up = pred_axis_up[0]
        pred_axis_forward = pred_axis_forward[0]
        standard_ax_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        standard_ax_forward = torch.tensor([0.0, -1.0, 0.0], device=device)
        if torch.dot(pred_axis_up, standard_ax_up) < 0.99:
            r_ax_up = torch.nn.functional.normalize(
                torch.cross(pred_axis_up, standard_ax_up), p=2, dim=-1
            )
            r_angle_up = torch.arccos(
                torch.dot(pred_axis_up, standard_ax_up)
                / (torch.norm(pred_axis_up) * torch.norm(standard_ax_up))
            )
            r_up = _rotation_matrix_torch(r_ax_up, r_angle_up).to(device)
            pred_axis_forward = torch.matmul(r_up, pred_axis_forward.unsqueeze(1)).squeeze()
        else:
            r_up = torch.eye(3, device=device)
        pred_axis_forward[-1] = 0
        if torch.dot(pred_axis_forward, standard_ax_forward) < 0.99:
            r_ax_forward = torch.nn.functional.normalize(
                torch.cross(pred_axis_forward, standard_ax_forward), p=2, dim=-1
            )
            r_angle_forward = torch.arccos(
                torch.dot(pred_axis_forward, standard_ax_forward)
                / (torch.norm(pred_axis_forward) * torch.norm(standard_ax_forward))
            )
            r_forward = _rotation_matrix_torch(r_ax_forward, r_angle_forward).to(device)
        else:
            r_forward = torch.eye(3, device=device)
        rot_stage0 = torch.matmul(r_forward, r_up).detach().cpu().numpy().astype(np.float64)

    r_pca = np.asarray(infer_set.rot_matrix_pca, dtype=np.float64)
    c_pca = np.asarray(infer_set.c, dtype=np.float64)
    origin_pts = np.asarray(infer_set.origin_data[:, 0:3], dtype=np.float64)
    r_total = rot_stage0 @ r_pca
    t_total = c_pca - rot_stage0 @ c_pca
    can_pts = origin_pts @ r_total.T + t_total
    return {
        "r_total": r_total,
        "t_total": t_total,
        "centroid_after": can_pts.mean(axis=0),
    }


def build_transform(info: dict, *, center_to_origin: bool = True) -> np.ndarray:
    r_total = info["r_total"]
    t_total = np.asarray(info["t_total"], dtype=np.float64).copy()
    if center_to_origin:
        t_total = t_total - info["centroid_after"]
    t4 = np.eye(4, dtype=np.float64)
    t4[:3, :3] = r_total
    t4[:3, 3] = t_total
    return t4


def transform_off_vertices(src: Path, dst: Path, transform: np.ndarray) -> None:
    with open(src, encoding="utf-8") as f:
        header = f.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise ValueError(f"unsupported OFF header: {src} -> {header!r}")
        counts = f.readline().rstrip("\n")
        nv, nf, ne = map(int, counts.split())
        verts = [f.readline().rstrip("\n") for _ in range(nv)]
        faces = [f.readline().rstrip("\n") for _ in range(nf)]
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out_verts = []
    for line in verts:
        parts = line.split()
        xyz = np.array(list(map(float, parts[:3])), dtype=np.float64)
        xyz_t = xyz @ rot.T + trans
        tail = parts[3:]
        xyz_str = f"{xyz_t[0]:.9f} {xyz_t[1]:.9f} {xyz_t[2]:.9f}"
        out_verts.append(xyz_str if not tail else xyz_str + " " + " ".join(tail))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(f"{nv} {nf} {ne}\n")
        for line in out_verts:
            f.write(line + "\n")
        for line in faces:
            f.write(line + "\n")


def canonicalize_jaw(
    *,
    ios_ply: Path,
    ddc_off: Path,
    ddc_label: Path,
    out_dir: Path,
    jaw: str,
    weights: Path,
    device: torch.device,
    model: torch.nn.Module | None = None,
) -> dict[str, Path]:
    """Write canonical `{jaw}_ios.ply`, `{jaw}_ddc.off`, `{jaw}_label.txt`."""
    if model is None:
        model = load_rotation_model(dms_model_root(weights), device)
    info = predict_stage0_rotation(ios_ply, model, device)
    t4 = build_transform(info, center_to_origin=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ios = out_dir / f"{jaw}_ios.ply"
    loaded = trimesh.load(str(ios_ply), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    loaded.apply_transform(t4)
    loaded.export(out_ios)
    out_off = out_dir / f"{jaw}_ddc.off"
    transform_off_vertices(ddc_off, out_off, t4)
    out_lab = out_dir / f"{jaw}_label.txt"
    out_lab.write_bytes(Path(ddc_label).read_bytes())
    (out_dir / f"{jaw}_canonical_meta.json").write_text(
        json.dumps({"jaw": jaw, "transform_4x4": t4.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"ios": out_ios, "ddc": out_off, "labels": out_lab}
