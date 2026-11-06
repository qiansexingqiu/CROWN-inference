"""Phase-2 root completion and the 7-channel volume for one canonical labeled jaw."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_mesh import read_labels, read_vertices, to_sequential, write_ply_xyz
from .volume import assemble_feat, jaw_bounds


def crowns_by_tooth(verts: np.ndarray, labels: np.ndarray, *, min_pts: int = 100) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for tid in range(1, 15):
        pts = verts[labels == tid]
        if len(pts) >= min_pts:
            out[tid] = pts
    return out


def weights_for_variant(root: Path, variant: str = "base") -> dict[str, Path]:
    """Resolve CROWN-base checkpoints. This release has no CROWN-pro weights."""
    if str(variant).lower() != "base":
        raise ValueError(
            "CROWN-pro weights are not in this package. "
            "Contact the authors: TODO_CONTACT_EMAIL"
        )
    pack = Path(__file__).resolve().parents[1]
    d = {
        "phase3": Path(root) / "CROWN-base" / "phase3_sdf.pt",
        "phase2": Path(root) / "CROWN-base" / "phase2_root.pt",
        "prior": pack / "priors" / "per_tooth_prior.json",
    }
    missing = [str(path) for path in d.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing CROWN-base files:\n  " + "\n  ".join(missing)
        )
    return d


def reconstruct_jaw(
    *,
    jaw: str,
    ios_mesh: Path,
    ddc_mesh: Path,
    labels_path: Path,
    ckpts: dict[str, Path],
    variant: str,
    device,
    out_dir: Path,
    grid_size: int = 128,
    margin_mm: float = 4.0,
    crown_margin_mm: float = 1.5,
    seed: int = 42,
    min_pts: int = 100,
    save_feat: bool = True,
) -> dict:
    from .roots import complete_tooth, denormalize, load_phase2

    ios_pts, ios_col = read_vertices(ios_mesh)
    ddc_pts, _ddc_col = read_vertices(ddc_mesh)
    seq = to_sequential(read_labels(labels_path, len(ddc_pts)))

    if str(variant).lower() != "base":
        raise ValueError(
            "CROWN-pro weights are not in this package. "
            "Contact the authors: TODO_CONTACT_EMAIL"
        )
    phase2, p2_args = load_phase2(ckpts["phase2"], device=device)
    num_in = int(p2_args.get("num_in_points", 4096))
    from .pertooth import load_prior_table, normalize_pertooth_unit_ball

    prior_table = load_prior_table(ckpts["prior"])

    crowns = crowns_by_tooth(ddc_pts, seq, min_pts=min_pts)
    if not crowns:
        raise RuntimeError(f"{jaw}: no teeth with ≥{min_pts} labeled vertices")

    root_mm_list: list[np.ndarray] = []
    per_tooth: list[dict] = []
    for tid, crown_mm in sorted(crowns.items()):
        got = normalize_pertooth_unit_ball(
            crown_mm, jaw=jaw, tooth_id=tid, prior_table=prior_table
        )
        if got is None:
            per_tooth.append({"tooth_id": tid, "status": "frame_failed"})
            continue
        crown_n, _full_n, center, scale = got
        chim_n = complete_tooth(
            phase2,
            crown_n,
            device=device,
            num_in=num_in,
            seed=seed + 1000 + tid,
            scale_mm=float(scale),
            crown_margin_mm=crown_margin_mm,
        )
        chim_mm = denormalize(chim_n, center, float(scale))
        root_mm_list.append(chim_mm)
        per_tooth.append(
            {
                "tooth_id": tid,
                "status": "ok",
                "n_crown": int(len(crown_mm)),
                "n_chimeric": int(len(chim_mm)),
                "scale_mm": float(scale),
            }
        )
    if not root_mm_list:
        raise RuntimeError(f"{jaw}: Phase-2 produced no roots")
    root_all = np.vstack(root_mm_list)
    bounds_min, extent = jaw_bounds(ios_pts, ddc_pts, margin_mm=margin_mm)
    feat = assemble_feat(
        ios_pts=ios_pts,
        ios_colors=ios_col,
        ddc_pts=ddc_pts,
        ddc_seq_labels=seq,
        root_pred_pts_mm=root_all,
        bounds_min=bounds_min,
        extent=extent,
        grid_size=grid_size,
        rgb_fill=0.0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "feat.npz",
        feat=feat.astype(np.float32),
        bounds_min_mm=bounds_min.astype(np.float32),
        extent_mm=extent.astype(np.float32),
    )
    write_ply_xyz(
        out_dir / "predicted_roots_mm.ply",
        root_all,
        np.tile([[255, 80, 60]], (len(root_all), 1)),
    )
    return {
        "jaw": jaw,
        "n_teeth": len(crowns),
        "phase2_arch": p2_args.get("arch"),
        "grid_size": int(grid_size),
        "margin_mm": float(margin_mm),
        "teeth": per_tooth,
        "feat": str(out_dir / "feat.npz"),
    }
