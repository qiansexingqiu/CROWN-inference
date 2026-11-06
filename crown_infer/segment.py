"""DentalModelSegmentation on a raw IOS mesh (same role as crown_web segment_ios)."""
from __future__ import annotations

from pathlib import Path

from .syspath import add_vendor_dms, dms_model_root


def _as_trimesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = tuple(mesh.geometry.values())
        if not geoms:
            raise ValueError(f"empty scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def prepare_case_jaw(case_dir: Path, case_id: str, jaw: str, mesh_path: Path) -> Path:
    """Write `{case}_DDC_{Upper|Lower}.off` and `{case}_{Upper|Lower}JawScan.ply`."""
    if jaw == "upper":
        cap = "Upper"
    elif jaw == "lower":
        cap = "Lower"
    else:
        raise ValueError(jaw)
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh = _as_trimesh(mesh_path)
    ply_path = case_dir / f"{case_id}_{cap}JawScan.ply"
    off_path = case_dir / f"{case_id}_DDC_{cap}.off"
    mesh.export(ply_path)
    mesh.export(off_path)
    return off_path


def run_segmentation(
    *,
    data_root: Path,
    case_id: str,
    jaws: list[str],
    weights: Path,
) -> dict[str, Path]:
    """Run DMS. Returns jaw -> label txt under `{case}/oral scan seg/`."""
    add_vendor_dms()
    import inference as dms_inference  # noqa: E402

    model_root = dms_model_root(weights)
    if not model_root.is_dir():
        raise FileNotFoundError(
            f"DMS weights missing: {model_root}. Expected third_party/DentalModelSegmentation/"
        )
    case_dir = Path(data_root) / case_id
    out_seg = case_dir / "oral scan seg"
    out_seg.mkdir(parents=True, exist_ok=True)
    labels: dict[str, Path] = {}
    for jaw in jaws:
        cap = "Upper" if jaw == "upper" else "Lower"
        off_path = case_dir / f"{case_id}_DDC_{cap}.off"
        if not off_path.is_file():
            raise FileNotFoundError(off_path)
        dms_inference.inference(
            str(off_path), str(out_seg), str(model_root), rotation=True, debug_vis=False
        )
        lab = out_seg / f"{case_id}_DDC_{cap}_label.txt"
        labeled_off = out_seg / f"{case_id}_DDC_{cap}.off"
        if not lab.is_file() or not labeled_off.is_file():
            raise RuntimeError(f"segmentation failed for {jaw}: missing {labeled_off} or {lab}")
        labels[jaw] = lab
    return labels
