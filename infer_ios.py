#!/usr/bin/env python3
"""IOS-only CROWN-base inference.

Input: raw intraoral mesh(es). Output per jaw: bone_mc.obj + pred_sdf.npz.

  python infer_ios.py --upper upper.ply --lower lower.ply \\
      --weights /path/to/weights --out ./out_case

Segmentation and RotationNet run in conda env ``3dpoint``; the SDF backbone
runs in ``nnunetv2``. Override with CROWN_ENV_3D / CROWN_ENV_NN or
CROWN_SKIP_CONDA=1. This entry point loads CROWN-base only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from crown_infer.condaenv import env_3d, env_nn, run_python
from crown_infer.reconstruct import weights_for_variant
from crown_infer.syspath import default_weights_dir


def _device(name: str):
    import torch

    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CROWN IOS-only alveolar-bone inference (raw IOS → bone mesh)"
    )
    p.add_argument("--weights", type=Path, default=None, help="directory that contains CROWN-base/")
    p.add_argument("--upper", type=Path, default=None, help="raw upper IOS mesh")
    p.add_argument("--lower", type=Path, default=None, help="raw lower IOS mesh")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case-id", type=str, default="Case001")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--margin-mm", type=float, default=4.0)
    p.add_argument("--crown-margin-mm", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stage",
        choices=("all", "preprocess", "reconstruct", "phase3"),
        default="all",
        help="internal staging for the two conda environments",
    )
    p.add_argument("--skip-segment", action="store_true", help="use --mesh/--labels instead of raw IOS")
    p.add_argument("--jaw", choices=("upper", "lower"), default=None)
    p.add_argument("--mesh", type=Path, default=None, help="with --skip-segment: labeled mesh")
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--ios-mesh", type=Path, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def _jaws_from_args(args: argparse.Namespace) -> list[str]:
    jaws = []
    if args.upper:
        jaws.append("upper")
    if args.lower:
        jaws.append("lower")
    if args.skip_segment and args.jaw:
        jaws = [args.jaw]
    return jaws


def stage_preprocess(args: argparse.Namespace, wroot: Path, jaws: list[str]) -> None:
    from crown_infer.canonicalize import canonicalize_jaw, load_rotation_model
    from crown_infer.segment import prepare_case_jaw, run_segmentation
    from crown_infer.syspath import dms_model_root

    out = Path(args.out).expanduser().resolve()
    case_id = args.case_id
    data_root = out / "_work"
    case_dir = data_root / case_id
    canon_dir = out / "_canonical"
    device = _device(args.device)

    for jaw in jaws:
        src = args.upper if jaw == "upper" else args.lower
        prepare_case_jaw(case_dir, case_id, jaw, Path(src).expanduser().resolve())
    run_segmentation(data_root=data_root, case_id=case_id, jaws=jaws, weights=wroot)
    rot = load_rotation_model(dms_model_root(wroot), device)
    for jaw in jaws:
        cap = "Upper" if jaw == "upper" else "Lower"
        canonicalize_jaw(
            ios_ply=case_dir / f"{case_id}_{cap}JawScan.ply",
            ddc_off=case_dir / "oral scan seg" / f"{case_id}_DDC_{cap}.off",
            ddc_label=case_dir / "oral scan seg" / f"{case_id}_DDC_{cap}_label.txt",
            out_dir=canon_dir,
            jaw=jaw,
            weights=wroot,
            device=device,
            model=rot,
        )
    print(json.dumps({"preprocess": str(canon_dir), "jaws": jaws}, indent=2))


def stage_reconstruct(args: argparse.Namespace, wroot: Path, jaws: list[str]) -> None:
    from crown_infer.reconstruct import reconstruct_jaw

    out = Path(args.out).expanduser().resolve()
    device = _device(args.device)
    ckpts = weights_for_variant(wroot, "base")
    for jaw in jaws:
        if args.skip_segment:
            ios = Path(args.ios_mesh or args.mesh).expanduser().resolve()
            ddc = Path(args.mesh).expanduser().resolve()
            labels = Path(args.labels).expanduser().resolve()
        else:
            canon = out / "_canonical"
            ios = canon / f"{jaw}_ios.ply"
            ddc = canon / f"{jaw}_ddc.off"
            labels = canon / f"{jaw}_label.txt"
        meta = reconstruct_jaw(
            jaw=jaw,
            ios_mesh=ios,
            ddc_mesh=ddc,
            labels_path=labels,
            ckpts=ckpts,
            variant="base",
            device=device,
            out_dir=out / jaw,
            grid_size=args.grid_size,
            margin_mm=args.margin_mm,
            crown_margin_mm=args.crown_margin_mm,
            seed=args.seed,
        )
        (out / jaw / "reconstruct_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"reconstruct": jaw, "feat": meta["feat"]}, indent=2))


def stage_phase3(args: argparse.Namespace, wroot: Path, jaws: list[str]) -> None:
    import numpy as np

    from crown_infer.sdf import export_marching_cubes_obj, load_phase3, predict_sdf

    out = Path(args.out).expanduser().resolve()
    device = _device(args.device)
    ckpts = weights_for_variant(wroot, "base")
    model, hparams = load_phase3(ckpts["phase3"], device=device)
    results = []
    for jaw in jaws:
        jaw_dir = out / jaw
        feat_path = jaw_dir / "feat.npz"
        z = np.load(feat_path)
        feat = z["feat"]
        bounds_min = z["bounds_min_mm"].astype(np.float64)
        extent = z["extent_mm"].astype(np.float64)
        pred = predict_sdf(model, feat, device=device, use_amp=not args.no_amp)
        np.savez_compressed(
            jaw_dir / "pred_sdf.npz",
            pred_sdf=pred.astype(np.float32),
            bounds_min_mm=bounds_min.astype(np.float32),
            extent_mm=extent.astype(np.float32),
        )
        iso = export_marching_cubes_obj(pred, bounds_min, extent, jaw_dir / "bone_mc.obj")
        rec = {}
        rec_path = jaw_dir / "reconstruct_meta.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        meta = {
            "case_id": args.case_id,
            "jaw": jaw,
            "variant": "base",
            "marching_cubes_iso": float(iso),
            "sdf_min": float(pred.min()),
            "sdf_max": float(pred.max()),
            "backbone": hparams.get("backbone"),
            "device": str(device),
            "note": (
                "IOS-only reconstruction. Not a radiographic substitute and not "
                "for clinical use."
            ),
            **{k: rec[k] for k in ("n_teeth", "phase2_arch", "teeth") if k in rec},
        }
        (jaw_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        results.append({"jaw": jaw, "bone": str(jaw_dir / "bone_mc.obj"), "sdf": str(jaw_dir / "pred_sdf.npz"), "iso": iso})
    print(json.dumps({"phase3": results}, indent=2))


def _reinvoke(stage: str, args: argparse.Namespace, env: str) -> None:
    cmd = [
        str(_PACK_ROOT / "infer_ios.py"),
        "--stage",
        stage,
        "--weights",
        str(Path(args.weights).resolve() if args.weights else default_weights_dir()),
        "--out",
        str(Path(args.out).resolve()),
        "--case-id",
        args.case_id,
        "--device",
        args.device,
        "--grid-size",
        str(args.grid_size),
        "--margin-mm",
        str(args.margin_mm),
        "--crown-margin-mm",
        str(args.crown_margin_mm),
        "--seed",
        str(args.seed),
    ]
    if args.upper:
        cmd.extend(["--upper", str(Path(args.upper).resolve())])
    if args.lower:
        cmd.extend(["--lower", str(Path(args.lower).resolve())])
    if args.skip_segment:
        cmd.append("--skip-segment")
        if args.jaw:
            cmd.extend(["--jaw", args.jaw])
        if args.mesh:
            cmd.extend(["--mesh", str(Path(args.mesh).resolve())])
        if args.labels:
            cmd.extend(["--labels", str(Path(args.labels).resolve())])
        if args.ios_mesh:
            cmd.extend(["--ios-mesh", str(Path(args.ios_mesh).resolve())])
    if args.no_amp:
        cmd.append("--no-amp")
    run_python(env, cmd, cwd=_PACK_ROOT)


def main() -> None:
    args = parse_args()
    wroot = args.weights or default_weights_dir()
    if wroot is None:
        raise SystemExit("pass --weights /path/to/directory/containing/CROWN-base")
    args.weights = Path(wroot).expanduser().resolve()
    jaws = _jaws_from_args(args)
    if not jaws:
        raise SystemExit("provide --upper and/or --lower (raw IOS), or --skip-segment --jaw --mesh --labels")

    if args.stage == "preprocess":
        stage_preprocess(args, args.weights, jaws)
        return
    if args.stage == "reconstruct":
        stage_reconstruct(args, args.weights, jaws)
        return
    if args.stage == "phase3":
        stage_phase3(args, args.weights, jaws)
        return

    skip_conda = os.environ.get("CROWN_SKIP_CONDA") == "1"
    if skip_conda:
        if not args.skip_segment:
            stage_preprocess(args, args.weights, jaws)
        stage_reconstruct(args, args.weights, jaws)
        stage_phase3(args, args.weights, jaws)
        return

    if not args.skip_segment:
        _reinvoke("preprocess", args, env_3d())
    _reinvoke("reconstruct", args, env_3d())
    _reinvoke("phase3", args, env_nn())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
