# CROWN-base inference

Read an intraoral scan (IOS) mesh and write a whole-arch alveolar-bone signed
distance field (SDF) plus a marching-cubes surface. This package runs
**CROWN-base** only. It does not train, and it does not include imaging.

This reconstruction is not a radiograph and is not for clinical use.

## CROWN-pro weights

CROWN-pro weights: contact the authors (`TODO_CONTACT_EMAIL`).

They are not in this directory and not in `CROWN-base-weights.tar.gz`.

## Install

The paper stack splits work across two conda environments:

| Stage | Environment | What it runs |
|---|---|---|
| Segmentation, canonical frame, Phase-2 roots | `3dpoint` (`CROWN_ENV_3D`) | DentalModelSegmentation, RotationNet, FoldingNet |
| Phase-3 bone SDF | `nnunetv2` (`CROWN_ENV_NN`) | nnU-Net PlainConv |

```bash
pip install -r requirements.txt
```

`environment.yml` is a single-environment sketch of the same packages. If one
Python can import both stacks, set `CROWN_SKIP_CONDA=1`. Otherwise the CLI
re-invokes `conda run -n 3dpoint` and `conda run -n nnunetv2`.

The vendored `pointnet2_cuda` binary is CPython 3.8 on Linux. On any other
Python, build `vendor/DentalModelSegmentation/src/pointnet2` before
segmentation.

CPU shape checks do not need a GPU or a checkpoint:

```bash
python selftest.py
```

## Download CROWN-base weights

The archive next to this folder is `../CROWN-base-weights.tar.gz`. Extract it
so the layout is:

```text
weights/
  CROWN-base/
    phase3_sdf.pt    # bone SDF, Table 1 CROWN-base
    phase2_root.pt   # root completion used as RGSP
    SHA256SUMS
```

From this directory the default weight root is `../weights` (or `./weights`).
Override with `--weights` or `CROWN_WEIGHTS`.

Checksums and the experiment ids are in `../WEIGHTS.md`.

### Third-party segmentation weights

Raw IOS segmentation also needs DentalModelSegmentation checkpoints. They are
**not** CROWN weights and are **not** in the base archive. Place them at:

```text
<weights>/third_party/DentalModelSegmentation/
  model_all_tooth_seg
  model_cls
  model_pct_refine_16k
  model_pct_refine_32k
  model_pred_centroid
  model_pred_centroid_10k
  model_refine
  model_rotation
```

If the mesh is already labeled, skip that step (see below).

## Input

- Raw IOS mesh: `.ply`, `.stl`, or `.obj`, one arch per file, millimetres.
- `--upper` and `--lower` are independent. Either may be omitted.
- No example scan is shipped. Use a scan you are allowed to process.

```bash
python infer_ios.py \
  --weights /path/to/weights \
  --upper upper.ply \
  --lower lower.ply \
  --out ./out_case \
  --case-id Case001
```

Already-labeled mesh (FDI labels in a text file, one integer per vertex):

```bash
python infer_ios.py \
  --weights /path/to/weights \
  --skip-segment \
  --jaw upper \
  --mesh labeled_upper.ply \
  --labels labeled_upper_fdi.txt \
  --ios-mesh raw_upper.ply \
  --out ./out_case
```

## Output

| File | Contents |
|---|---|
| `<out>/<jaw>/bone_mc.obj` | Marching-cubes bone surface (iso 0) |
| `<out>/<jaw>/pred_sdf.npz` | 128³ SDF, `bounds_min_mm`, `extent_mm` |
| `<out>/<jaw>/feat.npz` | 7-channel volume passed to Phase 3 |
| `<out>/<jaw>/meta.json` | Tooth counts and iso level |

Channel layout of `feat`: 0 IOS log-density, 1–3 RGB filled with 0 (norgb),
4 RGSP root log-density, 5 tooth id, 6 tooth boundary.

Phase 2 uses the aggregate per-tooth frame in `priors/per_tooth_prior.json`
(group means and scales, not a case record).

## What this package leaves out

- CROWN-pro checkpoints and FramePredictor
- Training loops, optimizers, and experiment logs
- Hospital scans, CBCT/CT, and identifier tables
- The confidential review ZIP (`CROWN-review-inference.zip`), which defaults to
  the pro variant and is not this public snapshot

## 中文

本目录只跑 CROWN-base：口扫网格进，牙槽骨 SDF 和网格出。不含训练、不含病例、不含 CROWN-pro 权重。CROWN-pro 权重请联系作者：`TODO_CONTACT_EMAIL`。权重说明见 `../WEIGHTS.md`。分割第三方权重需另放在 `third_party/DentalModelSegmentation/`，不在 base 压缩包里。
