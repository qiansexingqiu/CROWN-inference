# CROWN-base

CROWN-base predicts a whole-arch alveolar-bone signed distance field from an
intraoral-scan mesh, and writes that field as a volume and as a marching-cubes
surface. This repository is the inference package for the base variant.
The entry point is [`infer_ios.py`](infer_ios.py).

> Research reconstruction. Not a radiograph and not for clinical use.

## Pipeline

1. **Segment and orient.** DentalModelSegmentation labels the teeth.
   RotationNet places the arch in a canonical frame: predicted up toward +Z,
   predicted forward toward −Y, arch centroid at the origin.
   A mesh that already has per-vertex labels and already sits in that frame
   skips this stage.
2. **Complete roots.** Phase 2 turns each labeled crown into a root-completed
   point set, using `CROWN-base/phase2_root.pt` and the group statistics in
   [`priors/per_tooth_prior.json`](priors/per_tooth_prior.json).
3. **Build the volume.** The scan, the labels, and the predicted roots become
   a seven-channel grid. The default grid is 128³.
4. **Predict bone.** Phase 3 maps that grid to an alveolar-bone SDF with
   `CROWN-base/phase3_sdf.pt` and extracts the marching-cubes surface.

Color channels in the volume are zero. Vertex coordinates are millimetres.

## Installation

[`environment.yml`](environment.yml) creates one environment, `crown-infer-3d`,
with the packages listed in [`requirements.txt`](requirements.txt).

```bash
conda env create -f environment.yml
conda activate crown-infer-3d
export CROWN_SKIP_CONDA=1
python selftest.py
```

`CROWN_SKIP_CONDA=1` runs every stage in this environment. Leave it unset
only when conda environments named `3dpoint` and `nnunetv2` are already
installed. Other names are selected with `CROWN_ENV_3D` and `CROWN_ENV_NN`.

The vendored PointNet++ CUDA extension is built for CPython 3.8 on Linux.
In `crown-infer-3d` (Python 3.10) those operators use the PyTorch
implementations in the same module.

`python selftest.py` checks the tooth-label map and a small feature volume.
It does not need the checkpoints.

## Weights

Checkpoints are not in git. Two downloads are used, and both folders are
shared so that anyone with the link can view them. The segmentation
checkpoints are separate from the CROWN-base archive.

### CROWN-base

Download the five parts from
[CROWN-opensource](https://drive.google.com/drive/folders/1Ci98nj4MJSnLH1GMx4S478pfAP2RWlEH)
and join them in this order:

```bash
cat \
  CROWN-base-weights.tar.gz.partaa \
  CROWN-base-weights.tar.gz.partab \
  CROWN-base-weights.tar.gz.partac \
  CROWN-base-weights.tar.gz.partad \
  CROWN-base-weights.tar.gz.partae \
  > CROWN-base-weights.tar.gz
tar -xzf CROWN-base-weights.tar.gz
```

`--weights` is the directory that contains `CROWN-base/`, not the archive and
not the `CROWN-base` directory itself:

```text
weights/
  CROWN-base/
    phase2_root.pt
    phase3_sdf.pt
```

`CROWN_WEIGHTS` sets the same directory when `--weights` is omitted. The
prior file stays in this repository.

### Tooth-segmentation checkpoints

Download the seven files below from
[DentalModelSegmentation](https://drive.google.com/drive/folders/17zoALd8rb6BQfa8w8-orsrVzSIM1Xkhq).
Keep the names as they appear, and do not unzip the individual files. Place
them here:

```text
weights/third_party/DentalModelSegmentation/
  model_all_tooth_seg
  model_cls
  model_pct_refine_16k
  model_pct_refine_32k
  model_pred_centroid
  model_pred_centroid_10k
  model_rotation
```

The raw-mesh path loads these files. `--skip-segment` does not.

## Quickstart

From this repository, with `crown-infer-3d` active:

```bash
export CROWN_SKIP_CONDA=1
python infer_ios.py \
  --weights /path/to/weights \
  --upper upper.ply \
  --lower lower.ply \
  --out ./out_case \
  --case-id Case001
```

Each flag takes one arch. `.ply`, `.stl`, and `.obj` are the usual mesh
types. Either arch may be omitted.

A labeled arch, already in the canonical frame above:

```bash
export CROWN_SKIP_CONDA=1
python infer_ios.py \
  --weights /path/to/weights \
  --skip-segment \
  --jaw upper \
  --mesh labeled_upper.ply \
  --labels labeled_upper.txt \
  --out ./out_case
```

`--labels` is one integer per vertex of `--mesh`. When any label is 17 or
greater, the file is read as FDI notation; otherwise the values are
sequential tooth ids 1–14 and 0 is background. `--ios-mesh` is an optional
second mesh whose vertices fill the scan channel. When it is omitted, that
channel comes from `--mesh`.

## Inputs and outputs

Each jaw is written to `<out>/upper/` or `<out>/lower/`.

| File | Contents |
|---|---|
| `bone_mc.obj` | Marching-cubes bone surface, vertices in millimetres |
| `pred_sdf.npz` | Predicted SDF and the grid bounds in millimetres |
| `feat.npz` | Seven-channel volume and the same bounds. Color channels are zero |
| `predicted_roots_mm.ply` | Phase-2 points in millimetres |
| `meta.json` | Jaw name, variant `base`, iso level used for the surface, and the research-use note |

On the raw-mesh path, oriented copies and the segmenter workspace are kept
under `<out>/_canonical/` and `<out>/_work/<case-id>/`.

## Command-line reference

`--out` is required. The run also needs a weights directory and one jaw.

| Flag | Default | Role |
|---|---|---|
| `--weights` | `CROWN_WEIGHTS`, else a nearby `weights/` that contains `CROWN-base/` | Directory containing `CROWN-base/` |
| `--upper`, `--lower` | unset | Arch meshes |
| `--out` | required | Output directory |
| `--case-id` | `Case001` | Name used under `_work/` and in `meta.json` |
| `--device` | `cuda` | Torch device. A CUDA name uses CPU when CUDA is unavailable |
| `--grid-size` | `128` | Grid side length |
| `--margin-mm` | `4.0` | Margin around the arch, in millimetres |
| `--crown-margin-mm` | `1.5` | Phase-2 crown and root separation, in millimetres |
| `--seed` | `42` | Base sample seed |
| `--stage` | `all` | `all`, `preprocess`, `reconstruct`, or `phase3` |
| `--skip-segment` | off | Labeled-mesh path |
| `--jaw` | unset | `upper` or `lower` with `--skip-segment` |
| `--mesh`, `--labels` | unset | Labeled mesh and its per-vertex labels |
| `--ios-mesh` | the labeled mesh | Scan vertices on the labeled path |
| `--no-amp` | off | Turn off CUDA autocast in Phase 3 |

`python infer_ios.py --help` prints the same flags.

## Reproducibility

Phase 2 samples crown points with NumPy seed `--seed + 1000 + tooth id`.
The per-tooth center and scale come from
[`priors/per_tooth_prior.json`](priors/per_tooth_prior.json).
Phase 3 rebuilds the network from
[`vendor/nnunet/nnUNetPlans.json`](vendor/nnunet/nnUNetPlans.json)
(`3d_fullres`, seven input channels, one output channel) and loads
`phase3_sdf.pt`.

## Acknowledgements

The copyright line in [`LICENSE`](LICENSE) names Shanghai Ninth People's
Hospital, Shanghai Jiao Tong University School of Medicine.

[`vendor/README.md`](vendor/README.md) describes the vendored Phase-2
networks, the nnU-Net SDF backbone, and DentalModelSegmentation
(tooth segmentation, RotationNet, and the PointNet++ operators).

We thank PerioAI.

## License

See [`LICENSE`](LICENSE). The file is an MIT license marked as a draft for
the CROWN-base inference code in this directory. The draft note states that
CROWN-pro weights lie outside that text, that code under `vendor/` keeps its
own license, and that PointNet++ is MIT. DentalModelSegmentation has no
separate license file in this snapshot. The note's contact line is still a
placeholder; requests go to the address below.

## CROWN-pro

CROWN-pro and stronger variants are available from the authors. Academic use
is free. Commercial use is discussed separately. Those weights are not in
this repository or in the CROWN-base archive.

Email: [qiansexingqiutlr123@sjtu.edu.cn](mailto:qiansexingqiutlr123@sjtu.edu.cn)
