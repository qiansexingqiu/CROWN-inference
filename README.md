# CROWN-base

CROWN-base reads one intraoral-scan arch mesh and writes a whole-arch
alveolar-bone signed distance field, together with a marching-cubes surface of
that field. This repository is the inference package for that base variant.
The entry point is [`infer_ios.py`](infer_ios.py). It loads CROWN-base
checkpoints only.

> This reconstruction is not a radiograph and is not for clinical use.
> `meta.json` repeats that warning on every successful jaw.

## Contents

- [Release contents](#release-contents)
- [Installation](#installation)
- [CROWN-base weights](#crown-base-weights)
- [Quickstart](#quickstart)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Command-line interface](#command-line-interface)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [License](#license)
- [CROWN-pro](#crown-pro)
- [Third-party code](#third-party-code)

## Release contents

The git tree contains:

- the CROWN-base inference program and the vendored network definitions it imports
- the per-tooth prior [`priors/per_tooth_prior.json`](priors/per_tooth_prior.json)
- [`requirements.txt`](requirements.txt) and [`environment.yml`](environment.yml), with versions unpinned
- this documentation

The public weight archive is separate from git. Its reconstruction is
documented in [CROWN-base weights](#crown-base-weights).

Also outside this repository:

- CROWN-pro weights, stronger variants, and FramePredictor. CROWN-base uses the prior file above. See [CROWN-pro](#crown-pro).
- Training code, the training set, and the annotation pipeline.
- CBCT, CT, and any other radiographic volume. This program has no reader for them.
- Hospital scans and identifier tables. No example scan is shipped.
- DentalModelSegmentation checkpoints. Raw-mesh segmentation needs them, and they are not in `CROWN-base-weights.tar.gz`.
- A bibliographic record.

## Installation

Segmentation, RotationNet canonicalization, and Phase-2 root completion are
written for a conda environment named `3dpoint`. Phase-3 SDF inference is
written for a conda environment named `nnunetv2`. Override the names with
`CROWN_ENV_3D` and `CROWN_ENV_NN`.

| Stage | Default environment | Code |
|---|---|---|
| Segmentation, canonical frame, Phase-2 roots | `3dpoint` | vendored DentalModelSegmentation, RotationNet, `vendor/mint` |
| Phase-3 bone SDF and marching cubes | `nnunetv2` | `vendor/nnunet`, `nnunetv2`, scikit-image |

[`requirements.txt`](requirements.txt) lists the packages those stages import:
`torch`, `numpy`, `scipy`, `scikit-learn`, `trimesh`, `open3d`, `pymeshlab`,
`loguru`, `tqdm`, `nnunetv2`, and `scikit-image`. Versions are unpinned.

```bash
pip install -r requirements.txt
```

[`environment.yml`](environment.yml) is a single-environment sketch named
`crown-infer-3d` (Python 3.10 plus the same packages). Creating that
environment does not create `3dpoint` or `nnunetv2`. Point
`CROWN_ENV_3D` and `CROWN_ENV_NN` at the environments you actually created,
or run both stages in one interpreter:

```bash
export CROWN_SKIP_CONDA=1
```

Interpreter selection in [`crown_infer/condaenv.py`](crown_infer/condaenv.py):

- `CROWN_SKIP_CONDA=1` keeps the current Python for every stage.
- Otherwise, if `conda` is on `PATH`, the CLI re-invokes
  `conda run -n <env> --no-capture-output python`.
- If `conda` is not on `PATH`, the re-invocation uses the current Python.
- `CROWN_PYTHON_<ENV>` replaces that interpreter. For the default names the
  variables are `CROWN_PYTHON_3DPOINT` and `CROWN_PYTHON_NNUNETV2`.

The vendored PointNet++ operators try to load
[`vendor/DentalModelSegmentation/src/pointnet2/pointnet2_cuda.cpython-38-x86_64-linux-gnu.so`](vendor/DentalModelSegmentation/src/pointnet2/pointnet2_cuda.cpython-38-x86_64-linux-gnu.so).
That binary matches CPython 3.8 on Linux x86_64. If the import fails, or the
tensors are not CUDA tensors, the same module uses its PyTorch implementations.
[`setup.py`](vendor/DentalModelSegmentation/src/pointnet2/setup.py) builds a
CUDA extension and names compiler architectures 7.0, 7.5, 8.0, and 8.6. Those
are build targets. This repository does not report runtime or memory figures.

CPU checks that do not load a checkpoint:

```bash
python selftest.py
```

`selftest.py` checks `fdi_to_sequential` on `[11, 21, 0, 47]` and the shape of
a synthetic 16³ feature volume.

## CROWN-base weights

Git does not store checkpoints. The release archive and the directory you pass
to `--weights` are different objects.

| Object | What it is |
|---|---|
| `CROWN-base-weights.tar.gz.partaa` … `partae` | Five binary parts in Google Drive |
| `CROWN-base-weights.tar.gz` | Those parts joined, in order. 399820895 bytes |
| Directory passed as `--weights` | The extracted tree. It must contain `CROWN-base/` |

Google Drive stores the parts in
[CROWN-opensource](https://drive.google.com/drive/folders/1Ci98nj4MJSnLH1GMx4S478pfAP2RWlEH).
The folder is shared as “Anyone with the link” (Viewer). Per-part sizes and
digests are also written in
[PARTS.md](https://drive.google.com/file/d/1mOKrKAQ1fIcpQIXPlzxYjSPJA4V9oLuS/view).
The same folder contains `WEIGHTS.md`.

Download all five parts. Join them in this order. A glob is safe only when
these five names are the only matches and the shell sorts them as below.

```bash
cat \
  CROWN-base-weights.tar.gz.partaa \
  CROWN-base-weights.tar.gz.partab \
  CROWN-base-weights.tar.gz.partac \
  CROWN-base-weights.tar.gz.partad \
  CROWN-base-weights.tar.gz.partae \
  > CROWN-base-weights.tar.gz
sha256sum CROWN-base-weights.tar.gz
```

The joined archive is 399820895 bytes. The five part sizes sum to that
number. Expected SHA-256:

```text
7fa8032b0e600ab04d0e381b4b37e50eab05d090bfc47f49b9e47f4f63eec021
```

| Part | Bytes | SHA-256 |
|---|---:|---|
| `CROWN-base-weights.tar.gz.partaa` | 99614720 | `326e058476c53b3c72c342eaee06657147272c95465f27db1110e884c2ac0d87` |
| `CROWN-base-weights.tar.gz.partab` | 99614720 | `5675b96256f323ecfbcdc747f9fdfb58ec39421bdda23d310fd2a4c19e98a7c6` |
| `CROWN-base-weights.tar.gz.partac` | 99614720 | `b25eed27142ff7a46704b6e07bad71e93ae1d29136b4bc0e5eaf12e84f29a42d` |
| `CROWN-base-weights.tar.gz.partad` | 99614720 | `3b268b04d6d3705db36c45b2a722cf3e042e455b7dacaad84f17224e57ab8957` |
| `CROWN-base-weights.tar.gz.partae` | 1362015 | `3534776c1e1b3dd8204d74e29411a4681da95d2edb4a79e394e47fa584fb7eab` |

Extract the archive so the checkpoints sit here:

```text
weights/
  CROWN-base/
    phase2_root.pt
    phase3_sdf.pt
    SHA256SUMS
```

The loader opens the two `.pt` files. The layout previously published for this
archive also lists `SHA256SUMS` in that directory. The loader does not read
`SHA256SUMS`.

`--weights` and `CROWN_WEIGHTS` point at `weights/`, the directory that
contains `CROWN-base/`. They do not point at the `.tar.gz`, at a part file, or
at `weights/CROWN-base/` itself.

Resolution order in [`crown_infer/syspath.py`](crown_infer/syspath.py):

1. `--weights`, when passed
2. `CROWN_WEIGHTS`
3. `../weights`, then `./weights`, if that directory contains `CROWN-base/`

[`crown_infer/reconstruct.py`](crown_infer/reconstruct.py) requires
`CROWN-base/phase2_root.pt`, `CROWN-base/phase3_sdf.pt`, and
`priors/per_tooth_prior.json` from this repository. The prior is not taken
from the archive.

### Third-party segmentation checkpoints

The raw-mesh path also loads DentalModelSegmentation checkpoints. They are not
CROWN weights and they are not in the base archive. This repository does not
host them. Place the files the loader opens, with these exact names:

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

`--skip-segment` does not read this directory. The canonicalization model is
`model_rotation` in the same directory, and it is used only when segmentation
runs.

## Quickstart

From this repository, after the extracted `weights/` directory and the
third-party checkpoints are in place:

```bash
python infer_ios.py \
  --weights /path/to/weights \
  --upper upper.ply \
  --lower lower.ply \
  --out ./out_case \
  --case-id Case001
```

Either arch may be omitted. `--out` is required. The default device string is
`cuda`; if CUDA is unavailable, a `cuda*` device string falls back to CPU.

A jaw that is already labeled, and already in the canonical frame described
under [Inputs](#inputs), can skip segmentation:

```bash
python infer_ios.py \
  --weights /path/to/weights \
  --skip-segment \
  --jaw upper \
  --mesh labeled_upper.ply \
  --labels labeled_upper.txt \
  --ios-mesh upper_canonical.ply \
  --out ./out_case
```

`--ios-mesh` is optional. When it is omitted, channel 0 is taken from
`--mesh`.

Full inference needs the CROWN-base checkpoints, and the raw-mesh path also
needs the third-party segmentation checkpoints. `python selftest.py` does not
load either.

## Inputs

The default path accepts one arch mesh per jaw. [`crown_infer/segment.py`](crown_infer/segment.py)
loads that file with trimesh (`process=False`), concatenates a scene into one
mesh, and writes a PLY and an OFF copy for the segmenter. Typical files are
`.ply`, `.stl`, and `.obj`. The program does not open scanner project
databases, DICOM, CBCT, image volumes, or a precomputed `feat.npz` as its
input. An existing `feat.npz` is read only by the internal `--stage phase3`.

Vertex numbers are used as millimetres. Nothing in the pipeline converts
metres or scanner units. Canonicalization rotates and translates; it does not
rescale. The prior offsets and `--margin-mm` are millimetres in that same
coordinate system.

`--upper` and `--lower` are independent. The program does not split a mesh
that contains both arches.

### Default path

For each requested jaw the program:

1. Runs the vendored DentalModelSegmentation `inference()` on the OFF copy,
   with `rotation=True`. The label file written beside the labeled OFF is the
   label file used afterwards.
2. Runs RotationNet (`model_rotation`) on the jaw PLY. The predicted up axis
   is rotated onto `+Z` when its dot product with `+Z` is below 0.99. The
   predicted forward vector has its z component set to 0, then is rotated onto
   `−Y` when its dot product with `−Y` is below 0.99.
3. Applies that rotation to the IOS mesh and to the labeled OFF vertices, then
   subtracts the centroid of the RotationNet input points so the centroid
   moves to the origin. Labels are copied with the vertices and are not
   reindexed.
4. Runs Phase-2 and Phase-3 in that frame.

Intermediate files go to `<out>/_work/<case-id>/` and `<out>/_canonical/`.
`--case-id` (default `Case001`) is only a name in those paths and in
`meta.json`.

### `--skip-segment`

This mode does not segment and does not canonicalize. `--mesh` is the labeled
mesh. `--labels` is one integer per vertex of `--mesh`, readable by
`numpy.loadtxt`, and the count must equal the vertex count. `--jaw` is
`upper` or `lower`. When `--jaw` is set, that value selects the jaw, and
`--upper` / `--lower` are not used to choose it. Supply a mesh that is already
in the frame the default path would have produced. The prior offsets are added
in the coordinates of these files.

### Labels

[`crown_infer/io_mesh.py`](crown_infer/io_mesh.py) and
[`crown_infer/voxel.py`](crown_infer/voxel.py) implement the map below.

- If the maximum label is 17 or greater, labels are read as FDI. Quadrants 1
  and 4, tooth positions 1–7, become sequential ids 1–7. Any other quadrant
  digit, positions 1–7, becomes sequential ids 8–14. Position 8 and every
  other code, including 0, stay 0.
- If every label is below 17, the values are already sequential ids. 0 is
  background.
- Phase-2 keeps sequential ids 1–14 that have at least 100 vertices. The prior
  file also contains groups 15 and 16 for each jaw; this loop does not request
  them.
- The tooth-id volume channel runs `fdi_to_sequential` again on those
  sequential ids and divides by 14. After that second map, sequential 1–7
  become 8–14, sequential 11–14 become 1–4, and sequential 8–10 become 0.

`selftest.py` locks the first map: FDI `11, 21, 0, 47` becomes `1, 8, 0, 7`.

### Per-tooth frame and Phase-2

For each kept tooth, the center is the crown-vertex centroid plus that group's
`offset_mean_mm`. The call uses `scale_mode="crown_r_ratio"`. This prior file
has neither `scale_over_crown_r_mean` nor `crown_r_median_mm`, so the scale
that results is `scale_median_mm`. Crown points are mapped into that ball with
`(point - center) / scale`.

The checkpoint key `num_in_points` sets how many crown points are sampled
(fallback 4096 when the key is absent). Sampling is without replacement when
the crown is at least that large, and with replacement otherwise. The NumPy
seed is `--seed + 1000 + tooth_id` (default `--seed` is 42).

Predicted points farther from the sampled crown than
`crown_margin_mm / scale` in that unit ball are kept and concatenated with the
sampled crown. The default margin is 1.5 mm. If no predicted point passes,
the sampled crown is kept alone. Points are mapped back by
`point * scale + center`. A tooth whose prior group is missing is recorded as
`frame_failed` and is omitted from the root cloud.

### Phase-3 volume

The box is the axis-aligned bounds of the IOS vertices and the labeled-mesh
vertices, expanded by `--margin-mm` (default 4 mm) on every side. The default
grid is 128³ (`--grid-size`).

Voxel indices use

```text
voxel = floor( (point_mm - bounds_min_mm) / (extent_mm + 1e-9) * G )
```

with axes `(x, y, z)` in mesh coordinates. The network receives the full
`7×G×G×G` array in one forward pass. This entry point does not run nnU-Net
resampling, and it does not apply `transpose_forward` from the plans file.
The backbone is rebuilt from
[`vendor/nnunet/nnUNetPlans.json`](vendor/nnunet/nnUNetPlans.json),
configuration `3d_fullres`, with 7 input channels and 1 output channel.
Absolute plans paths stored inside `phase3_sdf.pt` are ignored.

Mesh vertex colors are read when present, then channels 1–3 are overwritten
with 0. CROWN-base does not consume color.

## Outputs

Each jaw is written to `<out>/upper/` or `<out>/lower/`.

| File | Contents |
|---|---|
| `bone_mc.obj` | Marching-cubes surface. Vertices are millimetres. Faces are 1-based. No normals are stored. |
| `pred_sdf.npz` | `pred_sdf` float32 `(G, G, G)`, plus `bounds_min_mm` and `extent_mm`, each float32 `(3,)`. |
| `feat.npz` | `feat` float32 `(7, G, G, G)` and the same bound arrays. |
| `predicted_roots_mm.ply` | ASCII PLY of the Phase-2 points in millimetres, with vertex color `(255, 80, 60)`. |
| `reconstruct_meta.json` | Tooth count, Phase-2 `arch` string from the checkpoint, grid size, margin, and per-tooth status. |
| `meta.json` | Case id, jaw, variant `base`, iso level actually used, SDF min/max, backbone field from the checkpoint when present, device, and the clinical warning. |

`pred_sdf` is the network output with no further rescaling. Marching cubes
uses iso 0 when that level lies between the minimum and the maximum. Otherwise
it uses the midpoint of those two values, provided they differ. A constant
field raises. The iso written into the OBJ is `marching_cubes_iso` in
`meta.json`.

A marching-cubes vertex index maps back by

```text
world_mm = bounds_min_mm + index * (extent_mm / G)
```

Index 0 is `bounds_min_mm`, not a voxel center. `G` is the side length of
`pred_sdf`. `reconstruct_meta.json` records the same value as `grid_size`.

`feat` channels:

| Channel | Value |
|---|---|
| 0 | IOS log-density, `log1p(count) / log1p(max count)`, on vertices inside the box. |
| 1–3 | 0. |
| 4 | The same log-density, computed independently on the Phase-2 points. Source comments call this channel RGSP. This repository does not expand that name. |
| 5 | Dominant tooth id after the second FDI map, divided by 14. |
| 6 | 1 where a voxel and a face-neighbor have different non-zero dominant ids. Neighbor lookup wraps (`numpy.roll`). |

On the segmentation path, `<out>/_canonical/` holds `<jaw>_ios.ply`,
`<jaw>_ddc.off`, `<jaw>_label.txt`, and `<jaw>_canonical_meta.json` (the 4×4
transform). `<out>/_work/` holds the segmenter workspace.

## Command-line interface

`python infer_ios.py --help` parses these flags. `--out` is the only flag
argparse marks required. A run also needs a weight directory and at least one
jaw (`--upper`, `--lower`, or `--skip-segment --jaw`).

| Flag | Default | Role |
|---|---|---|
| `--weights` | None, then `CROWN_WEIGHTS`, else `../weights` or `./weights` | Directory that contains `CROWN-base/` |
| `--upper` | unset | Upper-arch mesh |
| `--lower` | unset | Lower-arch mesh |
| `--out` | required | Output directory |
| `--case-id` | `Case001` | Intermediate directory name and `meta.json` field |
| `--device` | `cuda` | Torch device. A `cuda*` name is replaced with `cpu` when CUDA is unavailable |
| `--grid-size` | `128` | Cubic grid side `G` |
| `--margin-mm` | `4.0` | Box margin in millimetres |
| `--crown-margin-mm` | `1.5` | Phase-2 crown/root separation in millimetres |
| `--seed` | `42` | Base NumPy seed for crown sampling |
| `--stage` | `all` | `all`, `preprocess`, `reconstruct`, or `phase3` |
| `--skip-segment` | off | Use `--mesh` and `--labels` instead of the raw-mesh path |
| `--jaw` | unset | `upper` or `lower`, with `--skip-segment` |
| `--mesh` | unset | Labeled mesh, with `--skip-segment` |
| `--labels` | unset | Per-vertex labels for `--mesh` |
| `--ios-mesh` | None; the run then uses `--mesh` | Vertices for channel 0, with `--skip-segment` |
| `--no-amp` | off | Disable CUDA autocast in Phase-3. Autocast is already off on CPU |

`--stage` exists so the two conda environments can run separate processes.
`all` runs preprocess (unless `--skip-segment`), then reconstruct, then
phase 3. `preprocess` stops after `_canonical/`. `reconstruct` reads
`_canonical/` unless `--skip-segment` is set, and writes `feat.npz`.
`phase3` reads `feat.npz`.

## Repository layout

```text
infer_ios.py                 command-line entry point
selftest.py                  CPU checks, no checkpoints
crown_infer/                 segmentation, canonicalization, Phase-2, volume, Phase-3
priors/per_tooth_prior.json  group offset and scale statistics, millimetres
requirements.txt             unpinned packages
environment.yml              single-environment sketch
vendor/mint/                 Phase-2 completion networks
vendor/nnunet/               PlainConv SDF builder and nnUNetPlans.json
vendor/DentalModelSegmentation/
vendor/README.md             what the vendor tree is for
LICENSE
```

`vendor/` is not a training tree. Training scripts are not included.

## Reproducibility

- Phase-2 sampling uses NumPy seed `--seed + 1000 + tooth_id`.
- Importing the vendored segmenter sets `torch.manual_seed(2000228)` and, when
  CUDA is present, the matching CUDA seed. The CLI does not enable
  deterministic cuDNN algorithms.
- Channels 1–3 are forced to 0, including when the mesh carries vertex colors.
- `phase3_sdf.pt` may store a plans path from another machine. The loader
  rebuilds the network from the bundled plans file.
- Dependency versions are not pinned. The same checkpoint can disagree across
  PyTorch, CUDA, and scikit-image builds. This repository does not publish
  speed or memory measurements.
- `python selftest.py` checks the FDI map and a tiny feature volume. It does
  not load weights and it does not run a jaw.

## License

[`LICENSE`](LICENSE) is an MIT license. The copyright line names Shanghai
Ninth People's Hospital, Shanghai Jiao Tong University School of Medicine.

The draft note inside that file says the text is a draft for the CROWN-base
inference code in this directory, that it does not cover CROWN-pro weights,
and that third-party code under `vendor/` keeps its own license. The same note
names PointNet++ as MIT. This snapshot does not include a license file for
DentalModelSegmentation or for the vendored PointNet++ sources. The draft note
still has an unresolved contact placeholder. For CROWN-pro and commercial
requests, use the address in [CROWN-pro](#crown-pro).

This documentation does not change the license terms.

## CROWN-pro

CROWN-pro weights and stronger variants are not in this repository and not in
`CROWN-base-weights.tar.gz`. Request them from the authors. Academic use is
free. Commercial use is discussed separately.

Email: [qiansexingqiutlr123@sjtu.edu.cn](mailto:qiansexingqiutlr123@sjtu.edu.cn)

`infer_ios.py` calls the weight resolver with variant `base`. Any other name
raises.

## Third-party code

[`vendor/README.md`](vendor/README.md) lists the vendored pieces:

- `vendor/mint/model_point_completion.py`, Phase-2 completion networks
- `vendor/nnunet/backbone_nnunet_sdf.py` and `nnUNetPlans.json`
- `vendor/DentalModelSegmentation/`, IOS tooth segmentation and RotationNet

PointNet++ operators live under
`vendor/DentalModelSegmentation/src/pointnet2/`. nnU-Net is imported as the
`nnunetv2` package at Phase-3 build time.
