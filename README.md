# CROWN-base inference

Read an intraoral scan (IOS) mesh and write a whole-arch alveolar-bone signed
distance field (SDF) plus a marching-cubes surface. This package runs
**CROWN-base** only. It does not train, and it does not include imaging.

This reconstruction is not a radiograph and is not for clinical use.

## CROWN-pro and stronger variants

CROWN-pro weights and stronger variants are not in this repository and not in
`CROWN-base-weights.tar.gz`. Request them from the authors. Academic use is
free. Commercial use is discussed separately.

Email: [qiansexingqiutlr123@sjtu.edu.cn](mailto:qiansexingqiutlr123@sjtu.edu.cn)

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

Git does not store the checkpoints. The release archive is
`CROWN-base-weights.tar.gz`.

- Bytes: `399820895`
- sha256: `7fa8032b0e600ab04d0e381b4b37e50eab05d090bfc47f49b9e47f4f63eec021`

Google Drive keeps that archive as five sequential binary parts in the
[CROWN-opensource](https://drive.google.com/drive/folders/1Ci98nj4MJSnLH1GMx4S478pfAP2RWlEH)
folder. Per-part hashes and file ids are also written in
[PARTS.md](https://drive.google.com/file/d/1mOKrKAQ1fIcpQIXPlzxYjSPJA4V9oLuS/view).
Experiment notes are in `WEIGHTS.md` in that same folder.

The folder is shared as "Anyone with the link" (Viewer). The parts and notes
at the links above can be downloaded now. Download every part and join them in
alphabetical order (`partaa` through `partae`). Do not skip a file.

```bash
cat CROWN-base-weights.tar.gz.part* > CROWN-base-weights.tar.gz
sha256sum CROWN-base-weights.tar.gz
# expect 7fa8032b0e600ab04d0e381b4b37e50eab05d090bfc47f49b9e47f4f63eec021
```

| Part | Bytes | sha256 |
|---|---:|---|
| `CROWN-base-weights.tar.gz.partaa` | 99614720 | `326e058476c53b3c72c342eaee06657147272c95465f27db1110e884c2ac0d87` |
| `CROWN-base-weights.tar.gz.partab` | 99614720 | `5675b96256f323ecfbcdc747f9fdfb58ec39421bdda23d310fd2a4c19e98a7c6` |
| `CROWN-base-weights.tar.gz.partac` | 99614720 | `b25eed27142ff7a46704b6e07bad71e93ae1d29136b4bc0e5eaf12e84f29a42d` |
| `CROWN-base-weights.tar.gz.partad` | 99614720 | `3b268b04d6d3705db36c45b2a722cf3e042e455b7dacaad84f17224e57ab8957` |
| `CROWN-base-weights.tar.gz.partae` | 1362015 | `3534776c1e1b3dd8204d74e29411a4681da95d2edb4a79e394e47fa584fb7eab` |

Extract the archive so the layout is:

```text
weights/
  CROWN-base/
    phase3_sdf.pt    # bone SDF, Table 1 CROWN-base
    phase2_root.pt   # root completion used as RGSP
    SHA256SUMS
```

From this directory the default weight root is `../weights` (or `./weights`).
Override with `--weights` or `CROWN_WEIGHTS`.

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

- CROWN-pro checkpoints, stronger variants, and FramePredictor
- Training loops, optimizers, and experiment logs
- The training set and the patient pipeline. They are not released here and are not in the base-weight archive
- Hospital scans, CBCT/CT, and identifier tables
- The confidential review ZIP (`CROWN-review-inference.zip`), which defaults to the pro variant and is not this repository

## 中文

本仓库只提供 CROWN-base 推理代码：口扫网格进，牙槽骨 SDF 和网格出。不含训练代码。完整训练数据和患者流水线未开源，不能从本仓库或 base 权重包下载。

CROWN-base 权重不在 git 里。Drive 文件夹 [CROWN-opensource](https://drive.google.com/drive/folders/1Ci98nj4MJSnLH1GMx4S478pfAP2RWlEH) 中是 5 个分卷，从 `CROWN-base-weights.tar.gz.partaa` 到 `partae`。按字母序合并，不要漏文件：

```bash
cat CROWN-base-weights.tar.gz.part* > CROWN-base-weights.tar.gz
```

完整包为 399820895 字节，sha256 为 `7fa8032b0e600ab04d0e381b4b37e50eab05d090bfc47f49b9e47f4f63eec021`。分卷校验见 [PARTS.md](https://drive.google.com/file/d/1mOKrKAQ1fIcpQIXPlzxYjSPJA4V9oLuS/view)。该文件夹已设为「知道链接的任何人」可查看，分卷现在可以下载。

CROWN-pro 及更强变体不在发布包内，需要申请。学术使用免费，商业使用另议。联系 [qiansexingqiutlr123@sjtu.edu.cn](mailto:qiansexingqiutlr123@sjtu.edu.cn)。

分割用的第三方权重需另放在 `third_party/DentalModelSegmentation/`，不在 base 压缩包里。

## License

MIT. See `LICENSE`. That text covers this inference code. It does not cover
CROWN-pro weights or the third-party files under `vendor/`.
