# Vendored inference definitions

Network definitions and the third-party tooth-segmentation source used to run
CROWN-base. This is not a training tree. Training scripts and lab shell
scripts were left out of this snapshot.

| Path | Role |
|---|---|
| `mint/model_point_completion.py` | Phase-2 `CoarseToFineFoldingNet` and `PointNetEncoder` |
| `nnunet/backbone_nnunet_sdf.py` | 7-to-1 PlainConvUNet SDF backbone |
| `nnunet/nnUNetPlans.json` | Plans used to rebuild the backbone. Checkpoint `plans_json` paths are ignored |
| `DentalModelSegmentation/` | Third-party IOS tooth segmentation and RotationNet |

The draft note in the top-level `LICENSE` says third-party code under
`vendor/` keeps its own license, and it names PointNet++ as MIT. This snapshot
does not include `src/pointnet2/LICENSE` or a DentalModelSegmentation license
file.

The bundled `pointnet2_cuda` extension is CPython 3.8 on Linux x86_64.
`pointnet2_utils.py` uses that binary when the import succeeds and the tensors
are on CUDA; otherwise it uses the PyTorch implementations in the same file.
`src/pointnet2/setup.py` builds the CUDA extension for another Python.

Checkpoints are not stored here.
