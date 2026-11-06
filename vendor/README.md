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

`DentalModelSegmentation/src/pointnet2/LICENSE` is the MIT license for
PointNet++. The bundled `pointnet2_cuda` extension is CPython 3.8 / Linux
only. Other Pythons need a local build of `src/pointnet2`.

Checkpoints are not stored here.
