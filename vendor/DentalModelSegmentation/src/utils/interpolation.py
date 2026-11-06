import torch
from src.pointnet2 import pointnet2_utils

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def point_labels_interpolation(points, points_down, labels):
    points = torch.from_numpy(points)
    points_down = torch.from_numpy(points_down)
    labels = torch.from_numpy(labels)

    points = points.unsqueeze(0).to(_device, dtype=torch.float32)
    points_down = points_down.unsqueeze(0).to(_device, dtype=torch.float32)
    labels = labels.unsqueeze(0).unsqueeze(0).to(_device, dtype=torch.float32)

    dist, idx = pointnet2_utils.three_nn(points.contiguous(), points_down.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated_feats = pointnet2_utils.three_interpolate(labels.contiguous(), idx, weight)

    interpolated_feats = interpolated_feats[:, 0, :]
    return interpolated_feats.detach().cpu().numpy()


def batch_point_labels_interpolation(points, points_down, labels):
    points = torch.from_numpy(points)
    points_down = torch.from_numpy(points_down)
    labels = torch.from_numpy(labels)

    points = points.to(_device, dtype=torch.float32)
    points_down = points_down.to(_device, dtype=torch.float32)
    labels = labels.unsqueeze(1).to(_device, dtype=torch.float32)

    dist, idx = pointnet2_utils.three_nn(points.contiguous(), points_down.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated_feats = pointnet2_utils.three_interpolate(labels.contiguous(), idx, weight)

    interpolated_feats = interpolated_feats[:, 0, :]
    return interpolated_feats.detach().cpu().numpy()
