import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

try:
    import pointnet2_cuda as pointnet2
    _HAS_POINTNET2_CUDA = True
except Exception:
    pointnet2 = None
    _HAS_POINTNET2_CUDA = False


def _fps_cpu(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """CPU fallback for furthest point sampling. xyz: (B, N, 3), returns (B, npoint) int64."""
    B, N, _ = xyz.size()
    device = xyz.device
    batch_indices = []
    for b in range(B):
        pts = xyz[b]
        indices = torch.zeros(npoint, dtype=torch.long, device=device)
        farthest = torch.argmax(torch.sum(pts ** 2, dim=-1))
        indices[0] = farthest
        distances = torch.sum((pts - pts[farthest:farthest + 1]) ** 2, dim=-1)
        for i in range(1, npoint):
            farthest = torch.argmax(distances)
            indices[i] = farthest
            dist_new = torch.sum((pts - pts[farthest:farthest + 1]) ** 2, dim=-1)
            distances = torch.minimum(distances, dist_new)
        batch_indices.append(indices)
    return torch.stack(batch_indices)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        """
        B, N, _ = xyz.size()
        if not _HAS_POINTNET2_CUDA or not xyz.is_cuda:
            return _fps_cpu(xyz.contiguous(), npoint)
        assert xyz.is_contiguous()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        try:
            pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
            return output
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() or "device" in str(e).lower():
                return _fps_cpu(xyz.cpu(), npoint).to(xyz.device).to(torch.int32)
            raise

    @staticmethod
    def backward(xyz, a=None):
        return None, None


def furthest_point_sample(xyz, npoint):
    return FurthestPointSampling.apply(xyz.contiguous(), npoint)


def _gather_operation_cpu(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """CPU fallback: features (B, C, N), idx (B, npoint) -> (B, C, npoint)"""
    B, C, N = features.size()
    npoint = idx.size(1)
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1).long()
    return torch.gather(features, 2, idx_exp)


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()

        if not _HAS_POINTNET2_CUDA or not features.is_cuda:
            output = _gather_operation_cpu(features, idx)
            ctx.for_backwards = (idx, C, N)
            return output

        output = torch.cuda.FloatTensor(B, C, npoint)
        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)
        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        if not grad_out.is_cuda:
            grad_features = torch.zeros(B, C, N, dtype=grad_out.dtype, device=grad_out.device)
            idx_exp = idx.unsqueeze(1).expand(-1, C, -1).long()
            grad_features.scatter_add_(2, idx_exp, grad_out)
            return grad_features, None
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


def _three_nn_cpu(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback for 3 nearest neighbors."""
    dist = torch.cdist(unknown, known)
    dist3, idx = torch.topk(dist, 3, dim=2, largest=False)
    return dist3, idx.long()


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        B, N, _ = unknown.size()
        m = known.size(1)

        if not _HAS_POINTNET2_CUDA or not unknown.is_cuda:
            return _three_nn_cpu(unknown, known)

        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)
        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


def _three_interpolate_cpu(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU: output[b,c,i] = sum_j weight[b,i,j] * features[b,c,idx[b,i,j]]"""
    B, c, m = features.size()
    n = idx.size(1)
    device = features.device
    output = torch.zeros(B, c, n, device=device, dtype=features.dtype)
    for j in range(3):
        fidx = idx[:, :, j]
        w = weight[:, :, j:j+1].permute(0, 2, 1)
        gathered = torch.gather(features, 2, fidx.unsqueeze(1).expand(-1, c, -1))
        output += w * gathered
    return output


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()
        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)

        if not _HAS_POINTNET2_CUDA or not features.is_cuda:
            return _three_interpolate_cpu(features, idx, weight)

        output = torch.cuda.FloatTensor(B, c, n)
        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()
        if not grad_out.is_cuda or not _HAS_POINTNET2_CUDA:
            grad_features = torch.zeros(B, c, m, dtype=grad_out.dtype, device=grad_out.device)
            for j in range(3):
                fidx = idx[:, :, j]
                w = weight[:, :, j:j+1]
                grad_features.scatter_add_(2, fidx.unsqueeze(1).expand(-1, c, -1), w * grad_out)
            return grad_features, None, None
        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def _grouping_operation_cpu(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """CPU: features (B,C,N), idx (B,npoint,nsample) -> (B,C,npoint,nsample)"""
    B, C, N = features.size()
    nfeatures, nsample = idx.size(1), idx.size(2)
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1, -1).long()
    return torch.gather(features.unsqueeze(2).expand(-1, -1, nfeatures, -1), 3, idx_exp)


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: (B, C, N)
        :param idx: (B, npoint, nsample)
        :return: output: (B, C, npoint, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        if not _HAS_POINTNET2_CUDA or not features.is_cuda:
            output = _grouping_operation_cpu(features, idx)
            ctx.for_backwards = (idx, N)
            return output

        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)
        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)
        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards
        B, C, npoint, nsample = grad_out.size()
        if not grad_out.is_cuda:
            grad_features = torch.zeros(B, C, N, dtype=grad_out.dtype, device=grad_out.device)
            idx_flat = idx.reshape(B, -1)
            grad_flat = grad_out.reshape(B, C, -1)
            for b in range(B):
                for pos in range(idx_flat.size(1)):
                    k = idx_flat[b, pos].item()
                    grad_features[b, :, k] += grad_flat[b, :, pos]
            return grad_features, None
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def _ball_query_cpu(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """CPU fallback for ball query."""
    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    device = xyz.device
    idx = torch.zeros(B, npoint, nsample, dtype=torch.long, device=device)
    r2 = radius * radius
    for b in range(B):
        dist = torch.cdist(new_xyz[b:b+1], xyz[b:b+1]).squeeze(0)
        mask = dist <= r2
        for i in range(npoint):
            valid = torch.where(mask[i])[0]
            k = min(nsample, valid.size(0))
            if k > 0:
                _, topk_idx = torch.topk(dist[i, valid], k, largest=False)
                idx[b, i, :k] = valid[topk_idx]
    return idx


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return: idx: (B, npoint, nsample)
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)

        if not _HAS_POINTNET2_CUDA or not xyz.is_cuda:
            return _ball_query_cpu(radius, nsample, xyz, new_xyz)

        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
