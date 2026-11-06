"""
点云补全基线：牙冠 (N,3) -> 全牙 (M,3)，与 prep_root_dataset 的单位球坐标一致。

结构：PointNet 风格编码器 + MLP 解码器输出固定数量点（PCN 类 baseline）。
另：**FoldingNetCompletion**（全局折叠）、**CoarseToFineFoldingNet**（粗骨架 + 局部 patch 折叠）、**TransformerCompletionNet**（PoinTr-lite：FPS 代理 + Transformer 编解码 + 粗点 + 局部折叠上采样）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """点云 -> 全局特征向量。"""

    def __init__(self, in_channels: int = 3, latent_dim: int = 1024) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, N) -> (B, latent_dim)"""
        f = self.backbone(x)
        g = f.max(dim=2)[0]
        return self.fc(g)


class MLPDecoder(nn.Module):
    """全局特征 -> (B, num_points, 3)"""

    def __init__(self, latent_dim: int = 1024, num_points: int = 2048, hidden: int = 2048) -> None:
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = self.net(z)
        return o.view(-1, self.num_points, 3)


class PointCompletionPCN(nn.Module):
    """PCN 风格：编码牙冠点云，解码为完整牙点云。"""

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_output_points: int = 2048,
        decoder_hidden: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(3, latent_dim)
        self.decoder = MLPDecoder(latent_dim, num_output_points, hidden=decoder_hidden)

    def forward(self, crown_b3n: torch.Tensor) -> torch.Tensor:
        """crown_b3n: (B, 3, N_in) -> pred (B, num_output_points, 3)"""
        z = self.encoder(crown_b3n)
        return self.decoder(z)


class FoldingNetCompletion(nn.Module):
    """Folding 解码：平面规则网格 (G×G×2) + 全局特征 → 两阶段 MLP folding → (G*G, 3)。

    与 PointCompletionPCN 相同的 forward 接口：``(B, 3, N_in) -> (B, num_output_points, 3)``。
    ``num_output_points`` 须为完全平方数（默认 4096 = 64×64）。
    """

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_output_points: int = 4096,
        decoder_hidden: int = 2048,
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        gs = int(num_output_points**0.5 + 0.5) if grid_size is None else int(grid_size)
        if gs * gs != num_output_points:
            raise ValueError(
                f"num_output_points 须为完全平方数（如 4096=64²），当前 {num_output_points}，"
                f"grid_size={gs} -> {gs*gs}"
            )
        self.latent_dim = latent_dim
        self.num_output_points = num_output_points
        self.grid_size = gs

        self.encoder = PointNetEncoder(3, latent_dim)

        u = torch.linspace(-1.0, 1.0, gs)
        v = torch.linspace(-1.0, 1.0, gs)
        gu, gv = torch.meshgrid(u, v, indexing="ij")
        grid = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1).unsqueeze(0)
        self.register_buffer("grid_uv", grid)

        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )

    def forward(self, crown_b3n: torch.Tensor) -> torch.Tensor:
        z = self.encoder(crown_b3n)
        B, _ = z.shape
        P = self.num_output_points
        g = self.grid_uv.expand(B, -1, -1)
        zrep = z.unsqueeze(1).expand(B, P, -1)
        h1 = self.fold1(torch.cat([g, zrep], dim=-1))
        out = self.fold2(torch.cat([h1, zrep], dim=-1))
        return out


class CoarseToFineFoldingNet(nn.Module):
    """由粗到细：全局特征 → 粗糙骨架点 (S×3) → 每点局部 2D patch 折叠 → 精细点云 (S·K×3)。

    ``num_output_points = num_coarse * patch_grid_size ** 2``（默认 256×4²=4096）。
    **forward 返回 ``(fine_pts, coarse_pts)``**，推理/评估时取 ``fine_pts`` 即可。
    """

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_output_points: int = 4096,
        decoder_hidden: int = 2048,
        num_coarse: int = 256,
        patch_grid_size: int = 4,
    ) -> None:
        super().__init__()
        k = patch_grid_size * patch_grid_size
        if num_coarse * k != num_output_points:
            raise ValueError(
                f"num_coarse * patch_grid_size^2 须等于 num_output_points："
                f"{num_coarse}*{k} != {num_output_points}"
            )
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.patch_grid_size = patch_grid_size
        self.patch_k = k
        self.num_output_points = num_output_points

        self.encoder = PointNetEncoder(3, latent_dim)
        self.coarse_mlp = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, num_coarse * 3),
        )
        u = torch.linspace(-0.05, 0.05, patch_grid_size)
        gu, gv = torch.meshgrid(u, u, indexing="ij")
        patch = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)
        self.register_buffer("patch_uv", patch.unsqueeze(0).unsqueeze(0))

        in_dim = 2 + latent_dim + 3
        self.local_fold = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )

    def forward(self, crown_b3n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(crown_b3n)
        B, L = z.shape
        S, K = self.num_coarse, self.patch_k
        coarse = self.coarse_mlp(z).view(B, S, 3)
        g = self.patch_uv.expand(B, S, K, 2)
        cexp = coarse.unsqueeze(2).expand(B, S, K, 3)
        zexp = z.view(B, 1, 1, L).expand(B, S, K, L)
        feat = torch.cat([g, zexp, cexp], dim=-1)
        off = self.local_fold(feat.reshape(B * S * K, -1)).view(B, S, K, 3)
        fine = (cexp + off).reshape(B, S * K, 3)
        return fine, coarse


# 牙位形态类型：上/下颌 × 位置 1..8（切/侧/尖/前磨1/前磨2/磨1/磨2/智齿）。
# sequential tooth_id：1..7 一侧、8..14 对侧、15/16 智齿。左右镜像共享形态类型。
NUM_TOOTH_TYPES = 16


def tooth_type_id(jaw: str, tooth_id: int) -> int:
    """(jaw, sequential tooth_id) → 形态类型 id ∈ [0, NUM_TOOTH_TYPES)。上颌 0..7，下颌 8..15。"""
    tid = int(tooth_id)
    if tid in (15, 16):
        pos = 8
    else:
        pos = ((tid - 1) % 7) + 1  # 1..7
    jaw_off = 0 if str(jaw).strip().lower().startswith("u") else 8
    return jaw_off + (pos - 1)


class CoarseToFineFoldingNetTyped(nn.Module):
    """类型条件 + 每类可学习模板的粗-细补全（Route A）。

    - 牙位类型 embedding 注入编码特征（IOS-only 推理时由 DDC 分割给出，不依赖 CT）。
    - 每个形态类型持有可学习粗模板 ``templates[type]``；粗点 = 模板 + 网络残差，
      把冠→根欠定问题用 per-type 解剖先验兜住。
    - ``forward(crown_b3n, type_ids)`` → ``(fine, coarse)``；``type_ids=None`` 时退化为
      跨类型平均模板 + 零 embedding（供无类型信息的评估回退）。
    """

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_output_points: int = 4096,
        decoder_hidden: int = 2048,
        num_coarse: int = 256,
        patch_grid_size: int = 4,
        num_types: int = NUM_TOOTH_TYPES,
        type_embed_dim: int = 32,
        template_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        k = patch_grid_size * patch_grid_size
        if num_coarse * k != num_output_points:
            raise ValueError(
                f"num_coarse * patch_grid_size^2 须等于 num_output_points："
                f"{num_coarse}*{k} != {num_output_points}"
            )
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.patch_grid_size = patch_grid_size
        self.patch_k = k
        self.num_output_points = num_output_points
        self.num_types = int(num_types)
        self.type_embed_dim = int(type_embed_dim)

        self.encoder = PointNetEncoder(3, latent_dim)
        self.type_embed = nn.Embedding(self.num_types, self.type_embed_dim)
        nn.init.normal_(self.type_embed.weight, std=0.02)

        tpl = torch.empty(self.num_types, num_coarse, 3)
        if template_init is not None:
            tpl.copy_(template_init)
        else:
            nn.init.normal_(tpl, std=0.3)
        self.templates = nn.Parameter(tpl)

        cond = latent_dim + self.type_embed_dim
        self.coarse_mlp = nn.Sequential(
            nn.Linear(cond, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, num_coarse * 3),
        )
        u = torch.linspace(-0.05, 0.05, patch_grid_size)
        gu, gv = torch.meshgrid(u, u, indexing="ij")
        patch = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)
        self.register_buffer("patch_uv", patch.unsqueeze(0).unsqueeze(0))

        in_dim = 2 + cond + 3
        self.local_fold = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )

    def _cond(self, z: torch.Tensor, type_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        if type_ids is None:
            e = self.type_embed.weight.new_zeros(B, self.type_embed_dim)
            tpl = self.templates.mean(dim=0, keepdim=True).expand(B, -1, -1)
        else:
            e = self.type_embed(type_ids)
            tpl = self.templates[type_ids]
        return torch.cat([z, e], dim=-1), tpl

    def forward(
        self, crown_b3n: torch.Tensor, type_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(crown_b3n)
        zc, tpl = self._cond(z, type_ids)
        B, C = zc.shape
        S, K = self.num_coarse, self.patch_k
        coarse = tpl + self.coarse_mlp(zc).view(B, S, 3)
        g = self.patch_uv.expand(B, S, K, 2)
        cexp = coarse.unsqueeze(2).expand(B, S, K, 3)
        zexp = zc.view(B, 1, 1, C).expand(B, S, K, C)
        feat = torch.cat([g, zexp, cexp], dim=-1)
        off = self.local_fold(feat.reshape(B * S * K, -1)).view(B, S, K, 3)
        fine = (cexp + off).reshape(B, S * K, 3)
        return fine, coarse


def farthest_point_sample_indices(pts: torch.Tensor, n: int) -> torch.Tensor:
    """``(P, 3)`` 上 FPS，返回长度为 ``n`` 的点下标 ``(n,)`` long。"""
    p = int(pts.size(0))
    if p < n:
        raise ValueError(f"FPS: 点数 P={p} < n={n}")
    device, dtype = pts.device, pts.dtype
    chosen = torch.zeros(n, dtype=torch.long, device=device)
    dist = torch.full((p,), float("inf"), device=device, dtype=dtype)
    far = int(torch.randint(0, p, (1,), device=device).item())
    for i in range(n):
        chosen[i] = far
        ctr = pts[far : far + 1]
        d = ((pts - ctr) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        far = int(dist.argmax().item())
    return chosen


def farthest_point_sample_indices_batch(pts_bnc: torch.Tensor, n: int) -> torch.Tensor:
    """``(B, P, 3)`` → 每样本 FPS，下标 ``(B, n)`` long。"""
    B = int(pts_bnc.size(0))
    out = torch.empty(B, n, dtype=torch.long, device=pts_bnc.device)
    for b in range(B):
        out[b] = farthest_point_sample_indices(pts_bnc[b], n)
    return out


def _gather_knn_neighbors(
    pts_bnc: torch.Tensor,
    centers_bnc: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """``pts_bnc`` (B,N,3)，``centers_bnc`` (B,C,3) → 相对中心点的 K 近邻偏移 ``(B,C,K,3)``。"""
    dist = torch.cdist(centers_bnc, pts_bnc, p=2.0)
    _, idx = dist.topk(min(k, pts_bnc.size(1)), dim=-1, largest=False)
    # (B, C, K)
    b, c, kk = idx.shape
    expanded = pts_bnc.unsqueeze(1).expand(b, c, pts_bnc.size(1), 3)
    idx_e = idx.unsqueeze(-1).expand(b, c, kk, 3)
    neigh = torch.gather(expanded, 2, idx_e)
    rel = neigh - centers_bnc.unsqueeze(2)
    return rel


class TransformerCompletionNet(nn.Module):
    """PoinTr-lite：牙冠 FPS 代理 → 局部几何 Token → TransformerEncoder；

    可学习 Queries + TransformerDecoder（cross-attn）→ 粗骨架点；再每粗点 patch 折叠上采样至 ``num_output_points``。

    **``forward(crown_b3n) -> (fine_pts, coarse_pts)``**，与 ``CoarseToFineFoldingNet`` 一致，可直接接 ``masked_crown_root_loss_coarse_fine``。
    """

    def __init__(
        self,
        *,
        num_output_points: int = 4096,
        num_coarse: int = 256,
        patch_grid_size: int = 4,
        n_proxy: int = 128,
        d_model: int = 384,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1536,
        decoder_hidden: int = 2048,
        knn_k: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        k_patch = patch_grid_size * patch_grid_size
        if num_coarse * k_patch != num_output_points:
            raise ValueError(
                f"num_coarse * patch_grid_size^2 须等于 num_output_points："
                f"{num_coarse}*{k_patch} != {num_output_points}"
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 须能被 nhead={nhead} 整除")

        self.num_output_points = num_output_points
        self.num_coarse = num_coarse
        self.patch_grid_size = patch_grid_size
        self.patch_k = k_patch
        self.n_proxy = n_proxy
        self.d_model = d_model
        self.knn_k = knn_k

        self.group_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
        )
        self.pos_proj = nn.Linear(3, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.query_embed = nn.Parameter(torch.randn(num_coarse, d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.coarse_head = nn.Sequential(
            nn.Linear(d_model, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )

        u = torch.linspace(-0.05, 0.05, patch_grid_size)
        gu, gv = torch.meshgrid(u, u, indexing="ij")
        patch = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)
        self.register_buffer("patch_uv", patch.unsqueeze(0).unsqueeze(0))

        in_dim = 2 + d_model + 3
        self.local_fold = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 3),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, crown_b3n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``crown_b3n``: (B, 3, N_in) → ``fine`` (B, S*K, 3), ``coarse`` (B, S, 3)。"""
        B, _, n_in = crown_b3n.shape
        pts = crown_b3n.transpose(1, 2).contiguous()

        idx = farthest_point_sample_indices_batch(pts, self.n_proxy)
        batch = torch.arange(B, device=pts.device).unsqueeze(-1).expand_as(idx)
        proxy_xyz = pts[batch, idx]

        k_eff = min(self.knn_k, n_in)
        rel = _gather_knn_neighbors(pts, proxy_xyz, k_eff)
        if rel.size(2) < self.knn_k:
            pad = rel.new_zeros(B, self.n_proxy, self.knn_k - rel.size(2), 3)
            rel = torch.cat([rel, pad], dim=2)
        h = self.group_mlp(rel.reshape(-1, 3)).view(B, self.n_proxy, self.knn_k, self.d_model).max(dim=2).values
        tok = h + self.pos_proj(proxy_xyz)
        mem = self.tf_encoder(tok)

        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        dec = self.tf_decoder(q, mem)

        coarse = self.coarse_head(dec)
        S, K = self.num_coarse, self.patch_k
        g = self.patch_uv.expand(B, S, K, 2)
        cexp = coarse.unsqueeze(2).expand(B, S, K, 3)
        dexp = dec.unsqueeze(2).expand(B, S, K, self.d_model)
        feat = torch.cat([g, dexp, cexp], dim=-1)
        off = self.local_fold(feat.reshape(B * S * K, -1)).view(B, S, K, 3)
        fine = (cexp + off).reshape(B, S * K, 3)
        return fine, coarse


def compute_repulsion_loss(pred_pts: torch.Tensor, k: int = 4, h: float = 0.01) -> torch.Tensor:
    """互斥损失：鼓励预测点在局部彼此拉开，减轻块状聚集。

    对 batch 内每个样本用 ``torch.cdist`` 得 (N,N)，每行取除自身外 Top-K 最近邻距离
    ``d_ij``，再计算 ``mean(max(0, h - d_ij))``，最后在 (B, N, K) 上平均。

    pred_pts: (B, N, 3)，返回标量 tensor。
    """
    if pred_pts.dim() != 3 or pred_pts.size(-1) != 3:
        raise ValueError(f"pred_pts 须为 (B,N,3)，当前 shape={tuple(pred_pts.shape)}")
    B, N, _ = pred_pts.shape
    if N <= 1:
        return pred_pts.new_tensor(0.0)
    k = int(k)
    k_eff = min(k, N - 1)
    if k_eff < 1:
        return pred_pts.new_tensor(0.0)
    h_t = pred_pts.new_tensor(float(h))
    acc = pred_pts.new_tensor(0.0)
    for b in range(B):
        pb = pred_pts[b]
        d = torch.cdist(pb, pb, p=2.0)
        vals, _ = torch.topk(d, k=k_eff + 1, dim=-1, largest=False)
        nn_d = vals[:, 1 : 1 + k_eff]
        acc = acc + torch.relu(h_t - nn_d).sum()
    return acc / float(B * N * k_eff)


def density_aware_chamfer_pair(
    a: torch.Tensor, b: torch.Tensor, *, alpha: float = 200.0, n_lambda: float = 0.5
) -> torch.Tensor:
    """Density-aware Chamfer Distance（Wu et al. 2021）单对版本，a (P,3), b (Q,3)。

    相比 Chamfer L²，DCD 对「多点挤到同一目标」显式惩罚（除以匹配频次），能抑制根尖
    点云聚团/发散。距离为欧氏（非平方），配合单位球归一化 alpha≈100–400。空集返回 NaN。
    """
    if a.numel() == 0 or b.numel() == 0:
        return a.new_tensor(float("nan"))
    d = torch.cdist(a, b, p=2.0)  # (P,Q) 欧氏
    min_ab, idx_ab = d.min(dim=1)  # a→b: (P,)
    min_ba, idx_ba = d.min(dim=0)  # b→a: (Q,)
    p, q = a.size(0), b.size(0)
    count_b = torch.zeros(q, device=a.device, dtype=a.dtype)
    count_b.scatter_add_(0, idx_ab, torch.ones_like(min_ab))
    count_a = torch.zeros(p, device=a.device, dtype=a.dtype)
    count_a.scatter_add_(0, idx_ba, torch.ones_like(min_ba))
    w_ab = count_b[idx_ab].clamp(min=1.0).pow(-n_lambda)
    w_ba = count_a[idx_ba].clamp(min=1.0).pow(-n_lambda)
    loss_ab = (1.0 - w_ab * torch.exp(-alpha * min_ab)).mean()
    loss_ba = (1.0 - w_ba * torch.exp(-alpha * min_ba)).mean()
    return 0.5 * (loss_ab + loss_ba)


def farthest_point_sample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """pts (P,3) → (min(P,n),3)，最远点采样（GT 目标下采样）。"""
    p = int(pts.size(0))
    if p <= n:
        return pts
    device, dtype = pts.device, pts.dtype
    chosen = torch.zeros(n, dtype=torch.long, device=device)
    dist = torch.full((p,), float("inf"), device=device, dtype=dtype)
    far = int(torch.randint(0, p, (1,), device=device).item())
    for i in range(n):
        chosen[i] = far
        ctr = pts[far : far + 1]
        d = ((pts - ctr) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        far = int(dist.argmax().item())
    return pts[chosen]


def masked_crown_root_loss_coarse_fine(
    pred_fine: torch.Tensor,
    pred_coarse: torch.Tensor,
    gt: torch.Tensor,
    crown_b3n: torch.Tensor,
    scale_radius_mm: torch.Tensor,
    crown_margin_mm: float,
    *,
    n_coarse_gt: int = 256,
    alpha_fine: float = 1.0,
    weight_repulsion: float = 0.0,
    repulsion_k: int = 4,
    repulsion_h: float = 0.01,
    scale_invariant_root: bool = False,
    dcd_weight: float = 0.0,
    dcd_alpha: float = 200.0,
    dcd_n_lambda: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """粗-细 masked 训练：``loss = loss_coarse + alpha_fine * loss_root_fine``；指标与细点云一致。

    - loss_coarse：``Chamfer(pred_coarse[b], FPS/子采样 GT 根区 → n_coarse_gt)``。
    - loss_root_fine：与 ``masked_crown_root_loss_and_metrics`` 中根区 Chamfer 相同（作用于 pred_fine）。
    - 当 ``weight_repulsion > 0`` 时加上 ``weight_repulsion * compute_repulsion_loss(pred_fine)``（默认 0 不改变 Baseline）。
    返回第五项为未加权的 ``L_rep``（``weight_repulsion==0`` 时为 NaN，便于日志区分是否计算）。
    """
    loss_fine, m_global, m_crown, m_root = masked_crown_root_loss_and_metrics(
        pred_fine,
        gt,
        crown_b3n,
        scale_radius_mm,
        crown_margin_mm,
        scale_invariant_root=scale_invariant_root,
    )

    bsz, device, dtype = pred_coarse.size(0), pred_coarse.device, pred_coarse.dtype
    eps = epsilon_norm_ball(scale_radius_mm, crown_margin_mm)
    eps2 = (eps * eps).view(bsz, 1)
    d2_gt = min_sq_dist_to_crown(gt, crown_b3n)
    gt_root_m = d2_gt > eps2

    coarse_terms: list[torch.Tensor] = []
    for b in range(bsz):
        gg = gt[b]
        gtr = gt_root_m[b]
        gt_r = gg[gtr]
        pc = pred_coarse[b]
        if gt_r.size(0) < 4:
            continue
        tgt = farthest_point_sample(gt_r, min(n_coarse_gt, int(gt_r.size(0))))
        if scale_invariant_root:
            ext = gt_root_extent_norm(gt_r)
            coarse_terms.append(chamfer_l2_pair_scale_invariant(pc, tgt, ext))
        else:
            coarse_terms.append(chamfer_l2_pair(pc, tgt))

    if coarse_terms:
        loss_coarse = torch.stack(coarse_terms).mean()
    else:
        loss_coarse = pred_coarse.new_tensor(0.0)

    loss = loss_coarse + float(alpha_fine) * loss_fine

    if float(dcd_weight) > 0.0:
        d2_pr = min_sq_dist_to_crown(pred_fine, crown_b3n)
        pr_root_m = d2_pr > eps2
        dcd_terms: list[torch.Tensor] = []
        for b in range(bsz):
            gt_r = gt[b][gt_root_m[b]]
            pr_r = pred_fine[b][pr_root_m[b]]
            if gt_r.size(0) < 4 or pr_r.size(0) < 4:
                continue
            dcd_terms.append(
                density_aware_chamfer_pair(
                    pr_r, gt_r, alpha=float(dcd_alpha), n_lambda=float(dcd_n_lambda)
                )
            )
        if dcd_terms:
            loss = loss + float(dcd_weight) * torch.stack(dcd_terms).mean()

    if float(weight_repulsion) > 0.0:
        l_rep = compute_repulsion_loss(pred_fine, k=int(repulsion_k), h=float(repulsion_h))
        loss = loss + float(weight_repulsion) * l_rep
    else:
        l_rep = pred_fine.new_tensor(float("nan"))
    return loss, m_global, m_crown, m_root, l_rep


# 4096 点 × 大 batch 时，完整 (B,Q,N) cdist 会占数 GB；分块降低峰值显存。
_CHAMFER_CHUNK = 512
_MINCROWN_CHUNK = 512


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor, *, chunk: int = _CHAMFER_CHUNK) -> torch.Tensor:
    """双向 Chamfer（平方距离），pred (B,P,3), target (B,Q,3)。按 batch + 点维分块，降低峰值显存。"""
    B, P, _ = pred.shape
    _, Q, _ = target.shape
    l1s = torch.empty(B, device=pred.device, dtype=pred.dtype)
    l2s = torch.empty(B, device=pred.device, dtype=pred.dtype)
    for b in range(B):
        pr = pred[b : b + 1]
        tg = target[b : b + 1]
        parts1: list[torch.Tensor] = []
        for i in range(0, P, chunk):
            pchunk = pr[:, i : i + chunk, :]
            d = torch.cdist(pchunk, tg, p=2.0).pow(2).min(dim=2).values
            parts1.append(d)
        l1s[b] = torch.cat(parts1, dim=1).mean()

        parts2: list[torch.Tensor] = []
        for j in range(0, Q, chunk):
            tchunk = tg[:, j : j + chunk, :]
            d = torch.cdist(pr, tchunk, p=2.0).pow(2).min(dim=1).values
            parts2.append(d)
        l2s[b] = torch.cat(parts2, dim=1).mean()
    return (l1s + l2s).mean()


def min_sq_dist_to_crown(
    pts_bqg: torch.Tensor,
    crown_b3n: torch.Tensor,
    *,
    chunk_q: int = _MINCROWN_CHUNK,
) -> torch.Tensor:
    """pts (B,Q,3), crown_b3n (B,3,N) -> (B,Q) 每个点到冠点集的最小平方距离（单位球坐标下）。

    按样本 + Q 分块，避免 (B,Q,N) 同时驻留（大 B 与大 N 时曾导致 OOM）。
    """
    B, Q, _ = pts_bqg.shape
    c = crown_b3n.transpose(1, 2).contiguous()
    out = torch.empty(B, Q, device=pts_bqg.device, dtype=pts_bqg.dtype)
    for b in range(B):
        pb = pts_bqg[b : b + 1]
        cb = c[b : b + 1]
        for qs in range(0, Q, chunk_q):
            qe = min(qs + chunk_q, Q)
            d2 = torch.cdist(pb[:, qs:qe, :], cb, p=2.0).pow(2)
            out[b, qs:qe] = d2.min(dim=2).values.squeeze(0)
    return out


def epsilon_norm_ball(scale_radius_mm: torch.Tensor, crown_margin_mm: float) -> torch.Tensor:
    """物理距离 crown_margin_mm 在单位球中的等效半径阈值：epsilon = margin_mm / r_mm。shape (B,)。"""
    return crown_margin_mm / scale_radius_mm.clamp(min=1e-6)


def chamfer_l2_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """单对点集双向 Chamfer L²，a (P,3), b (Q,3)。空集返回 NaN。"""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_tensor(float("nan"))
    d = torch.cdist(a, b, p=2.0).pow(2)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


def gt_root_extent_norm(gt_root: torch.Tensor) -> torch.Tensor:
    """GT 根区在归一化坐标下的径向尺度（max ||p||），用于 ios_crown 尺度不变根区 Chamfer。"""
    if gt_root.numel() == 0:
        return gt_root.new_tensor(1.0)
    return torch.linalg.norm(gt_root, dim=1).max().clamp(min=1e-4)


def chamfer_l2_pair_scale_invariant(
    a: torch.Tensor, b: torch.Tensor, extent: torch.Tensor
) -> torch.Tensor:
    """将 pred/gt 同除以 extent 后算 Chamfer，等价于 Chamfer / extent²。"""
    inv = (1.0 / extent).clamp(max=1e4)
    return chamfer_l2_pair(a * inv, b * inv)


def _root_chamfer_terms(
    pr_r: torch.Tensor,
    gt_r: torch.Tensor,
    pg: torch.Tensor,
    *,
    scale_invariant_root: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单样本根区 Chamfer：(loss_term, metric_term)。"""
    extent = gt_root_extent_norm(gt_r) if scale_invariant_root else None

    def _pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if extent is not None:
            return chamfer_l2_pair_scale_invariant(a, b, extent)
        return chamfer_l2_pair(a, b)

    if pr_r.size(0) >= 4:
        t = _pair(pr_r, gt_r)
        return t, t
    if pr_r.size(0) == 0:
        d = torch.cdist(pg.unsqueeze(0), gt_r.unsqueeze(0), p=2.0).pow(2).squeeze(0)
        t = d.min(dim=0)[0].mean() * 2.0
        if extent is not None:
            inv2 = (1.0 / extent).pow(2).clamp(max=1e8)
            t = t * inv2
        return t, pg.new_tensor(float("nan"))
    t = _pair(pr_r, gt_r)
    return t, t


def masked_crown_root_loss_and_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    crown_b3n: torch.Tensor,
    scale_radius_mm: torch.Tensor,
    crown_margin_mm: float,
    *,
    scale_invariant_root: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """冠根按到 IOS 冠的最短距离与 epsilon=margin/r 划分；训练目标为 Chamfer(Pred_Root, GT_Root)。

    返回 (loss_root, m_global, m_crown, m_root)，除 loss_root 外均为 batch 平均（可含 nan）。

    pred: (B,P,3), gt: (B,G,3), crown_b3n: (B,3,Nin), scale_radius_mm: (B,)
    scale_invariant_root: 根区 Chamfer 按每样本 gt_root_extent² 归一化，减轻小牙权重偏低。
    """
    bsz, device, dtype = pred.size(0), pred.device, pred.dtype
    eps = epsilon_norm_ball(scale_radius_mm, crown_margin_mm)
    eps2 = (eps * eps).view(bsz, 1)

    d2_gt = min_sq_dist_to_crown(gt, crown_b3n)
    d2_pr = min_sq_dist_to_crown(pred, crown_b3n)
    gt_root_m = d2_gt > eps2
    gt_crown_m = ~gt_root_m
    pr_root_m = d2_pr > eps2
    pr_crown_m = ~pr_root_m

    root_loss_terms: list[torch.Tensor] = []
    glob_terms: list[torch.Tensor] = []
    crown_terms: list[torch.Tensor] = []
    root_metric_terms: list[torch.Tensor] = []

    for b in range(bsz):
        pg, gg = pred[b], gt[b]
        glob_terms.append(chamfer_l2_pair(pg, gg))

        gtr, gtc = gt_root_m[b], gt_crown_m[b]
        prr, prc = pr_root_m[b], pr_crown_m[b]
        pr_r, gt_r = pg[prr], gg[gtr]
        pc, gc = pg[prc], gg[gtc]

        # crown 参考指标
        if pc.size(0) >= 4 and gc.size(0) >= 4:
            crown_terms.append(chamfer_l2_pair(pc, gc))
        else:
            crown_terms.append(pg.new_tensor(float("nan")))

        # root loss（GT 根区过少则不贡献该项）
        if gt_r.size(0) >= 4:
            loss_t, metric_t = _root_chamfer_terms(
                pr_r, gt_r, pg, scale_invariant_root=scale_invariant_root
            )
            root_loss_terms.append(loss_t)
            root_metric_terms.append(metric_t)

    def nanmean_stack(tlist: list[torch.Tensor]) -> torch.Tensor:
        if not tlist:
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        return torch.nanmean(torch.stack(tlist))

    m_global = nanmean_stack(glob_terms)

    def _finite_mean(tlist: list[torch.Tensor]) -> torch.Tensor:
        if not tlist:
            return pred.new_tensor(float("nan"))
        t = torch.stack(tlist)
        t = t[torch.isfinite(t)]
        return t.mean() if t.numel() else pred.new_tensor(float("nan"))

    m_crown = _finite_mean(crown_terms)
    m_root = _finite_mean(root_metric_terms)

    if root_loss_terms:
        loss_root = torch.stack(root_loss_terms).mean()
    else:
        loss_root = chamfer_l2(pred, gt)

    return loss_root, m_global, m_crown, m_root


def chamfer_l2_root_aux(
    pred: torch.Tensor,
    gt: torch.Tensor,
    crown_b3n: torch.Tensor,
    quantile: float = 0.65,
) -> torch.Tensor:
    """对 GT 上距牙冠表面较远（启发式根区）的点做双向 Chamfer，按 batch 内有根区样本平均。

    pred: (B,P,3), gt: (B,G,3), crown_b3n: (B,3,Nin)
    """
    crown = crown_b3n.transpose(1, 2).contiguous()
    bsz = pred.size(0)
    acc = pred.new_tensor(0.0)
    n_ok = 0
    for b in range(bsz):
        dmin = torch.cdist(gt[b : b + 1], crown[b : b + 1], p=2.0).pow(2).min(dim=2).values.squeeze(0)
        thr = torch.quantile(dmin, quantile)
        m = dmin >= thr
        if int(m.sum().item()) < 4:
            continue
        gt_r = gt[b : b + 1, m, :]
        pr_b = pred[b : b + 1]
        acc = acc + chamfer_l2(pr_b, gt_r)
        n_ok += 1
    if n_ok == 0:
        return pred.new_tensor(0.0)
    return acc / float(n_ok)
