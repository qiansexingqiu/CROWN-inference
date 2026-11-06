"""Mesh / label IO for a single jaw. No hospital identifiers, no lab paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .voxel import fdi_to_sequential


def _load_trimesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = tuple(mesh.geometry.values())
        if not geoms:
            raise ValueError(f"empty scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def read_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (N,3) vertices and optional (N,3) RGB in [0, 1]."""
    path = Path(path)
    if path.suffix.lower() == ".off":
        verts = _read_off_vertices(path)
        return verts, None
    mesh = _load_trimesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    colors = None
    vis = getattr(mesh, "visual", None)
    vc = getattr(vis, "vertex_colors", None) if vis is not None else None
    if vc is not None and len(vc) == len(verts):
        arr = np.asarray(vc, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            rgb = arr[:, :3]
            if rgb.max(initial=0) > 1.0:
                rgb = rgb / 255.0
            colors = np.clip(rgb, 0.0, 1.0)
    return verts, colors


def _read_off_vertices(off_path: Path) -> np.ndarray:
    with open(off_path, encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"not an OFF file: {off_path}")
        nv, _nf, _ne = map(int, f.readline().split())
        verts = np.empty((nv, 3), dtype=np.float64)
        for i in range(nv):
            verts[i] = list(map(float, f.readline().split()[:3]))
    return verts


def read_labels(path: Path, n_verts: int) -> np.ndarray:
    raw = np.loadtxt(Path(path), dtype=np.int32)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if len(raw) != n_verts:
        raise ValueError(
            f"label count {len(raw)} != vertex count {n_verts} ({path})"
        )
    return raw


def to_sequential(labels: np.ndarray) -> np.ndarray:
    """FDI (11, 21, …) → sequential 1–14. Sequential labels are left as-is."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0:
        return labels
    if int(labels.max(initial=0)) >= 17:
        return fdi_to_sequential(labels)
    return labels


def write_ply_xyz(path: Path, pts: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for x, y, z in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            return
        rgb = np.asarray(rgb)
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
