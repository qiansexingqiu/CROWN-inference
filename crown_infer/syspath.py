"""sys.path helpers so vendored mint / nnU-Net / DMS modules import without lab roots."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
VENDOR_MINT = PACK_ROOT / "vendor" / "mint"
VENDOR_NNUNET = PACK_ROOT / "vendor" / "nnunet"
VENDOR_DMS = PACK_ROOT / "vendor" / "DentalModelSegmentation"
BUNDLED_PLANS = VENDOR_NNUNET / "nnUNetPlans.json"


def add_vendor_mint() -> None:
    p = str(VENDOR_MINT)
    if p not in sys.path:
        sys.path.insert(0, p)


def add_vendor_nnunet() -> None:
    p = str(VENDOR_NNUNET)
    if p not in sys.path:
        sys.path.insert(0, p)


def prepend_torch_lib() -> None:
    """Make vendored pointnet2_cuda find libc10 from the active PyTorch."""
    try:
        import torch
    except ImportError:
        return
    lib = Path(torch.__file__).resolve().parent / "lib"
    if not lib.is_dir():
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = str(lib)
    if cur.split(os.pathsep)[0] != prefix:
        os.environ["LD_LIBRARY_PATH"] = prefix + (os.pathsep + cur if cur else "")


def add_vendor_dms() -> None:
    prepend_torch_lib()
    for p in (
        VENDOR_DMS / "src" / "pointnet2",
        VENDOR_DMS / "src",
        VENDOR_DMS,
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def dms_model_root(weights: Path) -> Path:
    return Path(weights) / "third_party" / "DentalModelSegmentation"


def default_weights_dir() -> Path | None:
    env = os.environ.get("CROWN_WEIGHTS", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (PACK_ROOT.parent / "weights", PACK_ROOT / "weights"):
        if (candidate / "CROWN-base").is_dir():
            return candidate.resolve()
    return None
