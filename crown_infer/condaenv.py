"""Locate conda interpreters used by the paper web stack (3dpoint + nnunetv2)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def conda_python(env: str) -> list[str]:
    if os.environ.get("CROWN_SKIP_CONDA") == "1":
        return [sys.executable]
    override = os.environ.get(f"CROWN_PYTHON_{env.upper()}", "").strip()
    if override:
        return [override]
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return [sys.executable]
    return [conda_exe, "run", "-n", env, "--no-capture-output", "python"]


def env_3d() -> str:
    return os.environ.get("CROWN_ENV_3D", "3dpoint")


def env_nn() -> str:
    return os.environ.get("CROWN_ENV_NN", "nnunetv2")


def run_python(env: str, args: list[str], *, cwd: Path | None = None) -> None:
    cmd = [*conda_python(env), *args]
    extra = os.environ.copy()
    extra["CROWN_SKIP_CONDA"] = "1"
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=extra)
