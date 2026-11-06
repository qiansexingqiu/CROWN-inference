#!/usr/bin/env python3
"""Build nnU-Net PlainConvUNet for 7→1 SDF regression (no segmentation head)."""
from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def build_nnunet_sdf_backbone(
    plans_json: Path,
    *,
    configuration: str = "3d_fullres",
    in_channels: int = 7,
    out_channels: int = 1,
) -> nn.Module:
    plans = json.loads(plans_json.read_text(encoding="utf-8"))
    pm = PlansManager(plans)
    cm = pm.get_configuration(configuration)
    return get_network_from_plans(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        in_channels,
        out_channels,
        deep_supervision=False,
    )
