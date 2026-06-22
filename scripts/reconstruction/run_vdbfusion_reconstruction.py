import argparse
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add src to path so the lidar2mesh package is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lidar2mesh.vdbfusion_pipeline import build_points, reconstruct_once


def main() -> None:
    start = datetime.now()

    parser = argparse.ArgumentParser(description="VDBFusion one-shot meshing from LAS")
    parser.add_argument(
        "--config",
        default="configs/vdbfusion_config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    las_path = cfg["input"]["las_path"]
    out_dir = cfg["output"]["output_dir"]
    map_name = cfg["output"]["map_name"]

    voxel_size = float(cfg["fusion"]["voxel_size"])
    sdf_trunc = float(cfg["fusion"]["sdf_trunc"])
    space_carving = bool(cfg["fusion"]["space_carving"])
    fill_holes = bool(cfg["fusion"]["fill_holes"])
    min_weight = float(cfg["fusion"]["min_weight"])

    fixed_origin = np.asarray(
        cfg["static_mode"]["fixed_origin"], dtype=np.float64
    ).reshape(
        3,
    )
    sweep_cfg = cfg.get("sweep", {}) or {}
    sweep_enabled = bool(sweep_cfg.get("enabled", False))

    sweep_min_weights = [
        float(v) for v in (sweep_cfg.get("min_weight_values") or [min_weight])
    ]
    sweep_downsample_voxels = [
        float(v)
        for v in (
            sweep_cfg.get("downsample_voxel_sizes")
            or [cfg["preprocess"].get("downsample_voxel_size", 0.0)]
        )
    ]
    sweep_sdf_trunc_values = [
        float(v) for v in (sweep_cfg.get("sdf_trunc_values") or [sdf_trunc])
    ]
    sweep_voxel_size_values = [
        float(v) for v in (sweep_cfg.get("voxel_size_values") or [voxel_size])
    ]
    sweep_output_dir = sweep_cfg.get("output_subdir", "sweeps")

    os.makedirs(out_dir, exist_ok=True)

    for current_downsample_voxel in sweep_downsample_voxels:
        print(
            f"\n=== Reconstruction sweep: downsample_voxel={current_downsample_voxel} ==="
        )
        points = build_points(las_path, current_downsample_voxel)

        for (
            current_voxel_size,
            current_sdf_trunc,
            current_min_weight,
        ) in itertools.product(
            sweep_voxel_size_values,
            sweep_sdf_trunc_values,
            sweep_min_weights,
        ):
            print(
                "\n--- Sweep combo --- "
                f"voxel_size={current_voxel_size}, "
                f"sdf_trunc={current_sdf_trunc}, "
                f"min_weight={current_min_weight}"
            )
            reconstruct_once(
                points=points,
                fixed_origin=fixed_origin,
                voxel_size=current_voxel_size,
                sdf_trunc=current_sdf_trunc,
                space_carving=space_carving,
                fill_holes=fill_holes,
                min_weight=current_min_weight,
                out_dir=out_dir,
                map_name=map_name,
                downsample_voxel=current_downsample_voxel,
                sweep_output_dir=sweep_output_dir,
                sweep_enabled=sweep_enabled,
            )

    print(f"Total time: {datetime.now() - start}")


if __name__ == "__main__":
    print("running")
    main()
