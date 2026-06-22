# Ball Pivoting Algorithm (BPA)

This document describes how to run the BPA surface reconstruction script and the required dependencies.

## Overview

The BPA script reconstructs a triangle mesh from a PLY point cloud using the Ball Pivoting Algorithm. It mirrors the validated MeshLab v3 pipeline:

Resampling → Compute Normals → Orient Normals → Surface Reconstruction

A virtual ball of radius r is rolled over the point cloud. Wherever it simultaneously touches three points, a triangle is formed. The radius must be chosen relative to the local point spacing — too small leaves holes, too large merges separate surfaces.

## Requirements

See the top-level `requirements.txt` for the dependency list.

## Configuration

All parameters are set in `configs/bpa_config.yaml`. Key entries:

| Key | Description |
|---|---|
| `ply_path` | Path to the input PLY point cloud |
| `out_path` | Path for the output mesh PLY |
| `voxel_size` | Voxel downsampling size in metres (null to skip). Set to match Poisson-Disk r used in MeshLab (e.g. 0.030 for r=30mm) |
| `estimate_normals` | true to (re-)estimate normals, false to use existing ones |
| `normal_max_nn` | k-nearest neighbours for PCA normal estimation (MeshLab default: 16) |
| `normal_radius` | Hybrid search radius for normals in metres (recommended: ~3× voxel_size) |
| `orient_normals` | true to orient normals via tangent-plane propagation |
| `orient_normals_k` | k-neighbours for orientation propagation |
| `radii` | Explicit list of ball radii in metres, e.g. [0.05]. Leave empty ([]) to auto-compute |
| `radii_factors` | Multipliers applied to avg nn-distance when radii is empty |
| `remove_duplicates` | true to remove duplicate vertices/triangles after BPA |
| `color_knn` | Nearest neighbours into the original (full-resolution) cloud for color transfer |
| `n_threads` | Threads for KDTree color transfer (-1 = all cores) |

### Choosing the ball radius

The validated configuration uses:
- Voxel downsampling: 30 mm → uniform point spacing
- BPA radius: 50 mm (≈1.67× point spacing)

If you change `voxel_size`, adjust `radii` proportionally, or leave `radii` empty to let the script auto-compute from the average nearest-neighbour distance.

## Run

1. Activate your Python environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Edit `configs/bpa_config.yaml` with your paths and parameters.
4. Run from the repository root:

```
python scripts/reconstruction/run_ball_pivoting_reconstruction.py
```

## Pipeline steps

1. **Load** — reads the PLY point cloud from `ply_path`
2. **Resampling** — voxel downsampling to uniform point spacing (skipped if `voxel_size` is null)
3. **Normal estimation** — PCA normals with hybrid radius/kNN search
4. **Orient normals** — consistent tangent-plane propagation
5. **Ball Pivoting** — triangle mesh from the downsampled cloud
6. **Post-processing** — remove duplicate vertices, triangles, and degenerate triangles
7. **Color transfer** — nearest-neighbour lookup from the original (full-resolution) cloud
8. **Save** — writes the mesh to `out_path` (output directory is created automatically)
