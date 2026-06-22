# LiDAR to Mesh using Neural Kernel Surface Reconstruction

## Related work

- [Neural Kernel Surface Reconstruction](https://github.com/nv-tlabs/NKSR)
- [fVDB](https://github.com/openvdb/fvdb-core)
- [PCD Meshing](https://github.com/cvg/pcdmeshing)
- [POCO](https://github.com/valeoai/POCO)
- [VDBFusion](https://github.com/PRBonn/vdbfusion)

## Setup

### Repository

Clone the repository:
```bash
git clone git@github.com:dav1dclara/3d-vision.git
cd 3d-vision/
```

### Dependencies

Create a Python environment, then install the dependencies:
```bash
conda create -n 3DV python=3.10
conda activate 3DV
pip install -r requirements.txt
# GPU/external deps (torch, nksr, pcdmeshing) come from the team's shared env:
echo "/work/courses/3dv/team13/miniconda13/envs/nksr/lib/python3.10/site-packages/" > ~/miniconda3/envs/3DV/lib/python3.10/site-packages/shared_env.pth
```

Install the pre-commit hooks for automatic code formatting and linting on each commit:
```bash
pre-commit install
```

## Reconstruction Pipeline

The reconstruction pipeline converts a LiDAR point cloud into a colored triangle mesh using a geometry-aware chunking strategy combined with Neural Kernel Surface Reconstruction (NKSR).

### Overview
```
pointcloud.las + pointcloud.ply
        │
        ▼
Fine voxelization (0.25m cells)
        │
        ▼
Per-cell planarity analysis (SVD)
        │
        ▼
Region growing
  ├── Complex regions grow first, absorbing all adjacent planar chunks
  └── Remaining planar chunks form isolated planar-only regions
        │
        ▼
NKSR reconstruction per region
  ├── Complex units → high detail (vox=0.05m, detail=1.0, mise=2)
  └── Planar-only units → lower detail (vox=0.15-0.2m, detail=0.3-0.5)
        │
        ▼
Merge all chunks → nksr_reconstruction.ply
```

### Key design decisions

**Geometry-aware chunking** — instead of splitting the scene into a regular grid (which cuts through surfaces and creates boundary artifacts), the scene is first split into uniform 0.25m cells, then each cell's planarity is estimated using PCA/SVD. The ratio of the smallest to largest singular value gives a residual — low residual means flat, high residual means complex geometry.

**Complex-first region growing** — complex regions (objects, furniture, people) grow first and greedily absorb all neighbouring planar chunks (walls, floors, ceilings). This means NKSR sees the full context of every edge — the wall and the object in the same reconstruction call — producing clean transitions without stitching artifacts. Only planar chunks that are never touched by a complex region form their own isolated planar reconstruction units.

**Adaptive quality** — complex units are reconstructed at the highest detail level with a fine voxel size (0.05m). Planar-only units use coarser voxels (0.15-0.2m) since flat surfaces reconstruct well at lower resolution, saving significant GPU time.

**Normals from PLY** — the NavVis scanner stores precomputed normals directly in the PLY file, which are used directly instead of estimating them from scratch (saving several minutes per run).

### Configuration

All parameters are set in `configs/nksr_config.yaml`

### Running
```bash
conda activate 3DV
python scripts/reconstruction/run_nksr_reconstruction.py
```

Output is written to `outputs/nksr_reconstruction.ply`. Individual chunk PLYs are saved to `outputs/chunks/` during reconstruction and can be inspected separately.

### Output

The pipeline produces a colored triangle mesh in PLY format with per-vertex RGB colors projected from the original LiDAR scan. The mesh can be viewed in CloudCompare, MeshLab, or using the provided viewer script:
```bash
python scripts/visualization/view_mesh.py outputs/nksr_reconstruction.ply
```

## Mesh Quality Assessment

Evaluate a reconstructed mesh against a LiDAR ground-truth point cloud. Computes cloud-to-mesh distances, residual distribution, aspect ratios, and produces an interactive 3D heatmap.

### Running (GUI)
```bash
conda activate 3DV
python scripts/evaluation/run_quality_assessment.py
```

Select a mesh (`.ply`) and a point cloud (`.ply`), set the desired sample size and thresholds, then click **Evaluate**.

### Notes on large point clouds
The loader uses `numpy.memmap` to randomly sample the point cloud without loading the full file into RAM. Only the requested number of points (default 50 000) are paged in from disk — a 800 M-point cloud (≈ 20 GB) requires only ≈ 200 MB of physical RAM during loading.

Distance queries use Open3D's `RaycastingScene` (C++ BVH, float32 internally) instead of trimesh, making the computation safe for meshes with tens of millions of faces.

### Metrics
| Metric | Description |
|---|---|
| Hausdorff / RMSE / MAE | Cloud-to-mesh distance statistics |
| Residual distribution | % of points in Good / OK / Critical / Missing bands |
| Mean aspect ratio | Triangle quality (sampled, 10 000 faces) |
| Degenerate triangles | Faces with area < 1e-10 (extrapolated from sample) |
| Watertight / Manifold | Topology check (optional, vectorized) |
| F-Score | Bidirectional matching precision/recall (optional) |

## VDBFusion Experiment

This repository now includes an experimental VDBFusion pipeline inspired by the KITTI odometry notebook.

Install VDBFusion first:
```bash
pip install vdbfusion
```

Point `input.las_path` in `configs/vdbfusion_config.yaml` at your LAS point cloud, then run:
```bash
python scripts/reconstruction/run_vdbfusion_reconstruction.py --config configs/vdbfusion_config.yaml
```

Outputs:
- `outputs/vdbfusion_reconstruction.ply`
- `outputs/vdbfusion_reconstruction.vdb`
