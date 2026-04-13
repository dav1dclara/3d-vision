# LiDAR to Mesh using Neural Kernel Surface Reconstruction

## Related work

- [Neural Kernel Surface Reconstruction](https://github.com/nv-tlabs/NKSR)
- [fVDB](https://github.com/openvdb/fvdb-core)
- [PCD Meshing](https://github.com/cvg/pcdmeshing)
- [POCO](https://github.com/valeoai/POCO)

## Setup

### Repository

Clone the repository:
```bash
git clone git@github.com:dav1dclara/3d-vision.git
cd 3d-vision/
```

### Dependencies

Create a conda environment, then install the dependencies:
```bash
conda create -n 3DV --file requirements.txt
conda activate 3DV
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
Fine voxelization (0.5m cells)
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
Merge all chunks → reconstruction.ply
```

### Key design decisions

**Geometry-aware chunking** — instead of splitting the scene into a regular grid (which cuts through surfaces and creates boundary artifacts), the scene is first split into uniform 0.5m cells, then each cell's planarity is estimated using PCA/SVD. The ratio of the smallest to largest singular value gives a residual — low residual means flat, high residual means complex geometry.

**Complex-first region growing** — complex regions (objects, furniture, people) grow first and greedily absorb all neighbouring planar chunks (walls, floors, ceilings). This means NKSR sees the full context of every edge — the wall and the object in the same reconstruction call — producing clean transitions without stitching artifacts. Only planar chunks that are never touched by a complex region form their own isolated planar reconstruction units.

**Adaptive quality** — complex units are reconstructed at the highest detail level with a fine voxel size (0.05m). Planar-only units use coarser voxels (0.15-0.2m) since flat surfaces reconstruct well at lower resolution, saving significant GPU time.

**Normals from PLY** — the NavVis scanner stores precomputed normals directly in the PLY file, which are used directly instead of estimating them from scratch (saving several minutes per run).

### Configuration

All parameters are set in `config.yaml`

### Running
```bash
conda activate 3DV
python scripts/nksr_reconstruction.py
```

Output is written to `outputs/reconstruction.ply`. Individual chunk PLYs are saved to `outputs/chunks/` during reconstruction and can be inspected separately.

### Output

The pipeline produces a colored triangle mesh in PLY format with per-vertex RGB colors projected from the original LiDAR scan. The mesh can be viewed in CloudCompare, MeshLab, or using the provided viewer script:
```bash
python scripts/view_mesh.py outputs/reconstruction.ply
```