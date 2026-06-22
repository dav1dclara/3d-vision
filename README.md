# LiDAR to Mesh using Neural Kernel Surface Reconstruction

Semester Project in 3D Vision at ETH Zurich - Spring 2026

## ⚙️ Setup

### 📦 Repository

Clone the repository:

```bash
git clone git@github.com:dav1dclara/lidar2mesh.git
cd lidar2mesh/
```

### ⛓️ Dependencies

Create a conda environment, then install the dependencies:

```bash
conda create -n 3DV python=3.10
conda activate 3DV
python -m pip install -r requirements.txt
python -m pip install -e .
```

The heavy GPU dependencies (`torch`, `nksr`, `pcdmeshing`) are not installed by the
commands above. On the team cluster they are provided by the shared `nksr` conda
environment via a site-packages `.pth` file:

```bash
echo "/work/courses/3dv/team13/miniconda13/envs/nksr/lib/python3.10/site-packages/" > ~/miniconda3/envs/3DV/lib/python3.10/site-packages/shared_env.pth
```

Install the pre-commit hooks for automatic code formatting and linting on each commit:

```bash
pre-commit install
```

## 🚀 Usage

All commands are run from the repository root. Pipeline parameters live in `configs/`.

### 🧰 Tools — `scripts/tools/`

Helper utilities: crop a region of interest from a large point cloud (pre-reconstruction), or convert a mesh into a TSDF dataset (post-reconstruction):

```bash
python scripts/tools/crop_region.py --input scene.ply --output crop.ply --center X Y Z --radius R
python scripts/tools/mesh_to_dataset.py <mesh.ply>
```

### 🏗️ Reconstruction — `scripts/reconstruction/`

Reconstruct a mesh from the point cloud. NKSR is the main method; four classical baselines are also provided.

All reconstruction parameters — input/output paths, voxel sizes, thresholds, and per-method settings — live in the [`configs/`](configs) folder, one YAML file per method (`nksr_config.yaml`, `vdbfusion_config.yaml`, `poisson_config.yaml`, `pcd_config.yaml`, `bpa_config.yaml`). Adapt them to your data before running — in particular the input point-cloud paths and output directory, which currently point at the team's cluster paths.

```bash
# NKSR (main pipeline)
python scripts/reconstruction/run_nksr_reconstruction.py

# Baselines
python scripts/reconstruction/run_vdbfusion_reconstruction.py --config configs/vdbfusion_config.yaml
python scripts/reconstruction/run_poisson_reconstruction.py
python scripts/reconstruction/run_pcd_reconstruction.py
python scripts/reconstruction/run_ball_pivoting_reconstruction.py
```

Each method writes its mesh to the `output_dir`/`output_mesh` path set in its config (`out_path` for Ball Pivoting). With the default config, NKSR produces `nksr_reconstruction.ply` plus the individual per-region meshes under `<output_dir>/chunks/`, and VDBFusion additionally writes a `.vdb` grid alongside the `.ply`.

### 📊 Evaluation — `scripts/evaluation/`

Evaluate a reconstructed mesh against a reference point cloud (interactive GUI):

```bash
python scripts/evaluation/run_quality_assessment.py
```

### 🎥 Visualization — `scripts/visualization/`

Inspect meshes, point clouds, and reconstruction chunks:

```bash
# quick Rerun view
python scripts/visualization/view_mesh.py outputs/nksr_reconstruction.ply

# render / screenshot
python scripts/visualization/viewer.py --input cloud.ply --interactive

# first-person walkthrough
python scripts/visualization/walk.py --mesh scene.ply

# browse chunks
python scripts/visualization/browse_chunks.py --chunks-dir outputs/chunks
```

## 📚 Documentation

Detailed per-method documentation (overview, configuration, tuning, and pipeline steps) lives in [`docs/`](docs):

- [NKSR](docs/README_nksr.md) — main pipeline
- [VDBFusion](docs/README_vdbfusion.md) — volumetric TSDF baseline
- [Screened Poisson](docs/README_poisson.md) — global-implicit baseline
- [pcdmeshing](docs/README_pcd.md) — block Delaunay baseline
- [Ball Pivoting](docs/README_bpa.md) — interpolating baseline
- [Quality Assessment](docs/README_quality_assessment.md) — mesh evaluation tool

## 👥 Authors

- Victor Pacheco Aznar
- Otto Scipal
- David Clara
- Jeffrey Leisi
- Luca Dominiak

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
