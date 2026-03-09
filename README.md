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
conda create -n 3DV python=3.12
conda activate 3DV
pip install -r requirements.txt
pip install -e .
```

Install the pre-commit hooks for automatic code formatting and linting on each commit:

```bash
pre-commit install
```
