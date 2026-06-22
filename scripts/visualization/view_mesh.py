"""View a triangle mesh in the Rerun viewer.

Usage:
    python scripts/visualization/view_mesh.py outputs/nksr_reconstruction.ply
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr


def main() -> None:
    parser = argparse.ArgumentParser(description="View a triangle mesh with Rerun")
    parser.add_argument("mesh", type=Path, help="Path to a mesh file (e.g. .ply)")
    args = parser.parse_args()

    if not args.mesh.exists():
        raise FileNotFoundError(args.mesh)

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_vertex_normals()

    rr.init("mesh_viewer", spawn=True)
    rr.log(
        args.mesh.stem,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
        ),
    )

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
