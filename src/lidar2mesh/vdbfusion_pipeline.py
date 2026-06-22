"""Core VDBFusion reconstruction logic.

Pure, reusable helpers for the VDBFusion TSDF pipeline. Execution/config
orchestration lives in scripts/vdbfusion_reconstruction.py.
"""

import os
import time

import laspy
import numpy as np
import open3d as o3d
from vdbfusion import VDBVolume
from plyfile import PlyData, PlyElement


def build_points(las_path: str, downsample_voxel: float) -> np.ndarray:
    print("Loading LAS...")
    las = laspy.read(las_path)
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    # Strict cleanup for pybind/C++ safety.
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    points = np.ascontiguousarray(points, dtype=np.float64)

    if points.shape[0] == 0:
        raise RuntimeError("No valid points after cleanup.")

    print(f"Loaded: {len(points):,} points")

    if downsample_voxel > 0.0:
        print(f"Downsampling @ voxel={downsample_voxel}...")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
        points = np.asarray(pcd.points, dtype=np.float64)
        points = np.ascontiguousarray(points, dtype=np.float64)
        print(f"After downsample: {len(points):,} points")

    return points


def reconstruct_once(
    points: np.ndarray,
    fixed_origin: np.ndarray,
    voxel_size: float,
    sdf_trunc: float,
    space_carving: bool,
    fill_holes: bool,
    min_weight: float,
    out_dir: str,
    map_name: str,
    downsample_voxel: float,
    sweep_output_dir: str,
    sweep_enabled: bool,
) -> None:
    print(
        "Initializing VDBVolume "
        f"(voxel_size={voxel_size}, sdf_trunc={sdf_trunc}, min_weight={min_weight})..."
    )
    vdb = VDBVolume(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        space_carving=space_carving,
    )

    print(
        f"Integrating with origin={fixed_origin} and downsample_voxel={downsample_voxel} ..."
    )
    t0 = time.time()
    vdb.integrate(
        points=points, extrinsic=np.ascontiguousarray(fixed_origin, dtype=np.float64)
    )
    print(f"Integrate done in {time.time() - t0:.2f}s")

    os.makedirs(out_dir, exist_ok=True)
    vdb_path = os.path.join(
        out_dir,
        (
            f"{map_name}"
            f"_ds{str(downsample_voxel).replace('.', 'p')}"
            f"_vs{str(voxel_size).replace('.', 'p')}"
            f"_st{str(sdf_trunc).replace('.', 'p')}"
            f"_mw{str(min_weight).replace('.', 'p')}.vdb"
        ),
    )
    vdb.extract_vdb_grids(vdb_path)

    extract_dir = out_dir
    if sweep_enabled:
        extract_dir = os.path.join(out_dir, sweep_output_dir)
        os.makedirs(extract_dir, exist_ok=True)

    print("Extracting mesh...")
    verts, tris = vdb.extract_triangle_mesh(
        fill_holes=fill_holes, min_weight=min_weight
    )
    if len(verts) == 0 or len(tris) == 0:
        print("  empty mesh, skipping")
        print(f"Saved vdb:  {vdb_path}")
        return

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()

    ds_suffix = f"ds{str(downsample_voxel).replace('.', 'p')}"
    vs_suffix = f"vs{str(voxel_size).replace('.', 'p')}"
    st_suffix = f"st{str(sdf_trunc).replace('.', 'p')}"
    mw_suffix = f"mw{str(min_weight).replace('.', 'p')}"
    suffix = f"_{ds_suffix}_{vs_suffix}_{st_suffix}_{mw_suffix}"

    ply_path = os.path.join(extract_dir, f"{map_name}{suffix}.ply")
    vdb_sweep_path = os.path.join(extract_dir, f"{map_name}{suffix}.vdb")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    # write_ply_uint(ply_path, verts, tris)
    vdb.extract_vdb_grids(vdb_sweep_path)

    print(f"  saved mesh: {ply_path}")
    print(f"  saved vdb:  {vdb_sweep_path}")
    print(f"  mesh stats: {len(verts):,} verts, {len(tris):,} tris")
    print(f"Saved vdb:  {vdb_path}")


def write_ply_uint(ply_path: str, verts: np.ndarray, tris: np.ndarray) -> None:
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.uint32)

    v = np.empty(len(verts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]

    f = np.empty(len(tris), dtype=[("vertex_indices", "u4", (3,))])
    f["vertex_indices"] = tris

    face_el = PlyElement.describe(
        f,
        "face",
        len_types={"vertex_indices": "uint"},
        val_types={"vertex_indices": "uint"},
    )
    vert_el = PlyElement.describe(v, "vertex")
    PlyData([vert_el, face_el], text=False).write(ply_path)
