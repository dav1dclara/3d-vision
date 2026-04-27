import argparse
import os
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import laspy
import numpy as np
import open3d as o3d
import yaml
from vdbfusion import VDBVolume


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_mesh(path: str, vertices: np.ndarray, triangles: np.ndarray) -> None:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)


def read_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    calib: Dict[str, np.ndarray] = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split(" ")
            if not tokens or tokens[0] == "calib_time:":
                continue
            key = tokens[0].rstrip(":")
            values = np.array([float(v) for v in tokens[1:] if v], dtype=np.float64)
            calib[key] = values
    return calib


def read_kitti_poses(poses_path: str, tr_velo_to_cam: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(poses_path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    n = raw.shape[0]
    poses = np.concatenate(
        [raw, np.zeros((n, 3), dtype=np.float64), np.ones((n, 1), dtype=np.float64)],
        axis=1,
    ).reshape(n, 4, 4)

    tr = np.eye(4, dtype=np.float64)
    tr[:3, :4] = tr_velo_to_cam.reshape(3, 4)
    tr_inv = np.linalg.inv(tr)
    return tr_inv @ poses @ tr


def iter_kitti_scans(
    kitti_root_dir: str,
    sequence: int,
    min_range: float,
    max_range: float,
    max_scans: int,
    scan_stride: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    seq = f"{int(sequence):02d}"
    seq_dir = os.path.join(kitti_root_dir, "sequences", seq)
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    calib_path = os.path.join(seq_dir, "calib.txt")
    poses_path = os.path.join(kitti_root_dir, "poses", f"{seq}.txt")

    scan_files = sorted(
        os.path.join(velodyne_dir, f)
        for f in os.listdir(velodyne_dir)
        if f.endswith(".bin")
    )

    if not scan_files:
        raise FileNotFoundError(f"No KITTI scans found at {velodyne_dir}")

    calib = read_kitti_calib(calib_path)
    if "Tr" not in calib:
        raise KeyError(f"Expected key 'Tr' in KITTI calib file: {calib_path}")
    poses = read_kitti_poses(poses_path, calib["Tr"])

    selected_indices = list(range(0, min(len(scan_files), len(poses)), max(scan_stride, 1)))
    if max_scans > 0:
        selected_indices = selected_indices[:max_scans]

    for idx in selected_indices:
        scan = np.fromfile(scan_files[idx], dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        ranges = np.linalg.norm(scan, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        scan = scan[mask]
        if scan.size == 0:
            continue

        pose = poses[idx]
        scan_h = np.concatenate([scan, np.ones((scan.shape[0], 1), dtype=np.float64)], axis=1)
        points_world = (pose @ scan_h.T).T[:, :3]
        origin = pose[:3, 3].astype(np.float64)
        yield points_world, origin


def iter_las_chunks(
    las_path: str,
    chunk_size: int,
    origin_offset: List[float],
    fixed_origin: Optional[List[float]],
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    las = laspy.read(las_path)
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    if points.shape[0] == 0:
        raise ValueError(f"No points found in LAS file: {las_path}")

    n = points.shape[0]
    chunk_size = max(chunk_size, 1)

    if fixed_origin is not None:
        origin_const = np.array(fixed_origin, dtype=np.float64)
    else:
        origin_const = points.mean(axis=0) + offset

    origin_const = np.ascontiguousarray(origin_const.reshape(3,), dtype=np.float64)

    for start in range(0, n, chunk_size):
        chunk = points[start:start + chunk_size]
        if chunk.size == 0:
            continue
        chunk = np.ascontiguousarray(chunk, dtype=np.float64)
        yield chunk, origin_const


def integrate_dataset(scans: Iterable[Tuple[np.ndarray, np.ndarray]], vdb_volume: VDBVolume) -> int:
    n_integrated = 0
    for i, (points, pose) in enumerate(scans):
        if points.size == 0:
            continue
        
        points = np.asarray(points, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)
        
        # Validate
        if points.shape[1] != 3 or np.any(~np.isfinite(points)):
            print(f"  Chunk {i}: skipped (bad points)")
            continue
        if np.any(~np.isfinite(pose)):
            print(f"  Chunk {i}: skipped (bad pose)")
            continue
        
        try:
            print(f"  Chunk {i}: {len(points):,} points, pose {pose.shape}")
            vdb_volume.integrate(points=points, extrinsic=pose)
            n_integrated += 1
        except Exception as e:
            print(f"  Chunk {i}: {type(e).__name__}: {e}")
            continue
    
    return n_integrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental VDBFusion reconstruction script")
    parser.add_argument(
        "--config",
        default="configs/vdbfusion_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    dataset_type = cfg["input"]["dataset_type"].strip().lower()

    voxel_size = float(cfg["fusion"]["voxel_size"])
    sdf_trunc = float(cfg["fusion"]["sdf_trunc"])
    space_carving = bool(cfg["fusion"]["space_carving"])
    fill_holes = bool(cfg["fusion"]["fill_holes"])
    min_weight = float(cfg["fusion"]["min_weight"])

    output_dir = cfg["output"]["output_dir"]
    map_name = cfg["output"]["map_name"]
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing VDBVolume...")
    tsdf_volume = VDBVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc, space_carving=space_carving)

    if dataset_type == "kitti":
        print("Using KITTI odometry data loader...")
        scans = iter_kitti_scans(
            kitti_root_dir=cfg["input"]["kitti_root_dir"],
            sequence=int(cfg["input"]["sequence"]),
            min_range=float(cfg["preprocess"]["min_range"]),
            max_range=float(cfg["preprocess"]["max_range"]),
            max_scans=int(cfg["preprocess"]["max_scans"]),
            scan_stride=int(cfg["preprocess"]["scan_stride"]),
        )
    elif dataset_type == "las":
        print("Using LAS loader (chunked pseudo-scans)...")
        scans = iter_las_chunks(
            las_path=cfg["input"]["las_path"],
            chunk_size=50000,  # Smaller for safety; override config
            origin_offset=list(cfg["static_mode"]["origin_offset"]),
            fixed_origin=cfg["static_mode"].get("fixed_origin", None),
        )
    else:
        raise ValueError("Unsupported input.dataset_type. Use 'kitti' or 'las'.")

    t0 = time.time()
    n_scans = integrate_dataset(scans, tsdf_volume)
    dt = time.time() - t0
    if n_scans == 0:
        raise RuntimeError("No valid scans were integrated.")
    print(f"Integrated {n_scans} scans in {dt:.2f}s ({n_scans / max(dt, 1e-6):.2f} scans/s)")

    print("Extracting mesh...")
    vertices, triangles = tsdf_volume.extract_triangle_mesh(fill_holes=fill_holes, min_weight=min_weight)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError("VDBFusion produced an empty mesh. Try reducing min_weight or increasing scans.")

    ply_path = os.path.join(output_dir, f"{map_name}.ply")
    vdb_path = os.path.join(output_dir, f"{map_name}.vdb")
    write_mesh(ply_path, vertices, triangles)
    tsdf_volume.extract_vdb_grids(vdb_path)

    print(f"Saved mesh: {ply_path}")
    print(f"Saved VDB grids: {vdb_path}")
    print(f"Mesh stats: {len(vertices):,} vertices, {len(triangles):,} triangles")


if __name__ == "__main__":
    main()