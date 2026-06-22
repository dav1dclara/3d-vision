"""
Mesh Quality Assessment Module

This module provides functionality to evaluate the quality of 3D meshes by comparing them
against a ground truth point cloud. It computes multiple quality metrics and provides
visualization and reporting capabilities.

Main functions:
    - evaluate_single_mesh: Evaluate a single mesh (for internal testing)
    - evaluate_multiple_meshes: Evaluate and compare multiple meshes
"""

import gc
import numpy as np
import trimesh
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Helper & I/O Functions (Internal)
# ============================================================================

_PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _parse_ply_header(path: str) -> dict:
    """Parse PLY header; return vertex count, format, data offset, and property list."""
    result = {
        "n_vertices": 0,
        "is_binary": False,
        "little_endian": True,
        "data_offset": 0,
        "properties": [],
    }
    in_vertex = False
    with open(path, "rb") as f:
        for raw_line in f:
            result["data_offset"] += len(raw_line)
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            elif line.startswith("format"):
                fmt = line.split()[1]
                result["is_binary"] = fmt.startswith("binary")
                result["little_endian"] = fmt != "binary_big_endian"
            elif line.startswith("element vertex"):
                result["n_vertices"] = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and in_vertex:
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(
                        "List properties in vertex element are not supported"
                    )
                dt = _PLY_DTYPE_MAP.get(parts[1])
                if dt is None:
                    raise ValueError(f"Unknown PLY property type: {parts[1]}")
                result["properties"].append((parts[2], dt))
    return result


def _count_ply_points(path: str) -> int:
    """Read only the PLY header to get the vertex count (no data loaded)."""
    return _parse_ply_header(path)["n_vertices"]


def _load_ply_pointcloud_sampled(
    path: str, max_points: int, seed: int = 42
) -> np.ndarray:
    """Memory-efficient random sampling of a large binary PLY point cloud via memmap.

    Only touches the file pages for the sampled indices (~max_points * stride bytes),
    avoiding the need to allocate the full point cloud in RAM.
    """
    header = _parse_ply_header(path)
    if not header["is_binary"]:
        raise ValueError("Efficient sampling requires a binary PLY file")

    n_vertices = header["n_vertices"]
    properties = header["properties"]
    prop_names = [name for name, _ in properties]
    for coord in ("x", "y", "z"):
        if coord not in prop_names:
            raise ValueError(f"PLY file missing '{coord}' vertex property")

    endian = "<" if header["little_endian"] else ">"
    vertex_dtype = np.dtype([(name, endian + dt) for name, dt in properties])

    # Map file to virtual address space — OS pages in only the accessed blocks
    data = np.memmap(
        path,
        dtype=vertex_dtype,
        mode="r",
        offset=header["data_offset"],
        shape=(n_vertices,),
    )

    if n_vertices <= max_points:
        result = np.column_stack(
            [
                np.asarray(data["x"], dtype=np.float64),
                np.asarray(data["y"], dtype=np.float64),
                np.asarray(data["z"], dtype=np.float64),
            ]
        )
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_vertices, size=max_points, replace=False))
        result = np.column_stack(
            [
                np.asarray(data["x"][idx], dtype=np.float64),
                np.asarray(data["y"][idx], dtype=np.float64),
                np.asarray(data["z"][idx], dtype=np.float64),
            ]
        )

    del data
    return result


def _load_ply_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a mesh from a PLY file."""
    if not mesh_path.lower().endswith(".ply"):
        raise ValueError(f"Expected PLY file, got: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path, process=False)
    except Exception as e:
        raise ValueError(f"Failed to load mesh: {e}")

    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    return mesh


def _load_ply_pointcloud(
    pointcloud_path: str, max_points: Optional[int] = None
) -> np.ndarray:
    """Load a point cloud from a PLY file.

    When max_points is set, uses memmap-based random sampling so only
    ~max_points records are paged into RAM instead of the full file.
    Falls back to trimesh if the binary-sampling path fails.
    """
    if not pointcloud_path.lower().endswith(".ply"):
        raise ValueError(f"Expected PLY file, got: {pointcloud_path}")

    if max_points is not None:
        try:
            points = _load_ply_pointcloud_sampled(pointcloud_path, max_points)
            if len(points) == 0:
                raise ValueError(f"Point cloud has no points: {pointcloud_path}")
            return points
        except Exception:
            pass  # fall through to trimesh

    try:
        mesh = trimesh.load(pointcloud_path, process=False)
        points = mesh.vertices
    except Exception as e:
        raise ValueError(f"Failed to load point cloud: {e}")

    if len(points) == 0:
        raise ValueError(f"Point cloud has no points: {pointcloud_path}")

    return points


def _spatial_voxel_sample(
    points: np.ndarray,
    max_samples: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample points uniformly from a voxel grid for representative spatial coverage."""
    if len(points) <= max_samples:
        return points

    if rng is None:
        rng = np.random.default_rng()

    bbox_min = np.min(points, axis=0)
    bbox_size = np.max(points, axis=0) - bbox_min

    voxel_size = np.cbrt(np.prod(bbox_size) / max_samples)
    voxel_indices = np.floor((points - bbox_min) / voxel_size).astype(np.int32)

    # Encode each voxel as a single int64 key — avoids np.unique on 2D array
    dims = voxel_indices.max(axis=0) + 1
    keys = (
        voxel_indices[:, 0] * np.int64(dims[1]) * dims[2]
        + voxel_indices[:, 1] * np.int64(dims[2])
        + voxel_indices[:, 2]
    )

    # Shuffle indices so rng.choice(point_indices) == picking any shuffled position
    order = rng.permutation(len(points))
    keys_shuffled = keys[order]

    # Keep the first occurrence of each voxel key after shuffling (= random pick per voxel)
    _, first = np.unique(keys_shuffled, return_index=True)
    sampled_indices = order[first]

    if len(sampled_indices) > max_samples:
        sampled_indices = rng.choice(sampled_indices, size=max_samples, replace=False)

    return points[sampled_indices]


def _o3d_point_to_mesh_distances(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> np.ndarray:
    """Compute point-to-mesh distances via Open3D RaycastingScene.

    Uses a C++ BVH with float32 internally — avoids the large per-face numpy
    allocations that trimesh.proximity.closest_point requires (face normals,
    triangle arrays), making this safe for meshes with tens of millions of faces.
    """
    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(
        np.asarray(mesh.vertices, dtype=np.float32)
    )
    o3d_mesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    distances = scene.compute_distance(query).numpy().astype(np.float64)
    del scene, o3d_mesh, query
    return distances


def _compute_point_to_mesh_distance(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    sample_size: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute closest distance from each point to mesh SURFACE (exact to triangles).

    Strategy: Sample points uniformly via voxel grid, compute exact surface distances.
    - Ensures distance measured to mesh SURFACE (triangles), not just vertices
    - Voxel-based sampling guarantees uniform spatial representation
    - Makes quality metrics independent of input point cloud density
    - Sampling keeps performance acceptable for large point clouds

    Args:
        mesh: Trimesh object
        points: Nx3 array of points
        sample_size: Maximum number of points to sample (default 50000)

    Returns:
        Array of distances (for sampled points) or all points if already small
    """
    # Sample using voxel grid for uniform spatial distribution
    points_sampled = _spatial_voxel_sample(points, max_samples=sample_size, rng=rng)
    return _o3d_point_to_mesh_distances(mesh, points_sampled)


# ============================================================================
# Metric Computation Functions (Internal)
# ============================================================================


def _hausdorff_distance(
    mesh: trimesh.Trimesh,
    ground_truth_points: np.ndarray,
    sample_size: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute Hausdorff distance between mesh and point cloud.

    Uses voxel-based sampling internally to ensure representative maximum.
    No double sampling - the sample is consistent throughout computation.
    """
    # Single pass with voxel sampling (no double sampling)
    distances = _compute_point_to_mesh_distance(
        mesh, ground_truth_points, sample_size=sample_size, rng=rng
    )
    return float(np.max(distances))


def _rmse_mae(
    mesh: trimesh.Trimesh,
    ground_truth_points: np.ndarray,
    sample_size: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Compute RMSE and MAE using voxel-sampled distances."""
    distances = _compute_point_to_mesh_distance(
        mesh, ground_truth_points, sample_size=sample_size, rng=rng
    )
    rmse = float(np.sqrt(np.mean(distances**2)))
    mae = float(np.mean(np.abs(distances)))
    return rmse, mae


def _surface_smoothness(
    mesh: trimesh.Trimesh, rng: Optional[np.random.Generator] = None
) -> float:
    """Compute surface smoothness based on face normal variation."""
    face_normals = mesh.face_normals
    num_faces = len(mesh.faces)

    if num_faces > 10000:
        if rng is None:
            rng = np.random.default_rng()
        sample_idx = rng.choice(num_faces, min(10000, num_faces // 10), replace=False)
        face_normals = face_normals[sample_idx]

    angle_diffs = []
    for i in range(min(1000, len(face_normals) - 1)):
        for j in range(i + 1, min(i + 50, len(face_normals))):
            cos_angle = np.clip(np.dot(face_normals[i], face_normals[j]), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_diffs.append(angle)

    return float(np.std(angle_diffs)) if angle_diffs else 0.0


def _mesh_quality_stats(
    mesh: trimesh.Trimesh,
    ground_truth_points: np.ndarray,
    sample_size: int = 50000,
    thresholds_m: List[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, any]:
    """
    Compute mesh quality metrics: aspect ratio and residuum distribution.

    Residuum: distance between mesh and ground truth.
    - min_residuum_c2m: min distance from cloud to mesh (unrecovered areas)
    - max_residuum_c2m: max distance from cloud to mesh (critical gaps)
    Residuum distribution: percentage of points in each residuum category.
    """
    if thresholds_m is None:
        thresholds_m = [0.01, 0.02, 0.10]  # Default: 1cm, 2cm, 10cm in meters

    vertices = mesh.vertices
    faces = mesh.faces

    if len(faces) > 10000:
        if rng is None:
            rng = np.random.default_rng()
        faces = faces[rng.choice(len(faces), 10000, replace=False)]

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v2, axis=1)
    e1 = np.linalg.norm(v2 - v0, axis=1)
    e2 = np.linalg.norm(v0 - v1, axis=1)
    del v0, v1, v2
    edge_max = np.maximum(np.maximum(e0, e1), e2)
    edge_min = np.minimum(np.minimum(e0, e1), e2)
    valid = edge_min > 1e-10
    aspect_ratios = (edge_max[valid] / edge_min[valid]).tolist()
    del e0, e1, e2, edge_max, edge_min, valid

    # Compute residuum using voxel sampling
    residuum_c2m = _compute_point_to_mesh_distance(
        mesh, ground_truth_points, sample_size=sample_size, rng=rng
    )  # Cloud to Mesh

    # Compute residuum distribution percentages
    t1, t2, t3 = thresholds_m[0], thresholds_m[1], thresholds_m[2]
    total = len(residuum_c2m)

    pct_good = (np.sum(residuum_c2m < t1) / total * 100) if total > 0 else 0.0
    pct_ok = (
        (np.sum((residuum_c2m >= t1) & (residuum_c2m < t2)) / total * 100)
        if total > 0
        else 0.0
    )
    pct_critical = (
        (np.sum((residuum_c2m >= t2) & (residuum_c2m < t3)) / total * 100)
        if total > 0
        else 0.0
    )
    pct_missing = (np.sum(residuum_c2m >= t3) / total * 100) if total > 0 else 0.0

    return {
        "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else 1.0,
        "min_residuum_c2m": float(np.min(residuum_c2m))
        if len(residuum_c2m) > 0
        else 0.0,
        "max_residuum_c2m": float(np.max(residuum_c2m))
        if len(residuum_c2m) > 0
        else 0.0,
        "pct_good": pct_good,
        "pct_ok": pct_ok,
        "pct_critical": pct_critical,
        "pct_missing": pct_missing,
    }


def _mesh_structure_stats(
    mesh: trimesh.Trimesh, rng: Optional[np.random.Generator] = None
) -> Dict[str, int]:
    """Compute mesh structure statistics."""
    vertices = mesh.vertices
    faces = mesh.faces

    vertex_count = len(vertices)
    face_count = len(faces)

    n_sample = min(len(faces), 50000)
    if len(faces) > n_sample:
        if rng is None:
            rng = np.random.default_rng()
        faces_check = faces[rng.choice(len(faces), n_sample, replace=False)]
    else:
        faces_check = faces

    v0 = vertices[faces_check[:, 0]]
    v1 = vertices[faces_check[:, 1]]
    v2 = vertices[faces_check[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    del v0, v1, v2
    degen_sampled = int(np.sum(areas < 1e-10))
    degen_count = (
        int(degen_sampled * len(faces) / n_sample)
        if len(faces) > n_sample
        else degen_sampled
    )

    return {
        "vertices": vertex_count,
        "faces": face_count,
        "degenerate_triangles": degen_count,
    }


def _watertightness_manifoldness(mesh: trimesh.Trimesh) -> Tuple[bool, bool]:
    """Check watertightness and manifoldness (fully vectorized, no Python loops)."""
    faces = mesh.faces  # (F, 3) int32
    # Build all three directed edges per face, then sort each pair → undirected edges
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.concatenate([e01, e12, e20], axis=0)  # (3F, 2)

    # Encode as single int64 so np.unique works on 1D (faster, less memory)
    n_verts = int(mesh.vertices.shape[0])
    edge_keys = all_edges[:, 0].astype(np.int64) * n_verts + all_edges[:, 1]
    del all_edges, e01, e12, e20

    _, counts = np.unique(edge_keys, return_counts=True)
    del edge_keys

    is_manifold = bool(np.all(counts == 2))
    is_watertight = is_manifold and bool(mesh.is_watertight)
    return is_watertight, is_manifold


def _f_score(
    mesh: trimesh.Trimesh,
    ground_truth_points: np.ndarray,
    threshold: Optional[float] = None,
    sample_size: int = 50000,
) -> float:
    """Compute F-score (bidirectional matching) with voxel sampling."""
    mesh_points = mesh.sample(min(sample_size, len(ground_truth_points)))
    gt_pts = _spatial_voxel_sample(ground_truth_points, max_samples=sample_size)

    if threshold is None:
        bbox_size = np.linalg.norm(np.max(gt_pts, axis=0) - np.min(gt_pts, axis=0))
        threshold = 0.01 * bbox_size

    mesh_tree = cKDTree(mesh_points)
    gt_tree = cKDTree(gt_pts)

    dist_m2g, _ = mesh_tree.query(gt_pts)
    correct_m2g = np.sum(dist_m2g <= threshold)

    dist_g2m, _ = gt_tree.query(mesh_points)
    correct_g2m = np.sum(dist_g2m <= threshold)

    prec = correct_m2g / len(gt_pts) if len(gt_pts) > 0 else 0.0
    rec = correct_g2m / len(mesh_points) if len(mesh_points) > 0 else 0.0

    if prec + rec == 0:
        return 0.0
    return float(2 * (prec * rec) / (prec + rec))


# ============================================================================
# Visualization Functions
# ============================================================================


def _create_distance_heatmap_mesh(
    points_sampled: np.ndarray,
    distances_sampled: np.ndarray,
    thresholds_m: List[float] = None,
) -> go.Figure:
    """Create interactive 3D heatmap visualization using pre-computed distances.

    Accepts already-sampled points and distances so the caller's BVH scene is
    reused and no second scene build is needed.
    """
    if thresholds_m is None:
        thresholds_m = [0.01, 0.02, 0.10]

    # Dynamically create color categories based on thresholds
    t1, t2, t3 = thresholds_m[0], thresholds_m[1], thresholds_m[2]
    colors_normalized = np.zeros_like(distances_sampled)

    # Category 1: 0 to t1 (Green)
    mask1 = distances_sampled < t1
    colors_normalized[mask1] = (distances_sampled[mask1] / t1) * 0.25

    # Category 2: t1 to t2 (Yellow)
    mask2 = (distances_sampled >= t1) & (distances_sampled < t2)
    colors_normalized[mask2] = (
        0.25 + ((distances_sampled[mask2] - t1) / (t2 - t1)) * 0.25
    )

    # Category 3: t2 to t3 (Red)
    mask3 = (distances_sampled >= t2) & (distances_sampled < t3)
    colors_normalized[mask3] = (
        0.50 + ((distances_sampled[mask3] - t2) / (t3 - t2)) * 0.25
    )

    # Category 4: >= t3 (Red/Black)
    mask4 = distances_sampled >= t3
    colors_normalized[mask4] = 1.0

    cat1_label = f"Good (0-{t1 * 100:.1f}cm)"
    cat2_label = f"OK ({t1 * 100:.1f}-{t2 * 100:.1f}cm)"
    cat3_label = f"Critical ({t2 * 100:.1f}-{t3 * 100:.1f}cm)"
    cat4_label = f"Missing (≥{t3 * 100:.1f}cm)"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_sampled[:, 0],
                y=points_sampled[:, 1],
                z=points_sampled[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=colors_normalized,
                    colorscale=[
                        [0.00, "#00AA00"],  # Green (category 1: 0-t1)
                        [0.24, "#00AA00"],  # Green end
                        [0.25, "#FFFF00"],  # Yellow (category 2: t1-t2)
                        [0.49, "#FFFF00"],  # Yellow end
                        [0.50, "#FF0000"],  # Red (category 3: t2-t3)
                        [0.74, "#FF0000"],  # Red end
                        [0.75, "#000000"],  # Black (category 4: >=t3)
                        [1.00, "#000000"],  # Black end
                    ],
                    cmin=0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(
                        title="Residuum (cm)",
                        thickness=15,
                        len=0.7,
                        x=1.02,
                        tickvals=[0.125, 0.375, 0.625, 0.875],
                        ticktext=[cat1_label, cat2_label, cat3_label, cat4_label],
                    ),
                    opacity=0.8,
                    line=dict(width=0),
                ),
                text=[f"Residuum: {d * 100:.1f} cm" for d in distances_sampled],
                hoverinfo="text+x+y+z",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Cloud-to-Mesh Residual Analysis",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14},
        },
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode="data",
            bgcolor="rgba(240, 240, 240, 0.9)",
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=250, t=50, b=0),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
    )

    return fig


# ============================================================================
# Public API Functions
# ============================================================================


def evaluate_mesh(
    mesh: trimesh.Trimesh,
    ground_truth_points: np.ndarray,
    sample_size: int = 20000,
    thresholds_cm: List[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    detailed_summary: bool = True,
    mesh_label: Optional[str] = None,
    pointcloud_label: Optional[str] = None,
    compute_structure: bool = True,
    compute_distance: bool = True,
    compute_residual_distribution: bool = True,
    compute_watertightness: bool = True,
    compute_f_score: bool = True,
    compute_visualization: bool = True,
    original_point_count: Optional[int] = None,
) -> Dict:
    """
    Evaluate a single mesh against ground truth with object-based API.
    OPTIMIZED: Samples points ONCE, reuses across all distance-based metrics.

    Args:
        mesh: trimesh.Trimesh object
        ground_truth_points: Nx3 array of ground truth points (meters)
        sample_size: Max points to sample (default 50000)
        thresholds_cm: Error thresholds in cm, e.g. [1.0, 2.0, 10.0]
        seed: Optional random seed for deterministic sampling
        verbose: If True, print computation progress
        detailed_summary: If True, print detailed metrics and residuum distribution at end
        mesh_label: Optional label for mesh source (e.g. filename)
        pointcloud_label: Optional label for point cloud source (e.g. filename)
        compute_structure: If True, compute mesh structure and aspect ratios
        compute_distance: If True, compute distance metrics
        compute_residual_distribution: If True, compute residual distribution
        compute_watertightness: If True, compute watertightness and manifoldness
        compute_f_score: If True, compute F-score
        compute_visualization: If True, create Plotly visualization

    Returns:
        Dict with all metrics plus residuum_distribution breakdown
    """
    if thresholds_cm is None:
        thresholds_cm = [1.0, 2.0, 10.0]

    thresholds_m = [t / 100.0 for t in thresholds_cm]
    pc_vertices = (
        original_point_count
        if original_point_count is not None
        else len(ground_truth_points)
    )
    pc_sampled = len(ground_truth_points)

    if verbose:
        pass

    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    distances_sampled = None
    if compute_distance or compute_residual_distribution or compute_visualization:
        if verbose:
            print("Sampling points...")
        points_sampled = _spatial_voxel_sample(
            ground_truth_points, max_samples=sample_size, rng=rng
        )
        distances_sampled = _o3d_point_to_mesh_distances(mesh, points_sampled)

    if compute_distance and distances_sampled is not None:
        if verbose:
            print("Computing distance metrics...")
        hausdorff = float(np.max(distances_sampled))
        rmse = float(np.sqrt(np.mean(distances_sampled**2)))
        mae = float(np.mean(np.abs(distances_sampled)))
        mean_residuum_c2m = float(np.mean(distances_sampled))
        median_residuum_c2m = float(np.median(distances_sampled))
        min_residuum_c2m = float(np.min(distances_sampled))
        max_residuum_c2m = float(np.max(distances_sampled))
    else:
        if verbose and compute_distance:
            print("Computing distance metrics... skipped")
        hausdorff = None
        rmse = None
        mae = None
        mean_residuum_c2m = None
        median_residuum_c2m = None
        min_residuum_c2m = None
        max_residuum_c2m = None

    if compute_structure:
        if verbose:
            print("Mesh structure...")
        vertices = mesh.vertices
        faces = mesh.faces

        sample_faces = faces
        if len(faces) > 10000:
            sample_faces = faces[rng.choice(len(faces), 10000, replace=False)]

        v0 = vertices[sample_faces[:, 0]]
        v1 = vertices[sample_faces[:, 1]]
        v2 = vertices[sample_faces[:, 2]]
        e0 = np.linalg.norm(v1 - v2, axis=1)
        e1 = np.linalg.norm(v2 - v0, axis=1)
        e2 = np.linalg.norm(v0 - v1, axis=1)
        del v0, v1, v2
        edge_max = np.maximum(np.maximum(e0, e1), e2)
        edge_min = np.minimum(np.minimum(e0, e1), e2)
        valid = edge_min > 1e-10
        aspect_ratios = (edge_max[valid] / edge_min[valid]).tolist()
        del e0, e1, e2, edge_max, edge_min, valid
    else:
        if verbose:
            print("Mesh structure... skipped")
        aspect_ratios = []

    if compute_residual_distribution and distances_sampled is not None:
        if verbose:
            print("Residual distribution...")
        t1, t2, t3 = thresholds_m[0], thresholds_m[1], thresholds_m[2]
        total = len(distances_sampled)
        pct_good = (np.sum(distances_sampled < t1) / total * 100) if total > 0 else 0.0
        pct_ok = (
            (np.sum((distances_sampled >= t1) & (distances_sampled < t2)) / total * 100)
            if total > 0
            else 0.0
        )
        pct_critical = (
            (np.sum((distances_sampled >= t2) & (distances_sampled < t3)) / total * 100)
            if total > 0
            else 0.0
        )
        pct_missing = (
            (np.sum(distances_sampled >= t3) / total * 100) if total > 0 else 0.0
        )
    else:
        if verbose and compute_residual_distribution:
            print("Residual distribution... skipped")
        pct_good = None
        pct_ok = None
        pct_critical = None
        pct_missing = None

    if compute_structure:
        structure = _mesh_structure_stats(mesh, rng=rng)
    else:
        structure = {
            "vertices": None,
            "faces": None,
            "degenerate_triangles": None,
        }

    if compute_watertightness:
        if verbose:
            print("Watertightness...")
        watertight, manifold = _watertightness_manifoldness(mesh)
    else:
        if verbose:
            print("Watertightness... skipped")
        watertight, manifold = None, None

    if compute_f_score:
        if verbose:
            print("F-score...")
        f_score = _f_score(mesh, ground_truth_points, sample_size=sample_size)
    else:
        if verbose:
            print("F-score... skipped")
        f_score = None

    if compute_visualization:
        if verbose:
            print("Visualization...")
        if distances_sampled is not None:
            plotly_fig = _create_distance_heatmap_mesh(
                points_sampled, distances_sampled, thresholds_m=thresholds_m
            )
        else:
            # distances not computed yet — build them now only for visualization
            pts = _spatial_voxel_sample(
                ground_truth_points, max_samples=sample_size, rng=rng
            )
            dists = _o3d_point_to_mesh_distances(mesh, pts)
            plotly_fig = _create_distance_heatmap_mesh(
                pts, dists, thresholds_m=thresholds_m
            )
            del pts, dists
    else:
        if verbose:
            print("Visualization... skipped")
        plotly_fig = None

    # Release large arrays — they are no longer needed for the summary or return value
    del distances_sampled, ground_truth_points
    gc.collect()

    # Detailed summary with all metrics and residuum distribution (separate from progress log)
    if detailed_summary:
        print("\nResults:")
        if mesh_label:
            print(f"Mesh:       {mesh_label}")
        if pointcloud_label:
            print(f"Pointcloud: {pointcloud_label}")
        if mesh_label or pointcloud_label:
            print()

        if compute_structure:
            print("Structure:")
            if pc_sampled < pc_vertices:
                print(
                    f"  Points (Cloud):              {pc_vertices} (subsampled to {pc_sampled})"
                )
            else:
                print(f"  Points (Cloud):              {pc_vertices}")
            print(f"  Vertices (Mesh):             {structure['vertices']}")
            print(f"  Faces (Mesh):                {structure['faces']}")
            print(f"  Degenerate Triangles:        {structure['degenerate_triangles']}")
            print(
                f"  Mean Aspect Ratio:           {float(np.mean(aspect_ratios)) if aspect_ratios else 1.0:.3f}"
            )

        if compute_distance:
            print("Distance:")
            print(f"  Hausdorff Distance:          {hausdorff:.3f} m")
            print(f"  RMSE:                        {rmse:.3f} m")
            print(f"  MAE:                         {mae:.3f} m")
            print(f"  Mean Residuum:               {mean_residuum_c2m:.3f} m")
            print(f"  Median Residuum:             {median_residuum_c2m:.3f} m")
            print(f"  Min Residuum:                {min_residuum_c2m:.3f} m")
            print(f"  Max Residuum:                {max_residuum_c2m:.3f} m")

        if compute_residual_distribution:
            print("Residual Distribution:")
            print(f"  Thresholds: {thresholds_cm} cm")
            for label, pct in [
                (f"Good <{thresholds_cm[0]}cm", pct_good),
                (f"OK {thresholds_cm[0]}-{thresholds_cm[1]}cm", pct_ok),
                (f"Critical {thresholds_cm[1]}-{thresholds_cm[2]}cm", pct_critical),
                (f"Missing ≥{thresholds_cm[2]}cm", pct_missing),
            ]:
                bar = "#" * int(pct / 2) if pct is not None else ""
                pct_text = f"{pct:6.2f}%" if pct is not None else "  n/a"
                print(f"  {label:30} {pct_text} {bar}")

        if compute_watertightness:
            print("Topology:")
            print(f"  Watertight):                 {watertight}")
            print(f"  Manifold:                    {manifold}")

        if compute_f_score:
            print("F-Score:")
            print(f"  F-Score:                     {f_score:.3f}")

    return {
        "hausdorff": hausdorff,
        "rmse": rmse,
        "mae": mae,
        "mean_residuum_c2m": mean_residuum_c2m,
        "median_residuum_c2m": median_residuum_c2m,
        "min_residuum_c2m": min_residuum_c2m,
        "max_residuum_c2m": max_residuum_c2m,
        "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else None,
        "pct_good": pct_good,
        "pct_ok": pct_ok,
        "pct_critical": pct_critical,
        "pct_missing": pct_missing,
        "vertices_mesh": structure["vertices"],
        "vertices_pc": pc_vertices,
        "faces": structure["faces"],
        "degenerate_triangles": structure["degenerate_triangles"],
        "watertight": watertight,
        "manifold": manifold,
        "f_score": f_score,
        "mesh_with_colors": plotly_fig,
    }
