import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from pathlib import Path
import time
import yaml
from datetime import datetime

start = datetime.now()

# ── Load config ────────────────────────────────────────────────────────
with open("configs/bpa_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ── 1. Load ────────────────────────────────────────────────────────────
print("Loading...")
pcd = o3d.io.read_point_cloud(cfg["ply_path"])
pts    = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
print(f"Loaded {len(pts):,} points")
print(f"Has normals: {pcd.has_normals()}")

# ── 2. Resampling (Poisson-Disk equivalent) ────────────────────────────
voxel_size = cfg.get("voxel_size")
if voxel_size:
    print(f"Resampling (voxel size {voxel_size} m)...")
    pcd_bpa = pcd.voxel_down_sample(voxel_size)
    print(f"  {len(pts):,} → {len(pcd_bpa.points):,} points")
else:
    pcd_bpa = pcd

# ── 3. Normal estimation ───────────────────────────────────────────────
if cfg["estimate_normals"] or not pcd_bpa.has_normals():
    print("Estimating normals...")
    t0 = time.time()
    pcd_bpa.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg["normal_radius"],
            max_nn=cfg["normal_max_nn"],
        )
    )
    if cfg.get("orient_normals", True):
        pcd_bpa.orient_normals_consistent_tangent_plane(cfg.get("orient_normals_k", 16))
    print(f"Normals estimated in {time.time()-t0:.1f}s")
else:
    print("Using existing normals.")

# ── 4. Determine ball radii ────────────────────────────────────────────
radii = cfg.get("radii") or []
if not radii:
    print("Auto-computing ball radii from average nn-distance...")
    nn_dists = np.asarray(pcd_bpa.compute_nearest_neighbor_distance())
    avg_dist  = float(np.mean(nn_dists))
    factors   = cfg.get("radii_factors", [1.5, 2.5, 5.0])
    radii     = [avg_dist * f for f in factors]
    print(f"  avg nn-distance: {avg_dist:.4f} m")
print(f"  radii: {[f'{r:.4f}' for r in radii]} m")

# ── 5. Ball Pivoting ───────────────────────────────────────────────────
print("Running Ball Pivoting...")
t0 = time.time()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd_bpa,
    o3d.utility.DoubleVector(radii),
)
print(f"Done in {time.time()-t0:.1f}s")
print(f"Raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

# ── 6. Post-processing ─────────────────────────────────────────────────
if cfg.get("remove_duplicates", True):
    print("Removing duplicates...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    print(f"Clean mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

# ── 7. Color transfer ──────────────────────────────────────────────────
if colors.size > 0:
    print("Baking colors...")
    n_threads = cfg.get("n_threads", -1)
    verts = np.asarray(mesh.vertices)
    tree  = KDTree(pts)
    _, idx = tree.query(verts, workers=n_threads)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])

# ── 8. Save ────────────────────────────────────────────────────────────
Path(cfg["out_path"]).parent.mkdir(parents=True, exist_ok=True)
o3d.io.write_triangle_mesh(cfg["out_path"], mesh)
print(f"Saved to {cfg['out_path']}")

print(f"\nTotal time: {datetime.now() - start}")
