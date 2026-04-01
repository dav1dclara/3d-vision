import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import time
import yaml

# ── Load config ───────────────────────────────────────────────────────
with open("configs/poisson_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ── 1. Load ──────────────────────────────────────────────────────────
print("Loading...")
pcd = o3d.io.read_point_cloud(cfg["ply_path"])
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
print(f"Loaded {len(pts):,} points")
print(f"Has normals: {pcd.has_normals()}")

# ── 2. Poisson reconstruction ─────────────────────────────────────────
print("Running Poisson...")
t0 = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=cfg["depth"],
    width=cfg["width"],
    scale=cfg["scale"],
    linear_fit=cfg["linear_fit"],
    n_threads=cfg["n_threads"],
)
print(f"Done in {time.time()-t0:.1f}s")
print(f"Raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

# ── 3. Trim low-density vertices ──────────────────────────────────────
print("Trimming...")
densities = np.asarray(densities)
threshold = np.percentile(densities, cfg["density_percentile"])
verts_to_remove = densities < threshold
mesh.remove_vertices_by_mask(verts_to_remove)
print(f"After trim: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

# ── 4. Color transfer ─────────────────────────────────────────────────
print("Baking colors...")
verts = np.asarray(mesh.vertices)
tree = KDTree(pts)
_, idx = tree.query(verts, workers=cfg["n_threads"])
mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])

# ── 5. Save ───────────────────────────────────────────────────────────
o3d.io.write_triangle_mesh(cfg["out_path"], mesh)
print(f"Saved to {cfg['out_path']}")