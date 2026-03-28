import os
import numpy as np
import torch
import laspy
import open3d as o3d
import fvdb
import nksr

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/chunks", exist_ok=True)

# ── 1. Load point cloud ───────────────────────────────────────────────────
las = laspy.read("/work/scratch/oscipal/2026-03-09_16.19.44/pointcloud.las")
xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
    colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
    colors /= 65535.0
    has_color = True
    print("Color data found.")
else:
    has_color = False
    print("No color data found.")

centroid = xyz.mean(axis=0)
np.save("outputs/centroid.npy", centroid)
xyz -= centroid

print("Estimating normals...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=15)
print("Normals estimated.")

xyz_t    = torch.from_numpy(xyz).float().cuda()
normal_t = torch.from_numpy(np.asarray(pcd.normals)).float().cuda()
color_t  = torch.from_numpy(colors).float().cuda() if has_color else None

# ── 2. Build sparse fVDB grid ─────────────────────────────────────────────
VOXEL_SIZE   = 0.1
CHUNK_VOXELS = 80
OVERLAP      = 32

grid = fvdb.GridBatch.from_points(
    fvdb.JaggedTensor([xyz_t]),
    voxel_sizes=[VOXEL_SIZE] * 3,
    origins=[0.0, 0.0, 0.0],
)
print(f"Total voxels: {grid.total_voxels:,}")

bbox    = grid.bbox_at(0).cpu().numpy()
ijk_min = bbox[0].astype(int)
ijk_max = bbox[1].astype(int)
print(f"Approx chunks: {int(np.prod(np.ceil((ijk_max - ijk_min) / CHUNK_VOXELS)))}")

# ── 3. Chunk + NKSR — save each chunk to disk immediately ─────────────────
reconstructor = nksr.Reconstructor(torch.device("cuda:0"))
chunk_files = []

for ix in range(int(ijk_min[0]), int(ijk_max[0]), CHUNK_VOXELS):
    for iy in range(int(ijk_min[1]), int(ijk_max[1]), CHUNK_VOXELS):
        for iz in range(int(ijk_min[2]), int(ijk_max[2]), CHUNK_VOXELS):

            cmin = [ix - OVERLAP, iy - OVERLAP, iz - OVERLAP]
            cmax = [ix + CHUNK_VOXELS + OVERLAP,
                    iy + CHUNK_VOXELS + OVERLAP,
                    iz + CHUNK_VOXELS + OVERLAP]

            chunk_grid = grid.clipped_grid(ijk_min=cmin, ijk_max=cmax)
            if chunk_grid.total_voxels < 50:
                continue

            wmin = torch.tensor(cmin, device="cuda").float() * VOXEL_SIZE
            wmax = torch.tensor(cmax, device="cuda").float() * VOXEL_SIZE
            mask = ((xyz_t >= wmin) & (xyz_t <= wmax)).all(dim=1)

            chunk_pts = xyz_t[mask]
            chunk_nrm = normal_t[mask]
            chunk_clr = color_t[mask] if has_color else None

            if chunk_pts.shape[0] < 2000:
                continue

            print(f"  Chunk [{ix},{iy},{iz}]: {chunk_pts.shape[0]:,} pts")

            if chunk_pts.shape[0] > 300_000:
                idx = torch.randperm(chunk_pts.shape[0], device="cuda")[:300_000]
                chunk_pts = chunk_pts[idx]
                chunk_nrm = chunk_nrm[idx]
                if chunk_clr is not None:
                    chunk_clr = chunk_clr[idx]

            try:
                with torch.no_grad():
                    field = reconstructor.reconstruct(
                        chunk_pts, chunk_nrm, detail_level=1.0
                    )
                    if has_color and chunk_clr is not None:
                        field.set_texture_field(
                            nksr.fields.PCNNField(chunk_pts, chunk_clr)
                        )
                    mesh = field.extract_dual_mesh(mise_iter=1)

                verts = mesh.v.cpu().numpy() + centroid
                faces = mesh.f.cpu().numpy()

                if len(verts) == 0 or len(faces) == 0:
                    continue

                chunk_mesh = o3d.geometry.TriangleMesh()
                chunk_mesh.vertices  = o3d.utility.Vector3dVector(verts)
                chunk_mesh.triangles = o3d.utility.Vector3iVector(faces)
                if has_color and hasattr(mesh, 'c') and mesh.c is not None:
                    vc = np.clip(mesh.c.cpu().numpy(), 0.0, 1.0)
                    chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

                chunk_path = f"outputs/chunks/chunk_{ix}_{iy}_{iz}.ply"
                o3d.io.write_triangle_mesh(chunk_path, chunk_mesh)
                chunk_files.append(chunk_path)
                print(f"    Saved {len(verts):,} verts → {chunk_path}")

            except (torch.cuda.OutOfMemoryError, AttributeError, RuntimeError) as e:
                print(f"    Failed [{ix},{iy},{iz}]: {e}, skipping")
            finally:
                torch.cuda.empty_cache()

# ── 4. Merge chunk PLYs one at a time to avoid RAM explosion ──────────────
print(f"\nMerging {len(chunk_files)} chunk files...")
merged = o3d.geometry.TriangleMesh()
for i, path in enumerate(chunk_files):
    print(f"  Merging {i+1}/{len(chunk_files)}: {path}")
    chunk = o3d.io.read_triangle_mesh(path)
    merged += chunk
    del chunk

print("Cleaning up merged mesh...")
merged.merge_close_vertices(eps=VOXEL_SIZE * 1.0)
merged.remove_duplicated_vertices()
merged.remove_degenerate_triangles()
merged.remove_non_manifold_edges()
merged.compute_vertex_normals()

out_path = "outputs/reconstruction.ply"
o3d.io.write_triangle_mesh(out_path, merged)
print(f"Saved: {len(merged.vertices):,} vertices, {len(merged.triangles):,} faces → {out_path}")