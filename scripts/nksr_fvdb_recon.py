import numpy as np
import torch
import laspy
import open3d as o3d
import fvdb
import nksr

# ── 1. Load point cloud ───────────────────────────────────────────────────
las = laspy.read("/work/scratch/oscipal/2026-03-09_16.19.44/pointcloud.las")
xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

centroid = xyz.mean(axis=0)
xyz -= centroid

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
if not pcd.has_normals():
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

xyz_t    = torch.from_numpy(xyz).float().cuda()
normal_t = torch.from_numpy(np.asarray(pcd.normals)).float().cuda()

# ── 2. Build sparse fVDB grid ─────────────────────────────────────────────
VOXEL_SIZE   = 0.1
CHUNK_VOXELS = 64
OVERLAP      = 8

grid = fvdb.GridBatch.from_points(
    fvdb.JaggedTensor([xyz_t]),
    voxel_sizes=[VOXEL_SIZE] * 3,
    origins=[0.0, 0.0, 0.0],
)
print(f"Total voxels: {grid.total_voxels:,}")

bbox    = grid.bbox_at(0).cpu().numpy()
ijk_min = bbox[0].astype(int)
ijk_max = bbox[1].astype(int)
print(f"Grid bbox: {ijk_min} → {ijk_max}")
print(f"Scene size in voxels: {ijk_max - ijk_min}")
print(f"Approx chunks: {int(np.prod(np.ceil((ijk_max - ijk_min) / CHUNK_VOXELS)))}")

# ── 3. Chunk + NKSR ───────────────────────────────────────────────────────
reconstructor = nksr.Reconstructor(torch.device("cuda:0"))

all_vertices = []
all_faces    = []
face_offset  = 0

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

            if chunk_pts.shape[0] < 50:
                continue

            print(f"  Chunk [{ix},{iy},{iz}]: {chunk_pts.shape[0]:,} pts")

            if chunk_pts.shape[0] > 300_000:
                idx = torch.randperm(chunk_pts.shape[0], device="cuda")[:300_000]
                chunk_pts = chunk_pts[idx]
                chunk_nrm = chunk_nrm[idx]

            try:
                with torch.no_grad():
                    field = reconstructor.reconstruct(
                        chunk_pts, chunk_nrm, detail_level=1.0
                    )
                    mesh = field.extract_dual_mesh(mise_iter=1)

                verts = mesh.v.cpu().numpy()
                faces = mesh.f.cpu().numpy()

                if len(verts) == 0 or len(faces) == 0:
                    continue

                all_vertices.append(verts)
                all_faces.append(faces + face_offset)
                face_offset += len(verts)

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM on chunk [{ix},{iy},{iz}], skipping")
            finally:
                torch.cuda.empty_cache()

# ── 4. Merge and save ─────────────────────────────────────────────────────
if not all_vertices:
    print("No chunks produced geometry — check point count / normals.")
else:
    merged_v = np.concatenate(all_vertices, axis=0)
    merged_f = np.concatenate(all_faces,    axis=0)
    merged_v += centroid

    out = o3d.geometry.TriangleMesh()
    out.vertices  = o3d.utility.Vector3dVector(merged_v)
    out.triangles = o3d.utility.Vector3iVector(merged_f)
    out.compute_vertex_normals()
    out.merge_close_vertices(eps=VOXEL_SIZE * 0.5)
    out.remove_duplicated_vertices()
    out.remove_degenerate_triangles()
    out.remove_non_manifold_edges()

o3d.io.write_triangle_mesh("outputs/reconstruction.ply", out)
print(f"Saved: {len(merged_v):,} vertices, {len(merged_f):,} faces")