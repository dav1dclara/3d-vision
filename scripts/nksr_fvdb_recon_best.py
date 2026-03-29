import os
import shutil
import numpy as np
import torch
import laspy
import open3d as o3d
import fvdb
import nksr
from collections import defaultdict
from scipy.spatial import cKDTree

os.makedirs("outputs", exist_ok=True)
if os.path.exists("outputs/chunks"):
    shutil.rmtree("outputs/chunks")
os.makedirs("outputs/chunks", exist_ok=True)

# ── 1. Load point cloud ───────────────────────────────────────────────────
print("Loading point cloud...")
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
xyz -= centroid
print(f"Loaded {len(xyz):,} points.")

# ── 2. High accuracy normal estimation on full cloud ──────────────────────
print("Estimating normals (full cloud, high accuracy)...")
pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(xyz)
pcd_full.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
)
pcd_full.orient_normals_consistent_tangent_plane(k=20)
normals = np.asarray(pcd_full.normals).astype(np.float32)
print("Normals estimated.")

# ── 3. Assign points to fine chunks ───────────────────────────────────────
VOXEL_SIZE = 0.1
FINE_SIZE  = 0.5

print("Assigning points to fine chunks...")
chunk_indices = np.floor(xyz / FINE_SIZE).astype(np.int32)

point_chunks = defaultdict(list)
for i, key in enumerate(map(tuple, chunk_indices)):
    point_chunks[key].append(i)
print(f"Unique fine chunks: {len(point_chunks)}")

# ── 4. Fit plane to each fine chunk ───────────────────────────────────────
def fit_plane(pts):
    centered = pts - pts.mean(axis=0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    normal   = Vt[-1]
    d        = normal @ pts.mean(axis=0)
    residual = s[-1] / (s[0] + 1e-8)
    return normal, d, residual

def normals_similar(n1, n2, angle_thresh=15.0):
    cos_angle = abs(np.dot(n1, n2))
    return cos_angle > np.cos(np.radians(angle_thresh))

def planes_coplanar(n1, d1, pts2, dist_thresh=0.1):
    distances = np.abs(pts2 @ n1 - d1)
    return distances.mean() < dist_thresh

print("Computing per-chunk planarity...")
chunk_data = {}
for key, indices in point_chunks.items():
    if len(indices) < 20:
        continue
    indices_np = np.array(indices)
    pts        = xyz[indices_np]
    normal, d, residual = fit_plane(pts)
    is_planar  = residual < 0.05
    chunk_data[key] = {
        'indices':   indices_np,
        'normal':    normal,
        'd':         d,
        'residual':  residual,
        'is_planar': is_planar,
        'n_pts':     len(indices),
    }

n_planar  = sum(1 for c in chunk_data.values() if c['is_planar'])
n_complex = sum(1 for c in chunk_data.values() if not c['is_planar'])
print(f"Planar chunks: {n_planar}, Complex chunks: {n_complex}")

# ── 5. Region growing ─────────────────────────────────────────────────────
def get_neighbours(key):
    ix, iy, iz = key
    return [
        (ix+1, iy, iz), (ix-1, iy, iz),
        (ix, iy+1, iz), (ix, iy-1, iz),
        (ix, iy, iz+1), (ix, iy, iz-1),
    ]

print("Running region growing...")
visited = set()
regions = []

for start_key in chunk_data:
    if start_key in visited:
        continue

    region = []
    queue  = [start_key]

    while queue:
        key = queue.pop()
        if key in visited or key not in chunk_data:
            continue

        visited.add(key)
        region.append(key)
        chunk = chunk_data[key]

        for nb_key in get_neighbours(key):
            if nb_key in visited or nb_key not in chunk_data:
                continue
            nb = chunk_data[nb_key]

            if chunk['is_planar'] and nb['is_planar']:
                if (normals_similar(chunk['normal'], nb['normal']) and
                        planes_coplanar(chunk['normal'], chunk['d'],
                                        xyz[nb['indices']])):
                    queue.append(nb_key)

            elif not chunk['is_planar'] and not nb['is_planar']:
                queue.append(nb_key)

    regions.append(region)

print(f"Total regions: {len(regions)}")
sizes = sorted([len(r) for r in regions], reverse=True)
print(f"Top 10 region sizes (fine chunks): {sizes[:10]}")

# ── 6. Reconstruction setup ───────────────────────────────────────────────
reconstructor    = nksr.Reconstructor(torch.device("cuda:0"))
MIN_PTS          = 500
OVERLAP_DEFAULT  = 0.1
OVERLAP_BOUNDARY = 0.3
CONTEXT_VOX      = 0.15
chunk_files      = []
chunk_counter    = [0]

# No MAX_PTS cap — use full density always
# Only subsample if OOM

def save_mesh(verts, faces, mesh, core_min, core_max, label):
    trim_min = core_min + centroid
    trim_max = core_max + centroid
    in_core  = np.all((verts >= trim_min) & (verts <= trim_max), axis=1)
    face_mask  = (in_core[faces[:, 0]] &
                  in_core[faces[:, 1]] &
                  in_core[faces[:, 2]])
    faces_core = faces[face_mask]

    if len(faces_core) == 0:
        return None

    used        = np.unique(faces_core)
    remap       = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    verts_core  = verts[used]
    faces_core  = remap[faces_core]

    chunk_mesh = o3d.geometry.TriangleMesh()
    chunk_mesh.vertices  = o3d.utility.Vector3dVector(verts_core)
    chunk_mesh.triangles = o3d.utility.Vector3iVector(faces_core)
    if has_color and hasattr(mesh, 'c') and mesh.c is not None:
        vc = np.clip(mesh.c.cpu().numpy(), 0.0, 1.0)[used]
        chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    chunk_path = f"outputs/chunks/{label}_{chunk_counter[0]:05d}.ply"
    chunk_counter[0] += 1
    o3d.io.write_triangle_mesh(chunk_path, chunk_mesh)
    return chunk_path

def run_nksr(pts_np, nrm_np, clr_np, detail, mise, vox_size):
    chunk_pts = torch.from_numpy(pts_np).float().cuda()
    chunk_nrm = torch.from_numpy(nrm_np).float().cuda()
    chunk_clr = (torch.from_numpy(clr_np).float().cuda()
                 if clr_np is not None else None)
    try:
        with torch.no_grad():
            field = reconstructor.reconstruct(
                chunk_pts, chunk_nrm,
                detail_level=detail,
                voxel_size=vox_size,
            )
            if has_color and chunk_clr is not None:
                field.set_texture_field(
                    nksr.fields.PCNNField(chunk_pts, chunk_clr)
                )
            mesh = field.extract_dual_mesh(mise_iter=mise)
        verts = mesh.v.cpu().numpy() + centroid
        faces = mesh.f.cpu().numpy()
        if len(verts) == 0 or len(faces) == 0:
            return None
        return verts, faces, mesh
    except (torch.cuda.OutOfMemoryError, AttributeError, RuntimeError) as e:
        print(f"      NKSR failed: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

def smart_subsample(pts_np, nrm_np, clr_np, max_pts, vox_size):
    if len(pts_np) <= max_pts:
        return pts_np, nrm_np, clr_np

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts_np)
    pcd.normals = o3d.utility.Vector3dVector(nrm_np)
    if clr_np is not None:
        pcd.colors = o3d.utility.Vector3dVector(clr_np)

    vs = vox_size
    while True:
        down = pcd.voxel_down_sample(voxel_size=vs)
        if len(down.points) <= max_pts:
            break
        vs *= 1.5

    return (np.asarray(down.points).astype(np.float32),
            np.asarray(down.normals).astype(np.float32),
            np.asarray(down.colors).astype(np.float32)
            if clr_np is not None else None)

# ── 7. Planar region reconstruction ──────────────────────────────────────
def reconstruct_planar(region_pts, region_indices,
                       detail, mise, vox_size,
                       overlap_m, label):
    core_min = region_pts.min(axis=0)
    core_max = region_pts.max(axis=0)

    rmin = core_min - overlap_m
    rmax = core_max + overlap_m
    mask = np.all((xyz >= rmin) & (xyz <= rmax), axis=1)

    pts_exp = xyz[mask]
    nrm_exp = normals[mask]
    clr_exp = colors[mask] if has_color else None

    # Planar regions get smart subsampled — full density not needed
    pts_exp, nrm_exp, clr_exp = smart_subsample(
        pts_exp, nrm_exp, clr_exp, 300_000, vox_size
    )

    result = run_nksr(pts_exp, nrm_exp, clr_exp, detail, mise, vox_size)
    if result is None:
        return None
    verts, faces, mesh = result
    return save_mesh(verts, faces, mesh, core_min, core_max, label)

# ── 8. Complex region reconstruction with planar context ─────────────────
def reconstruct_complex(region_pts, region_indices,
                        region_keys, label):
    core_min = region_pts.min(axis=0)
    core_max = region_pts.max(axis=0)

    # Full density complex points — no cap
    pts_list = [region_pts]
    nrm_list = [normals[region_indices]]
    clr_list = [colors[region_indices]] if has_color else []

    # Downsampled planar context — 2 chunks deep for better edge quality
    region_key_set    = set(region_keys)
    planar_nb_indices = set()

    # First ring of planar neighbours
    first_ring = set()
    for key in region_keys:
        for nb_key in get_neighbours(key):
            if nb_key not in chunk_data:
                continue
            if (chunk_data[nb_key]['is_planar'] and
                    nb_key not in region_key_set):
                first_ring.add(nb_key)
                planar_nb_indices.update(
                    chunk_data[nb_key]['indices'].tolist()
                )

    # Second ring of planar neighbours
    for key in first_ring:
        for nb_key in get_neighbours(key):
            if nb_key not in chunk_data:
                continue
            if (chunk_data[nb_key]['is_planar'] and
                    nb_key not in region_key_set and
                    nb_key not in first_ring):
                planar_nb_indices.update(
                    chunk_data[nb_key]['indices'].tolist()
                )

    if planar_nb_indices:
        nb_idx     = np.array(list(planar_nb_indices))
        planar_pts = xyz[nb_idx]
        planar_nrm = normals[nb_idx]
        planar_clr = colors[nb_idx] if has_color else None

        pcd_ctx = o3d.geometry.PointCloud()
        pcd_ctx.points  = o3d.utility.Vector3dVector(planar_pts)
        pcd_ctx.normals = o3d.utility.Vector3dVector(planar_nrm)
        if planar_clr is not None:
            pcd_ctx.colors = o3d.utility.Vector3dVector(planar_clr)
        pcd_ctx_down = pcd_ctx.voxel_down_sample(voxel_size=CONTEXT_VOX)

        ctx_pts = np.asarray(pcd_ctx_down.points).astype(np.float32)
        ctx_nrm = np.asarray(pcd_ctx_down.normals).astype(np.float32)
        ctx_clr = (np.asarray(pcd_ctx_down.colors).astype(np.float32)
                   if has_color else None)

        pts_list.append(ctx_pts)
        nrm_list.append(ctx_nrm)
        if has_color:
            clr_list.append(ctx_clr)

    pts_merged = np.concatenate(pts_list, axis=0)
    nrm_merged = np.concatenate(nrm_list, axis=0)
    clr_merged = np.concatenate(clr_list, axis=0) if has_color else None

    n_complex_pts = len(region_pts)
    n_context_pts = len(pts_merged) - n_complex_pts
    print(f"    Complex: {n_complex_pts:,} pts (full density) + "
          f"context: {n_context_pts:,} pts → total: {len(pts_merged):,}")

    # Try full density first
    result = run_nksr(pts_merged, nrm_merged, clr_merged,
                      1.0, 2, VOXEL_SIZE * 0.5)

    if result is None:
        # OOM — subsample complex part only, keep full context
        print(f"    OOM — subsampling complex part only...")
        ctx_pts = pts_merged[n_complex_pts:]
        ctx_nrm = nrm_merged[n_complex_pts:]
        ctx_clr = clr_merged[n_complex_pts:] if has_color else None

        # Try progressively harder subsampling
        for max_c in [500_000, 300_000, 150_000, 50_000]:
            c_pts, c_nrm, c_clr = smart_subsample(
                pts_merged[:n_complex_pts],
                nrm_merged[:n_complex_pts],
                clr_merged[:n_complex_pts] if has_color else None,
                max_c, VOXEL_SIZE * 0.5,
            )
            pts_try = np.concatenate([c_pts, ctx_pts])
            nrm_try = np.concatenate([c_nrm, ctx_nrm])
            clr_try = np.concatenate([c_clr, ctx_clr]) if has_color else None

            print(f"      Trying {len(pts_try):,} pts "
                  f"(complex capped at {max_c:,})...")
            result = run_nksr(pts_try, nrm_try, clr_try,
                              1.0, 2, VOXEL_SIZE * 0.5)
            if result is not None:
                break

        # If still failing, fall back to coarser voxel
        if result is None:
            print(f"      Falling back to vox=0.1...")
            pts_s, nrm_s, clr_s = smart_subsample(
                pts_merged, nrm_merged, clr_merged,
                100_000, VOXEL_SIZE
            )
            result = run_nksr(pts_s, nrm_s, clr_s,
                              1.0, 2, VOXEL_SIZE)
            if result is None:
                return None

    verts, faces, mesh = result
    return save_mesh(verts, faces, mesh, core_min, core_max, label)

# ── 9. Reconstruct all regions ────────────────────────────────────────────
print("\nReconstructing regions...")

for region_idx, region_keys in enumerate(regions):
    region_indices = np.concatenate([chunk_data[k]['indices']
                                     for k in region_keys])
    region_pts     = xyz[region_indices]

    if len(region_pts) < MIN_PTS:
        continue

    is_planar_region = all(chunk_data[k]['is_planar'] for k in region_keys)
    max_residual     = np.max([chunk_data[k]['residual']
                               for k in region_keys])

    touches_boundary = False
    for key in region_keys:
        for nb_key in get_neighbours(key):
            if nb_key not in chunk_data:
                continue
            if chunk_data[nb_key]['is_planar'] != is_planar_region:
                touches_boundary = True
                break
        if touches_boundary:
            break

    overlap_m = OVERLAP_BOUNDARY if touches_boundary else OVERLAP_DEFAULT

    if is_planar_region:
        if max_residual < 0.02:
            detail, mise, vox_size = 0.3, 1, VOXEL_SIZE * 2.0
        else:
            detail, mise, vox_size = 0.5, 1, VOXEL_SIZE * 1.5
        if touches_boundary:
            mise = max(mise, 2)

        print(f"  Region {region_idx+1}/{len(regions)} [planar]: "
              f"{len(region_pts):,} pts | "
              f"residual={max_residual:.4f} | "
              f"boundary={'yes' if touches_boundary else 'no'} | "
              f"detail={detail}, vox={vox_size:.3f}")

        path = reconstruct_planar(
            region_pts, region_indices,
            detail, mise, vox_size,
            overlap_m,
            label=f"planar_{region_idx:05d}"
        )

    else:
        print(f"  Region {region_idx+1}/{len(regions)} [complex]: "
              f"{len(region_pts):,} pts | "
              f"residual={max_residual:.4f} | "
              f"boundary={'yes' if touches_boundary else 'no'}")

        path = reconstruct_complex(
            region_pts, region_indices,
            region_keys,
            label=f"complex_{region_idx:05d}"
        )

    if path:
        chunk_files.append(path)

# ── 10. Merge all PLYs ────────────────────────────────────────────────────
print(f"\nMerging {len(chunk_files)} files...")
merged = o3d.geometry.TriangleMesh()
for i, path in enumerate(chunk_files):
    print(f"  {i+1}/{len(chunk_files)}: {path}")
    chunk = o3d.io.read_triangle_mesh(path)
    merged += chunk
    del chunk

out_path = "outputs/reconstruction.ply"
o3d.io.write_triangle_mesh(out_path, merged)
print(f"Saved: {len(merged.vertices):,} vertices, "
      f"{len(merged.triangles):,} faces → {out_path}")