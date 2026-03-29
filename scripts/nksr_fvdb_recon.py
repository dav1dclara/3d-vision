import os
import shutil
import numpy as np
import torch
import laspy
import open3d as o3d
import nksr
from collections import defaultdict

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

# ── 2. High accuracy normal estimation ────────────────────────────────────
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
regions = []  # list of (region_keys, is_planar)

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

    is_planar_region = all(chunk_data[k]['is_planar'] for k in region)
    regions.append((region, is_planar_region))

print(f"Total regions: {len(regions)}")
planar_regions  = [(r, p) for r, p in regions if p]
complex_regions = [(r, p) for r, p in regions if not p]
print(f"Planar regions: {len(planar_regions)}, "
      f"Complex regions: {len(complex_regions)}")

# ── 6. Build reconstruction units ────────────────────────────────────────
# Each reconstruction unit is a set of fine chunk keys that get
# reconstructed together in one NKSR call.
# Strategy:
# - Complex region + ALL adjacent planar chunks → one unit
# - Small/medium planar regions adjacent to complex → absorbed
# - Large isolated planar regions → own unit

SMALL_PLANAR_PTS  = 500
MEDIUM_PLANAR_PTS = 2000

# Map each chunk key to its region index
chunk_to_region = {}
for region_idx, (region_keys, is_planar) in enumerate(regions):
    for key in region_keys:
        chunk_to_region[key] = region_idx

# Find which planar regions are adjacent to each complex region
complex_adjacent_planar = defaultdict(set)  # complex_region_idx -> set of planar region idxs
for region_idx, (region_keys, is_planar) in enumerate(regions):
    if is_planar:
        continue
    for key in region_keys:
        for nb_key in get_neighbours(key):
            if nb_key not in chunk_data:
                continue
            nb_region_idx = chunk_to_region.get(nb_key)
            if nb_region_idx is None:
                continue
            if regions[nb_region_idx][1]:  # neighbour is planar
                complex_adjacent_planar[region_idx].add(nb_region_idx)

# Determine which planar regions get absorbed into complex units
absorbed_planar = set()  # planar region indices that are absorbed

reconstruction_units = []  # list of (chunk_keys_set, label)

# Build complex units — each complex region + its adjacent planar regions
for region_idx, (region_keys, is_planar) in enumerate(regions):
    if is_planar:
        continue

    unit_keys = set(region_keys)
    adjacent_planar_idxs = complex_adjacent_planar[region_idx]

    for planar_idx in adjacent_planar_idxs:
        planar_keys, _ = regions[planar_idx]
        n_pts = sum(chunk_data[k]['n_pts'] for k in planar_keys)

        # Always absorb small and medium planar regions
        # Also absorb large ones that are adjacent to this complex region
        unit_keys.update(planar_keys)
        absorbed_planar.add(planar_idx)

    reconstruction_units.append((unit_keys, f"complex_{region_idx:05d}"))

# Build planar-only units for non-absorbed planar regions
for region_idx, (region_keys, is_planar) in enumerate(regions):
    if not is_planar:
        continue
    if region_idx in absorbed_planar:
        continue

    n_pts = sum(chunk_data[k]['n_pts'] for k in region_keys)
    reconstruction_units.append((set(region_keys),
                                  f"planar_{region_idx:05d}"))

print(f"Reconstruction units: {len(reconstruction_units)}")
print(f"  Complex+planar units: "
      f"{sum(1 for _, l in reconstruction_units if 'complex' in l)}")
print(f"  Planar-only units: "
      f"{sum(1 for _, l in reconstruction_units if 'planar' in l)}")

# ── 7. Reconstruction helpers ─────────────────────────────────────────────
reconstructor  = nksr.Reconstructor(torch.device("cuda:0"))
MIN_PTS        = 200
chunk_files    = []
chunk_counter  = [0]

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

def save_mesh(verts, faces, mesh, label):
    """Save without trimming — no boundary artifacts."""
    chunk_mesh = o3d.geometry.TriangleMesh()
    chunk_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    chunk_mesh.triangles = o3d.utility.Vector3iVector(faces)
    if has_color and hasattr(mesh, 'c') and mesh.c is not None:
        vc = np.clip(mesh.c.cpu().numpy(), 0.0, 1.0)
        chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    chunk_path = f"outputs/chunks/{label}_{chunk_counter[0]:05d}.ply"
    chunk_counter[0] += 1
    o3d.io.write_triangle_mesh(chunk_path, chunk_mesh)
    return chunk_path

# ── 8. Reconstruct all units ──────────────────────────────────────────────
print("\nReconstructing units...")

for unit_idx, (unit_keys, label) in enumerate(reconstruction_units):
    # Collect all point indices in this unit
    all_indices = np.concatenate([chunk_data[k]['indices']
                                   for k in unit_keys])
    unit_pts = xyz[all_indices]

    if len(unit_pts) < MIN_PTS:
        continue

    unit_nrm = normals[all_indices]
    unit_clr = colors[all_indices] if has_color else None

    is_complex_unit = 'complex' in label
    n_pts = len(unit_pts)

    # Adaptive quality
    if is_complex_unit:
        detail, mise, vox_size = 1.0, 2, VOXEL_SIZE * 0.5
    else:
        # Planar-only unit
        max_residual = np.max([chunk_data[k]['residual']
                                for k in unit_keys])
        if max_residual < 0.02:
            detail, mise, vox_size = 0.3, 1, VOXEL_SIZE * 2.0
        else:
            detail, mise, vox_size = 0.5, 1, VOXEL_SIZE * 1.5

    print(f"  Unit {unit_idx+1}/{len(reconstruction_units)} "
          f"[{'complex+planar' if is_complex_unit else 'planar'}]: "
          f"{n_pts:,} pts | vox={vox_size:.3f}")

    # Try full density first for complex units
    # Subsample for planar-only units
    if not is_complex_unit:
        unit_pts, unit_nrm, unit_clr = smart_subsample(
            unit_pts, unit_nrm, unit_clr, 300_000, vox_size
        )

    result = run_nksr(unit_pts, unit_nrm, unit_clr,
                      detail, mise, vox_size)

    if result is None and is_complex_unit:
        # OOM — progressively subsample
        print(f"    OOM — subsampling...")
        for max_c in [500_000, 300_000, 150_000, 50_000]:
            s_pts, s_nrm, s_clr = smart_subsample(
                unit_pts, unit_nrm, unit_clr, max_c, vox_size
            )
            print(f"      Trying {len(s_pts):,} pts...")
            result = run_nksr(s_pts, s_nrm, s_clr,
                              detail, mise, vox_size)
            if result is not None:
                break

        if result is None:
            print(f"      Falling back to vox=0.1...")
            s_pts, s_nrm, s_clr = smart_subsample(
                unit_pts, unit_nrm, unit_clr, 100_000, VOXEL_SIZE
            )
            result = run_nksr(s_pts, s_nrm, s_clr,
                              1.0, 2, VOXEL_SIZE)

    if result is None:
        print(f"    Failed, skipping.")
        continue

    verts, faces, mesh = result
    path = save_mesh(verts, faces, mesh, label)
    if path:
        chunk_files.append(path)

# ── 9. Merge all PLYs ─────────────────────────────────────────────────────
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