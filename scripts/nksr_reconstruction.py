import os
import shutil
import numpy as np
import torch
import laspy
import open3d as o3d
import nksr
import yaml
from collections import defaultdict

# ── Load config ───────────────────────────────────────────────────────────
with open("configs/nksr_config.yaml") as f:
    cfg = yaml.safe_load(f)

POINTCLOUD_LAS = cfg['paths']['pointcloud_las']
POINTCLOUD_PLY = cfg['paths']['pointcloud_ply']
OUTPUT_DIR     = cfg['paths']['output_dir']

VOXEL_SIZE  = cfg['voxel']['base_size']
FINE_SIZE   = cfg['voxel']['fine_chunk_size']

RESIDUAL_THRESHOLD      = cfg['planarity']['residual_threshold']
ANGLE_THRESHOLD_DEG     = cfg['planarity']['angle_threshold_deg']
COPLANAR_DIST_THRESHOLD = cfg['planarity']['coplanar_dist_threshold']
MIN_PTS_PER_CHUNK       = cfg['planarity']['min_points_per_chunk']

COMPLEX_DETAIL          = cfg['reconstruction']['complex_detail_level']
COMPLEX_MISE            = cfg['reconstruction']['complex_mise_iter']
COMPLEX_VOX_FACTOR      = cfg['reconstruction']['complex_voxel_factor']

PLANAR_VERY_FLAT_THRESH = cfg['reconstruction']['planar_very_flat_threshold']
PLANAR_VF_DETAIL        = cfg['reconstruction']['planar_very_flat_detail']
PLANAR_VF_MISE          = cfg['reconstruction']['planar_very_flat_mise_iter']
PLANAR_VF_VOX_FACTOR    = cfg['reconstruction']['planar_very_flat_voxel_factor']

PLANAR_DETAIL           = cfg['reconstruction']['planar_flat_detail']
PLANAR_MISE             = cfg['reconstruction']['planar_flat_mise_iter']
PLANAR_VOX_FACTOR       = cfg['reconstruction']['planar_flat_voxel_factor']

PLANAR_MAX_PTS          = cfg['subsampling']['planar_max_pts']
OOM_FALLBACK_LEVELS     = cfg['subsampling']['complex_oom_fallback_levels']
LAST_RESORT_PTS         = cfg['subsampling']['complex_last_resort_pts']
LAST_RESORT_VOX_FACTOR  = cfg['subsampling']['complex_last_resort_voxel_factor']

MIN_PTS_PER_UNIT        = cfg['misc']['min_pts_per_unit']
GPU_DEVICE              = cfg['misc']['gpu_device']

CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists(CHUNKS_DIR):
    shutil.rmtree(CHUNKS_DIR)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ── 1. Load point cloud ───────────────────────────────────────────────────
print("Loading point cloud...")
las = laspy.read(POINTCLOUD_LAS)
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

# ── 2. Load normals from PLY ──────────────────────────────────────────────
print("Loading normals from PLY...")
pcd = o3d.io.read_point_cloud(POINTCLOUD_PLY)
normals = np.asarray(pcd.normals).astype(np.float32)
del pcd
print("Normals loaded.")

# ── 3. Assign points to fine chunks ───────────────────────────────────────
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

def normals_similar(n1, n2):
    cos_angle = abs(np.dot(n1, n2))
    return cos_angle > np.cos(np.radians(ANGLE_THRESHOLD_DEG))

def planes_coplanar(n1, d1, pts2):
    distances = np.abs(pts2 @ n1 - d1)
    return distances.mean() < COPLANAR_DIST_THRESHOLD

print("Computing per-chunk planarity...")
chunk_data = {}
for key, indices in point_chunks.items():
    if len(indices) < MIN_PTS_PER_CHUNK:
        continue
    indices_np = np.array(indices)
    pts        = xyz[indices_np]
    normal, d, residual = fit_plane(pts)
    is_planar  = residual < RESIDUAL_THRESHOLD
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

# ── 5. Neighbours ─────────────────────────────────────────────────────────
def get_neighbours(key):
    ix, iy, iz = key
    return [
        (ix+1, iy, iz), (ix-1, iy, iz),
        (ix, iy+1, iz), (ix, iy-1, iz),
        (ix, iy, iz+1), (ix, iy, iz-1),
    ]

# ── 6. Region growing — complex first ────────────────────────────────────
print("Growing complex regions (absorbing adjacent planar)...")
visited = set()
regions = []

# Pass 1 — complex regions absorb all touching planar
complex_keys = [k for k, v in chunk_data.items() if not v['is_planar']]

for start_key in complex_keys:
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

        for nb_key in get_neighbours(key):
            if nb_key in visited or nb_key not in chunk_data:
                continue

            nb  = chunk_data[nb_key]
            cur = chunk_data[key]

            if not cur['is_planar'] and not nb['is_planar']:
                queue.append(nb_key)
            elif not cur['is_planar'] and nb['is_planar']:
                queue.append(nb_key)
            elif cur['is_planar'] and not nb['is_planar']:
                queue.append(nb_key)
            elif cur['is_planar'] and nb['is_planar']:
                if (normals_similar(cur['normal'], nb['normal']) and
                        planes_coplanar(cur['normal'], cur['d'],
                                        xyz[nb['indices']])):
                    queue.append(nb_key)

    if region:
        regions.append((region, True))

print(f"Complex units: {len(regions)}")

# Pass 2 — grow remaining planar regions
print("Growing remaining planar regions...")
planar_keys = [k for k, v in chunk_data.items()
               if v['is_planar'] and k not in visited]

for start_key in planar_keys:
    if start_key in visited:
        continue

    region = []
    queue  = [start_key]

    while queue:
        key = queue.pop()
        if key in visited or key not in chunk_data:
            continue
        if not chunk_data[key]['is_planar']:
            continue

        visited.add(key)
        region.append(key)
        chunk = chunk_data[key]

        for nb_key in get_neighbours(key):
            if nb_key in visited or nb_key not in chunk_data:
                continue
            nb = chunk_data[nb_key]
            if not nb['is_planar']:
                continue

            if (normals_similar(chunk['normal'], nb['normal']) and
                    planes_coplanar(chunk['normal'], chunk['d'],
                                    xyz[nb['indices']])):
                queue.append(nb_key)

    if region:
        regions.append((region, False))

print(f"Total reconstruction units: {len(regions)}")
print(f"  Complex units:      {sum(1 for _, c in regions if c)}")
print(f"  Planar-only units:  {sum(1 for _, c in regions if not c)}")

# ── 7. Reconstruction helpers ─────────────────────────────────────────────
reconstructor = nksr.Reconstructor(torch.device(GPU_DEVICE))
chunk_files   = []
chunk_counter = [0]

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
    chunk_mesh = o3d.geometry.TriangleMesh()
    chunk_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    chunk_mesh.triangles = o3d.utility.Vector3iVector(faces)
    if has_color and hasattr(mesh, 'c') and mesh.c is not None:
        vc = np.clip(mesh.c.cpu().numpy(), 0.0, 1.0)
        chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    chunk_path = os.path.join(CHUNKS_DIR,
                              f"{label}_{chunk_counter[0]:05d}.ply")
    chunk_counter[0] += 1
    o3d.io.write_triangle_mesh(chunk_path, chunk_mesh)
    return chunk_path

# ── 8. Reconstruct all units ──────────────────────────────────────────────
print("\nReconstructing units...")

for unit_idx, (unit_keys, is_complex) in enumerate(regions):
    all_indices = np.concatenate([chunk_data[k]['indices']
                                   for k in unit_keys])
    unit_pts = xyz[all_indices]

    if len(unit_pts) < MIN_PTS_PER_UNIT:
        continue

    unit_nrm = normals[all_indices]
    unit_clr = colors[all_indices] if has_color else None

    if is_complex:
        detail   = COMPLEX_DETAIL
        mise     = COMPLEX_MISE
        vox_size = VOXEL_SIZE * COMPLEX_VOX_FACTOR
        label    = f"complex_{unit_idx:05d}"
    else:
        max_residual = np.max([chunk_data[k]['residual']
                                for k in unit_keys])
        if max_residual < PLANAR_VERY_FLAT_THRESH:
            detail   = PLANAR_VF_DETAIL
            mise     = PLANAR_VF_MISE
            vox_size = VOXEL_SIZE * PLANAR_VF_VOX_FACTOR
        else:
            detail   = PLANAR_DETAIL
            mise     = PLANAR_MISE
            vox_size = VOXEL_SIZE * PLANAR_VOX_FACTOR
        label = f"planar_{unit_idx:05d}"

    print(f"  Unit {unit_idx+1}/{len(regions)} "
          f"[{'complex' if is_complex else 'planar'}]: "
          f"{len(unit_pts):,} pts | vox={vox_size:.3f}")

    if not is_complex:
        unit_pts, unit_nrm, unit_clr = smart_subsample(
            unit_pts, unit_nrm, unit_clr, PLANAR_MAX_PTS, vox_size
        )

    result = run_nksr(unit_pts, unit_nrm, unit_clr,
                      detail, mise, vox_size)

    if result is None and is_complex:
        print(f"    OOM — subsampling...")
        for max_c in OOM_FALLBACK_LEVELS:
            s_pts, s_nrm, s_clr = smart_subsample(
                unit_pts, unit_nrm, unit_clr, max_c, vox_size
            )
            print(f"      Trying {len(s_pts):,} pts...")
            result = run_nksr(s_pts, s_nrm, s_clr,
                              detail, mise, vox_size)
            if result is not None:
                break

        if result is None:
            print(f"      Falling back to vox={VOXEL_SIZE * LAST_RESORT_VOX_FACTOR}...")
            s_pts, s_nrm, s_clr = smart_subsample(
                unit_pts, unit_nrm, unit_clr,
                LAST_RESORT_PTS,
                VOXEL_SIZE * LAST_RESORT_VOX_FACTOR
            )
            result = run_nksr(s_pts, s_nrm, s_clr,
                              detail, mise,
                              VOXEL_SIZE * LAST_RESORT_VOX_FACTOR)

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

out_path = os.path.join(OUTPUT_DIR, "nksr_reconstruction.ply")
o3d.io.write_triangle_mesh(out_path, merged)
print(f"Saved: {len(merged.vertices):,} vertices, "
      f"{len(merged.triangles):,} faces → {out_path}")