import os
import shutil
import argparse
import numpy as np
import torch
import laspy
import open3d as o3d
import nksr
import yaml
from collections import defaultdict
from datetime import datetime

start = datetime.now()
t = {}   # stage timings

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NKSR chunked reconstruction")
parser.add_argument("--config", default="configs/nksr_config.yaml",
                    help="Path to YAML config")
parser.add_argument("--save-chunks", action="store_true",
                    help="Keep individual chunk PLY files after merge")
parser.add_argument("--save-boundaries", action="store_true",
                    help="Save chunk_boundaries.ply with region bounding boxes")
parser.add_argument("--save-voxel-grid", action="store_true",
                    help="Save voxel_grid.ply with individual 0.25m fine-cell outlines")
args = parser.parse_args()

# ── Load config ───────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

POINTCLOUD_LAS = cfg['paths']['pointcloud_las']
POINTCLOUD_PLY = cfg['paths']['pointcloud_ply']
OUTPUT_DIR     = cfg['paths']['output_dir']
OUTPUT_MESH    = cfg['paths']['output_mesh']

VOXEL_SIZE  = cfg['voxel']['base_size']
FINE_SIZE   = cfg['voxel']['fine_chunk_size']

RESIDUAL_THRESHOLD      = cfg['planarity']['residual_threshold']
ANGLE_THRESHOLD_DEG     = cfg['planarity']['angle_threshold_deg']
COPLANAR_DIST_THRESHOLD = cfg['planarity']['coplanar_dist_threshold']
MIN_PTS_PER_CHUNK       = cfg['planarity']['min_points_per_chunk']

COMPLEX_DETAIL       = cfg['reconstruction']['complex_detail_level']
COMPLEX_MISE         = cfg['reconstruction']['complex_mise_iter']
COMPLEX_VOX_FACTOR   = cfg['reconstruction']['complex_voxel_factor']
COMPLEX_MAX_EXTENT_M = cfg['reconstruction']['complex_max_extent_m']
PLANAR_OVERLAP_M     = cfg['reconstruction']['planar_overlap_m']

PLANAR_VERY_FLAT_THRESH = cfg['reconstruction']['planar_very_flat_threshold']
PLANAR_VF_DETAIL        = cfg['reconstruction']['planar_very_flat_detail']
PLANAR_VF_MISE          = cfg['reconstruction']['planar_very_flat_mise_iter']
PLANAR_VF_VOX_FACTOR    = cfg['reconstruction']['planar_very_flat_voxel_factor']

PLANAR_DETAIL     = cfg['reconstruction']['planar_flat_detail']
PLANAR_MISE       = cfg['reconstruction']['planar_flat_mise_iter']
PLANAR_VOX_FACTOR = cfg['reconstruction']['planar_flat_voxel_factor']

PLANAR_MAX_PTS   = cfg['subsampling']['planar_max_pts']
MIN_PTS_PER_UNIT = cfg['misc']['min_pts_per_unit']
GPU_DEVICE       = cfg['misc']['gpu_device']

CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists(CHUNKS_DIR):
    shutil.rmtree(CHUNKS_DIR)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ── 1. Load point cloud ───────────────────────────────────────────────────
t['start_load'] = datetime.now()
print("Loading point cloud...")
if POINTCLOUD_LAS:
    las = laspy.read(POINTCLOUD_LAS)
    xyz = np.vstack([las.x, las.y, las.z]).T  # float64 for precision

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
    xyz = xyz.astype(np.float32)

    print(f"Loaded {len(xyz):,} points.")
    print("Loading normals from PLY...")
    pcd = o3d.io.read_point_cloud(POINTCLOUD_PLY)
    normals = np.asarray(pcd.normals).astype(np.float32)
    del pcd
    print("Normals loaded.")
else:
    print("No LAS file configured — loading XYZ, normals and colors from PLY...")
    pcd = o3d.io.read_point_cloud(POINTCLOUD_PLY)
    xyz = np.asarray(pcd.points)  # float64

    centroid = xyz.mean(axis=0)
    xyz -= centroid
    xyz = xyz.astype(np.float32)

    normals = np.asarray(pcd.normals).astype(np.float32)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors).astype(np.float32)
        has_color = True
        print("Color data found.")
    else:
        has_color = False
        print("No color data found.")

    del pcd
    print(f"Loaded {len(xyz):,} points.")

# ── 3. Assign points to fine chunks ───────────────────────────────────────
t['start_voxelize'] = datetime.now()
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

t['start_svd'] = datetime.now()
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
residuals = np.array([c['residual'] for c in chunk_data.values()])
for pct in [50, 75, 90, 95, 99, 100]:
    print(f"  residual p{pct}: {np.percentile(residuals, pct):.4f}")

# ── 5. Neighbours ─────────────────────────────────────────────────────────
def get_neighbours(key):
    ix, iy, iz = key
    return [
        (ix+1, iy, iz), (ix-1, iy, iz),
        (ix, iy+1, iz), (ix, iy-1, iz),
        (ix, iy, iz+1), (ix, iy, iz-1),
    ]

MAX_EXTENT_CHUNKS  = int(COMPLEX_MAX_EXTENT_M / FINE_SIZE)
COMPLEX_MAX_PTS    = cfg['reconstruction']['complex_max_pts']

# ── 6. Region growing ─────────────────────────────────────────────────────
t['start_growing'] = datetime.now()
print("Growing complex regions (absorbing adjacent planar)...")
visited = set()
regions = []

complex_keys = [k for k, v in chunk_data.items() if not v['is_planar']]

for start_key in complex_keys:
    if start_key in visited:
        continue

    region    = []
    queue     = [start_key]
    bbox_min  = np.array(start_key)
    bbox_max  = np.array(start_key)
    total_pts = 0

    while queue:
        key = queue.pop()
        if key in visited or key not in chunk_data:
            continue

        cur = chunk_data[key]

        # Extent cap only on complex-to-complex merging
        if not cur['is_planar']:
            key_arr  = np.array(key)
            new_min  = np.minimum(bbox_min, key_arr)
            new_max  = np.maximum(bbox_max, key_arr)
            if (new_max - new_min).max() > MAX_EXTENT_CHUNKS:
                continue
            bbox_min = new_min
            bbox_max = new_max

        if total_pts + cur['n_pts'] > COMPLEX_MAX_PTS:
            continue

        visited.add(key)
        region.append(key)
        total_pts += cur['n_pts']

        for nb_key in get_neighbours(key):
            if nb_key in visited or nb_key not in chunk_data:
                continue

            nb = chunk_data[nb_key]

            if not cur['is_planar'] and not nb['is_planar']:
                queue.append(nb_key)
            elif not cur['is_planar'] and nb['is_planar']:
                queue.append(nb_key)
            elif cur['is_planar'] and not nb['is_planar']:
                queue.append(nb_key)
            # planar-to-planar chaining removed: complex regions only absorb
            # the immediately adjacent planar layer, not entire surfaces

    if region:
        regions.append((region, True))

print(f"Complex units: {len(regions)}")

# Track absorbed planar chunks
absorbed_planar_chunks = set()
for region_keys, is_complex in regions:
    if is_complex:
        for key in region_keys:
            if chunk_data[key]['is_planar']:
                absorbed_planar_chunks.add(key)
print(f"Absorbed planar chunks: {len(absorbed_planar_chunks)}")

# Build world-space bboxes for all complex units — used for planar trimming
complex_bboxes = []
for region_keys, is_complex in regions:
    if is_complex:
        indices = np.concatenate([chunk_data[k]['indices']
                                   for k in region_keys])
        pts = xyz[indices]
        complex_bboxes.append((pts.min(axis=0), pts.max(axis=0)))

print("Growing remaining planar regions (unlimited)...")
planar_keys = [k for k, v in chunk_data.items()
               if v['is_planar']
               and k not in visited
               and k not in absorbed_planar_chunks]

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
print(f"  Complex units:     {sum(1 for _, c in regions if c)}")
print(f"  Planar-only units: {sum(1 for _, c in regions if not c)}")

# ── 7. Reconstruction helpers ─────────────────────────────────────────────
reconstructor = nksr.Reconstructor(torch.device(GPU_DEVICE))
chunk_files   = []
chunk_counter = [0]

def run_nksr(pts_np, nrm_np, clr_np, detail, mise, vox_size):
    global reconstructor
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
    except torch.cuda.OutOfMemoryError as e:
        print(f"      NKSR OOM: {e}")
        torch.cuda.empty_cache()
        return None
    except (AttributeError, RuntimeError) as e:
        print(f"      NKSR failed: {e}")
        # Fatal CUDA errors corrupt the device context — reinitialize to recover
        try:
            torch.cuda.empty_cache()
            reconstructor = nksr.Reconstructor(torch.device(GPU_DEVICE))
            print("      Reconstructor reinitialized.")
        except Exception as reinit_err:
            print(f"      Reinit failed: {reinit_err}")
        return None

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

def save_mesh(verts, faces, mesh_obj, label, used_indices=None):
    chunk_mesh = o3d.geometry.TriangleMesh()
    chunk_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    chunk_mesh.triangles = o3d.utility.Vector3iVector(faces)
    if has_color and hasattr(mesh_obj, 'c') and mesh_obj.c is not None:
        vc = np.clip(mesh_obj.c.cpu().numpy(), 0.0, 1.0)
        if used_indices is not None:
            vc = vc[used_indices]
        chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    chunk_path = os.path.join(CHUNKS_DIR,
                              f"{label}_{chunk_counter[0]:05d}.ply")
    chunk_counter[0] += 1
    o3d.io.write_triangle_mesh(chunk_path, chunk_mesh)
    return chunk_path

def mask_planar_verts(verts, core_min_w, core_max_w):
    """Keep planar mesh vertices inside the core bbox, minus any complex unit
    bbox that actually overlaps the planar core (avoids double surfaces at
    complex/planar boundaries without over-trimming distant planar regions)."""
    in_core = np.all((verts >= core_min_w) & (verts <= core_max_w), axis=1)
    for cplx_min, cplx_max in complex_bboxes:
        cplx_min_w = cplx_min + centroid
        cplx_max_w = cplx_max + centroid
        # skip complex bboxes that don't overlap this planar unit's core at all
        if np.any(cplx_max_w < core_min_w) or np.any(cplx_min_w > core_max_w):
            continue
        in_complex = np.all((verts >= cplx_min_w) & (verts <= cplx_max_w), axis=1)
        in_core &= ~in_complex
    return in_core

# ── 8. Reconstruct all units ──────────────────────────────────────────────
t['start_reconstruction'] = datetime.now()
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

        print(f"  Unit {unit_idx+1}/{len(regions)} [complex]: "
              f"{len(unit_pts):,} pts | vox={vox_size:.3f}")

        result = run_nksr(unit_pts, unit_nrm, unit_clr,
                          detail, mise, vox_size)

        if result is None:
            print(f"    Failed, skipping.")
            continue

        # No trimming — complex mesh covers the transition zone
        verts, faces, mesh_obj = result
        path = save_mesh(verts, faces, mesh_obj, label)
        if path:
            chunk_files.append(path)

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

        core_min = unit_pts.min(axis=0)
        core_max = unit_pts.max(axis=0)

        # Expand with overlap for reconstruction context
        rmin = core_min - PLANAR_OVERLAP_M
        rmax = core_max + PLANAR_OVERLAP_M
        overlap_mask = np.all((xyz >= rmin) & (xyz <= rmax), axis=1)

        exp_pts = xyz[overlap_mask]
        exp_nrm = normals[overlap_mask]
        exp_clr = colors[overlap_mask] if has_color else None

        exp_pts, exp_nrm, exp_clr = smart_subsample(
            exp_pts, exp_nrm, exp_clr, PLANAR_MAX_PTS, vox_size
        )

        print(f"  Unit {unit_idx+1}/{len(regions)} [planar]: "
              f"{len(unit_pts):,} pts | expanded: {len(exp_pts):,} pts | "
              f"vox={vox_size:.3f}")

        result = run_nksr(exp_pts, exp_nrm, exp_clr,
                          detail, mise, vox_size)

        if result is None:
            print(f"    Failed, skipping.")
            continue

        verts, faces, mesh_obj = result

        core_min_w = core_min + centroid
        core_max_w = core_max + centroid
        in_core = mask_planar_verts(verts, core_min_w, core_max_w)
        face_mask  = (in_core[faces[:, 0]] &
                      in_core[faces[:, 1]] &
                      in_core[faces[:, 2]])
        faces_core = faces[face_mask]

        if len(faces_core) == 0:
            print(f"    Empty after trim, skipping.")
            continue

        used        = np.unique(faces_core)
        remap       = np.full(len(verts), -1)
        remap[used] = np.arange(len(used))
        verts_core  = verts[used]
        faces_core  = remap[faces_core]

        path = save_mesh(verts_core, faces_core, mesh_obj,
                         label, used_indices=used)
        if path:
            chunk_files.append(path)

# ── 9. Merge all PLYs ─────────────────────────────────────────────────────
t['start_merge'] = datetime.now()
print(f"\nMerging {len(chunk_files)} files...")
merged = o3d.geometry.TriangleMesh()
for i, path in enumerate(chunk_files):
    print(f"  {i+1}/{len(chunk_files)}: {path}")
    chunk = o3d.io.read_triangle_mesh(path)
    merged += chunk
    del chunk

out_path = os.path.join(OUTPUT_DIR, OUTPUT_MESH)
o3d.io.write_triangle_mesh(out_path, merged)
print(f"Saved: {len(merged.vertices):,} vertices, "
      f"{len(merged.triangles):,} faces → {out_path}")

if args.save_chunks:
    print(f"Chunk PLYs kept in: {CHUNKS_DIR}")
else:
    shutil.rmtree(CHUNKS_DIR)

# ── 10. Optional: chunk boundary wireframe ────────────────────────────────
if args.save_boundaries:
    print("Building chunk boundary PLY...")
    all_points = []
    all_lines  = []
    all_colors = []
    offset = 0

    COMPLEX_COLOR = [1.0, 0.2, 0.2]  # red  — complex regions
    PLANAR_COLOR  = [0.2, 0.4, 1.0]  # blue — planar regions

    for unit_keys, is_complex in regions:
        valid_keys = [k for k in unit_keys if k in chunk_data]
        if not valid_keys:
            continue
        all_idx = np.concatenate([chunk_data[k]['indices'] for k in valid_keys])
        pts_w   = xyz[all_idx] + centroid  # world space
        bmin    = pts_w.min(axis=0)
        bmax    = pts_w.max(axis=0)

        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [
            [offset+0, offset+1], [offset+1, offset+2],
            [offset+2, offset+3], [offset+3, offset+0],
            [offset+4, offset+5], [offset+5, offset+6],
            [offset+6, offset+7], [offset+7, offset+4],
            [offset+0, offset+4], [offset+1, offset+5],
            [offset+2, offset+6], [offset+3, offset+7],
        ]
        color = COMPLEX_COLOR if is_complex else PLANAR_COLOR

        all_points.append(corners)
        all_lines.extend(edges)
        all_colors.extend([color] * 12)
        offset += 8

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    ls.lines  = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector(all_colors)

    boundary_path = os.path.join(OUTPUT_DIR, "chunk_boundaries.ply")
    o3d.io.write_line_set(boundary_path, ls)
    print(f"Chunk boundaries saved → {boundary_path}")
    print(f"  {sum(1 for _,c in regions if c)} complex (red), "
          f"{sum(1 for _,c in regions if not c)} planar (blue)")

# ── 11. Optional: fine voxel cell wireframe ───────────────────────────────
if args.save_voxel_grid:
    print("Building voxel grid PLY...")
    all_points = []
    all_lines  = []
    all_colors = []
    offset = 0

    COMPLEX_COLOR = [1.0, 0.2, 0.2]
    PLANAR_COLOR  = [0.2, 0.4, 1.0]

    for key, data in chunk_data.items():
        ix, iy, iz = key
        bmin = np.array([ix,   iy,   iz  ], dtype=np.float32) * FINE_SIZE + centroid
        bmax = np.array([ix+1, iy+1, iz+1], dtype=np.float32) * FINE_SIZE + centroid

        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [
            [offset+0, offset+1], [offset+1, offset+2],
            [offset+2, offset+3], [offset+3, offset+0],
            [offset+4, offset+5], [offset+5, offset+6],
            [offset+6, offset+7], [offset+7, offset+4],
            [offset+0, offset+4], [offset+1, offset+5],
            [offset+2, offset+6], [offset+3, offset+7],
        ]
        color = COMPLEX_COLOR if not data['is_planar'] else PLANAR_COLOR

        all_points.append(corners)
        all_lines.extend(edges)
        all_colors.extend([color] * 12)
        offset += 8

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    ls.lines  = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector(all_colors)

    voxel_path = os.path.join(OUTPUT_DIR, "voxel_grid.ply")
    o3d.io.write_line_set(voxel_path, ls)
    print(f"Voxel grid saved → {voxel_path}  ({len(chunk_data):,} cells)")
    print(f"  {sum(1 for d in chunk_data.values() if not d['is_planar'])} complex (red), "
          f"{sum(1 for d in chunk_data.values() if d['is_planar'])} planar (blue)")

t['end'] = datetime.now()

def fmt(delta): return str(delta).split('.')[0]  # HH:MM:SS

print("\n── Timing summary ───────────────────────────────────────────")
print(f"  Load point cloud   : {fmt(t['start_voxelize']    - t['start_load'])}")
print(f"  Voxelization       : {fmt(t['start_svd']         - t['start_voxelize'])}")
print(f"  SVD / classification: {fmt(t['start_growing']    - t['start_svd'])}")
print(f"  Region growing     : {fmt(t['start_reconstruction'] - t['start_growing'])}")
print(f"  NKSR reconstruction: {fmt(t['start_merge']       - t['start_reconstruction'])}")
print(f"  Merge & write      : {fmt(t['end']               - t['start_merge'])}")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  Total              : {fmt(t['end'] - start)}")