"""Browse chunk PLY files in an interactive Open3D viewer.

Default mode  : loads one chunk at a time — fast, good for inspecting individual chunks.
--show-all    : loads all chunks upfront as dim grey; the current chunk is highlighted
                white. Use this to find chunks by their position in the full scene.

Controls:
    Right arrow / N  : next chunk
    Left  arrow / P  : previous chunk
    R                : reset camera to fit current chunk
    S                : save current view as transparent PNG (white geometry, black bg keyed out)
    Q / Esc          : quit

Usage:
    # Default — one chunk at a time
    python scripts/browse_chunks.py --chunks-dir outputs/chunks

    # Full-scene context, highlight current chunk
    python scripts/browse_chunks.py --chunks-dir outputs/chunks --show-all

    # Only browse complex chunks
    python scripts/browse_chunks.py --chunks-dir outputs/chunks --pattern "complex_*"

    # Save PNGs to a separate folder
    python scripts/browse_chunks.py --chunks-dir outputs/chunks --save-dir outputs/poster_chunks

Flags:
    --chunks-dir DIR    Directory containing chunk PLY files (required)
    --pattern GLOB      File filter (default: *.ply)
    --point-size FLOAT  Rendered point size (default: 2.0)
    --save-dir DIR      Where to write PNGs from S (default: same as --chunks-dir)
    --show-all          Pre-load all chunks; dim grey background, white highlight on current
"""

import argparse
import glob
import os
import numpy as np
import open3d as o3d
from PIL import Image

parser = argparse.ArgumentParser(description="Cycle through chunk PLY files")
parser.add_argument("--chunks-dir", required=True)
parser.add_argument("--pattern", default="*.ply")
parser.add_argument("--point-size", type=float, default=2.0)
parser.add_argument("--save-dir", default=None,
                    help="Directory for saved PNGs (default: same as --chunks-dir)")
parser.add_argument("--show-all", action="store_true",
                    help="Show all chunks as dim background; highlight current in white")
args = parser.parse_args()

SAVE_DIR = args.save_dir or args.chunks_dir
os.makedirs(SAVE_DIR, exist_ok=True)

BG_COLOR      = np.array([0.0, 0.0, 0.0])
HIGHLIGHT     = [0.85, 0.85, 0.85] # light grey — current chunk (room for shadows)
DIM_COLOR     = [0.2,  0.2,  0.2]  # dark grey — background chunks
BG_THRESHOLD  = 8

files = sorted(glob.glob(os.path.join(args.chunks_dir, args.pattern)))
if not files:
    print(f"No files matched in {args.chunks_dir}")
    raise SystemExit(1)

print(f"Found {len(files)} chunks. Use ← → to browse, S to save PNG, Q to quit.")

# ── pre-load all geometries ───────────────────────────────────────────────
def read_geom(path):
    geom = o3d.io.read_triangle_mesh(path)
    if len(geom.triangles) == 0:
        geom = o3d.io.read_point_cloud(path)
    else:
        geom.compute_vertex_normals()
    return geom

if args.show_all:
    print("Loading all chunks (--show-all mode)...")
    all_geoms = []
    for i, path in enumerate(files):
        g = read_geom(path)
        g.paint_uniform_color(DIM_COLOR)
        all_geoms.append(g)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")
    print("Done.")

# ── viewer ────────────────────────────────────────────────────────────────
idx = [0]
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Chunk browser", width=1280, height=720)

opt = vis.get_render_option()
opt.point_size = args.point_size
opt.background_color = BG_COLOR
opt.light_on = True
opt.mesh_show_back_face = True

current_geom = [None]   # tracked only in single-chunk mode

def print_label(i):
    print(f"[{i+1:>3}/{len(files)}]  {os.path.basename(files[i])}")

def refresh(new_idx):
    idx[0] = new_idx % len(files)
    print_label(idx[0])

    if args.show_all:
        # repaint previous chunk dim, new chunk white
        prev = current_geom[0]
        if prev is not None:
            all_geoms[prev].paint_uniform_color(DIM_COLOR)
            vis.update_geometry(all_geoms[prev])
        all_geoms[idx[0]].paint_uniform_color(HIGHLIGHT)
        vis.update_geometry(all_geoms[idx[0]])
        current_geom[0] = idx[0]
        vis.poll_events()
        vis.update_renderer()
    else:
        geom = read_geom(files[idx[0]])
        geom.paint_uniform_color(HIGHLIGHT)
        if current_geom[0] is not None:
            vis.remove_geometry(current_geom[0], reset_bounding_box=False)
        vis.add_geometry(geom, reset_bounding_box=False)
        current_geom[0] = geom

def next_chunk(vis):
    refresh(idx[0] + 1)
    return False

def prev_chunk(vis):
    refresh(idx[0] - 1)
    return False

def reset_cam(vis):
    vis.reset_view_point(True)
    return False

def save_transparent(vis):
    img_float = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    img_u8    = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    bg_u8     = (BG_COLOR * 255).astype(np.uint8)
    diff      = np.abs(img_u8.astype(np.int16) - bg_u8.astype(np.int16))
    bg_mask   = np.all(diff <= BG_THRESHOLD, axis=2)
    alpha     = np.where(bg_mask, 0, 255).astype(np.uint8)
    rgba      = np.dstack([img_u8, alpha])
    stem      = os.path.splitext(os.path.basename(files[idx[0]]))[0]
    out       = os.path.join(SAVE_DIR, f"{stem}_render.png")
    Image.fromarray(rgba).save(out, "PNG")
    print(f"  Saved → {out}")
    return False

GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT  = 263

vis.register_key_callback(GLFW_KEY_RIGHT, next_chunk)
vis.register_key_callback(GLFW_KEY_LEFT,  prev_chunk)
vis.register_key_callback(ord("N"), next_chunk)
vis.register_key_callback(ord("P"), prev_chunk)
vis.register_key_callback(ord("R"), reset_cam)
vis.register_key_callback(ord("S"), save_transparent)

# ── initial scene ─────────────────────────────────────────────────────────
if args.show_all:
    for g in all_geoms:
        vis.add_geometry(g, reset_bounding_box=False)
    vis.reset_view_point(True)
    # highlight first chunk
    current_geom[0] = None
    refresh(0)
else:
    g = read_geom(files[0])
    g.paint_uniform_color(HIGHLIGHT)
    vis.add_geometry(g, reset_bounding_box=True)
    current_geom[0] = g
    print_label(0)

vis.run()
vis.destroy_window()
