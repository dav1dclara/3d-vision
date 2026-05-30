"""Render a point cloud or mesh to a PNG with a transparent background.

Accepts point clouds (.ply/.pcd) and triangle meshes (.ply) as --input.
Overlays (voxel grids, chunk boundaries, extra meshes) can be added with --overlay.

Usage:
    # Offscreen render → transparent PNG
    python scripts/viewer.py --input cloud.ply

    # Interactive: position camera, press S to save
    python scripts/viewer.py --input cloud.ply --interactive

    # Poster quality, white geometry, 4K
    python scripts/viewer.py --input cloud.ply --color 1 1 1 --bg-color 0 0 0 \
        --width 3840 --height 2160 --point-size 3 --voxel 0.05

    # Overlay voxel grid in amber, keep camera consistent across renders
    python scripts/viewer.py --input cloud.ply \
        --overlay outputs/voxel_grid.ply outputs/chunk_boundaries.ply \
        --overlay-color 0.9 0.7 0.2 \
        --interactive --save-camera camera.json

    python scripts/viewer.py --input cloud.ply --load-camera camera.json

Flags:
    --input FILE            Point cloud or mesh to render (required)
    --output FILE           Output PNG path (default: <input>_render.png)
    --color R G B           Geometry colour 0-1 (default: 0.85 0.85 0.85)
    --voxel FLOAT           Voxel downsample in metres, point clouds only
    --point-size FLOAT      Rendered point size in pixels (default: 2.0)
    --width INT             Image width  (default: 1920)
    --height INT            Image height (default: 1080)
    --bg-color R G B        Background colour for keying (default: 0 0 0 black)
    --bg-threshold INT      Per-channel key tolerance (default: 8)
    --overlay FILE [FILE …] Extra PLY files drawn on top (LineSets, meshes, clouds)
    --overlay-color R G B   Paint all overlays this colour instead of their own
    --interactive           Open viewer window; press S to capture, Q to quit
    --save-camera FILE      Save camera to JSON when pressing S
    --load-camera FILE      Restore camera from a previously saved JSON
"""

import argparse
import numpy as np
import open3d as o3d
from PIL import Image

parser = argparse.ArgumentParser(description="Render point cloud → transparent PNG")
parser.add_argument("--input", required=True, help="Path to PLY/LAS point cloud")
parser.add_argument("--output", default=None,
                    help="Output PNG path (default: <input_stem>_render.png)")
parser.add_argument("--color", nargs=3, type=float, default=[0.85, 0.85, 0.85],
                    metavar=("R", "G", "B"),
                    help="Point color in 0-1 range (default: light grey)")
parser.add_argument("--voxel", type=float, default=None,
                    help="Voxel downsample size in metres (default: no downsampling)")
parser.add_argument("--point-size", type=float, default=2.0,
                    help="Rendered point size in pixels (default: 2.0)")
parser.add_argument("--width", type=int, default=1920, help="Image width  (default: 1920)")
parser.add_argument("--height", type=int, default=1080, help="Image height (default: 1080)")
parser.add_argument("--bg-color", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("R", "G", "B"),
                    help="Background colour used for keying (default: black). "
                         "Must contrast with --color.")
parser.add_argument("--bg-threshold", type=int, default=8,
                    help="Per-channel tolerance for background keying (default: 8)")
parser.add_argument("--interactive", action="store_true",
                    help="Open interactive viewer — position camera, then press S to save PNG")
parser.add_argument("--save-camera", default=None, metavar="FILE",
                    help="Save camera parameters to JSON when pressing S (e.g. camera.json)")
parser.add_argument("--load-camera", default=None, metavar="FILE",
                    help="Load camera parameters from a previously saved JSON file")
parser.add_argument("--overlay", nargs="+", default=[],
                    metavar="FILE",
                    help="Additional PLY files to display on top (e.g. voxel_grid.ply)")
parser.add_argument("--overlay-color", nargs=3, type=float, default=None,
                    metavar=("R", "G", "B"),
                    help="Paint all overlays this uniform color, ignoring their original colors "
                         "(default: keep original). Suggested: 0.9 0.7 0.2 for amber")
args = parser.parse_args()

# ── derive output path ────────────────────────────────────────────────────
if args.output is None:
    from pathlib import Path
    args.output = str(Path(args.input).with_suffix("")) + "_render.png"

# ── load ──────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
mesh = o3d.io.read_triangle_mesh(args.input)
if len(mesh.triangles) > 0:
    print(f"  TriangleMesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    mesh.compute_vertex_normals()
    if args.voxel:
        print("  (--voxel ignored for meshes)")
    mesh.paint_uniform_color(args.color)
    pcd = mesh
else:
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"  PointCloud: {len(pcd.points):,} points")
    if args.voxel:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"  Downsampled to {len(pcd.points):,} points (voxel={args.voxel}m)")
    pcd.paint_uniform_color(args.color)

# ── overlays ──────────────────────────────────────────────────────────────
overlays = []
for path in args.overlay:
    print(f"Loading overlay {path} ...")
    ls = o3d.io.read_line_set(path)
    if len(ls.lines) > 0:
        if args.overlay_color:
            ls.paint_uniform_color(args.overlay_color)
        overlays.append(ls)
        print(f"  LineSet: {len(ls.lines):,} lines")
    else:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            if args.overlay_color:
                mesh.paint_uniform_color(args.overlay_color)
            overlays.append(mesh)
            print(f"  TriangleMesh: {len(mesh.triangles):,} faces")
        else:
            pcd2 = o3d.io.read_point_cloud(path)
            if args.overlay_color:
                pcd2.paint_uniform_color(args.overlay_color)
            overlays.append(pcd2)
            print(f"  PointCloud: {len(pcd2.points):,} points")

# ── interactive mode ──────────────────────────────────────────────────────
def key_out_and_save(img_float):
    img_u8 = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    bg_u8  = (np.array(args.bg_color) * 255).astype(np.uint8)
    diff   = np.abs(img_u8.astype(np.int16) - bg_u8.astype(np.int16))
    bg_mask = np.all(diff <= args.bg_threshold, axis=2)
    alpha   = np.where(bg_mask, 0, 255).astype(np.uint8)
    rgba    = np.dstack([img_u8, alpha])
    Image.fromarray(rgba).save(args.output, "PNG")
    print(f"Saved → {args.output}  ({rgba.shape[1]}×{rgba.shape[0]}, "
          f"{bg_mask.sum():,} transparent pixels)")

if args.interactive:
    print("Interactive viewer — use mouse to position camera.")
    print("  Press S  : capture current view → transparent PNG")
    print("  Press Q/Esc : quit without saving")

    saved = [False]

    def on_key(vis, action, mods):
        if action == 1:  # key down
            return False
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Position camera — press S to save",
                      width=args.width, height=args.height)
    vis.add_geometry(pcd)
    for ov in overlays:
        vis.add_geometry(ov)
    opt = vis.get_render_option()
    opt.background_color = np.array(args.bg_color)
    opt.point_size = args.point_size
    vis.reset_view_point(True)

    if args.load_camera:
        params = o3d.io.read_pinhole_camera_parameters(args.load_camera)
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)
        print(f"Camera loaded from {args.load_camera}")

    def save_callback(vis):
        if args.save_camera:
            params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(args.save_camera, params)
            print(f"Camera saved → {args.save_camera}")
        img_float = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        key_out_and_save(img_float)
        saved[0] = True
        return False

    vis.register_key_callback(ord("S"), save_callback)
    vis.run()
    vis.destroy_window()
    if not saved[0]:
        print("Viewer closed without saving.")
    import sys; sys.exit(0)

# ── offscreen render ──────────────────────────────────────────────────────
print("Rendering...")
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=args.width, height=args.height)
vis.add_geometry(pcd)
for ov in overlays:
    vis.add_geometry(ov)

opt = vis.get_render_option()
opt.background_color = np.array(args.bg_color)
opt.point_size = args.point_size

if args.load_camera:
    params = o3d.io.read_pinhole_camera_parameters(args.load_camera)
    vis.get_view_control().convert_from_pinhole_camera_parameters(params)
    print(f"Camera loaded from {args.load_camera}")
else:
    vis.reset_view_point(True)
vis.poll_events()
vis.update_renderer()

img_float = np.asarray(vis.capture_screen_float_buffer(do_render=True))
vis.destroy_window()

# ── key out background → alpha ────────────────────────────────────────────
key_out_and_save(img_float)
