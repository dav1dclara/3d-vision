"""First-person walkthrough of a large scene with chunk streaming.

Input is either a single large mesh (--mesh, split into an XY grid in memory on
load) or a directory of pre-split chunk PLYs (--chunks-dir). Only chunks within
--load-radius of the camera are kept in the renderer, so the full scene never
needs to fit in memory. The ground level is auto-detected (5th-percentile Z) and
the camera floats --height metres above it; nudge the ground with +/-.

Rendering uses a hidden Open3D window (WGL, works on Windows) blitted into a
pygame window for input. Click-to-move raycasts against the loaded geometry.

Controls:
    Mouse         : look around
    W / S         : walk forward / back
    A / D         : strafe left / right
    Left click    : move to the clicked point (glides there)
    + / -         : raise / lower ground level by 0.1 m
    Esc           : release mouse (click to re-capture) / quit when released

Usage:
    python scripts/visualization/walk.py --mesh ../meshes/scene.ply
    python scripts/visualization/walk.py --mesh scene.ply --cell-size 5 --load-radius 12
    python scripts/visualization/walk.py --mesh scene.ply --color          # original vertex colors
    python scripts/visualization/walk.py --chunks-dir outputs/chunks

Flags:
    --mesh FILE        Single large PLY, split into --cell-size m XY cells (xor --chunks-dir)
    --chunks-dir DIR   Directory of pre-split chunk PLYs                  (xor --mesh)
    --pattern GLOB     Chunk filter for --chunks-dir (default: *.ply)
    --cell-size FLOAT  XY grid cell size in m when splitting --mesh (default: 5)
    --load-radius FLOAT    Load chunks within this distance (m, default: 10)
    --unload-radius FLOAT  Unload chunks beyond this distance (m, default: 16)
    --height FLOAT     Eye height above ground (m, default: 1.8)
    --width / --win-height INT   Window size (default: 1280 x 720)
    --fov FLOAT        Field of view in degrees (default: 70)
    --move-speed FLOAT Walk speed m/s (default: 3)
    --mouse-sens FLOAT Mouse look sensitivity (default: 0.0015)
    --color            Show original vertex colors instead of uniform grey
"""

import argparse
import glob
import os
import sys
import numpy as np
import pygame
import open3d as o3d

# ── args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--mesh",
    metavar="FILE",
    help="Single large PLY — split into a spatial grid on load",
)
group.add_argument(
    "--chunks-dir", metavar="DIR", help="Directory of pre-split chunk PLY files"
)
parser.add_argument("--pattern", default="*.ply")
parser.add_argument(
    "--cell-size",
    type=float,
    default=5.0,
    help="XY grid cell size in metres when using --mesh (default 5)",
)
parser.add_argument("--load-radius", type=float, default=10.0)
parser.add_argument("--unload-radius", type=float, default=16.0)
parser.add_argument("--height", type=float, default=1.8)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--win-height", type=int, default=720)
parser.add_argument("--fov", type=float, default=70.0)
parser.add_argument("--move-speed", type=float, default=3.0)
parser.add_argument("--mouse-sens", type=float, default=0.0015)
parser.add_argument(
    "--color",
    action="store_true",
    help="Show original vertex colors instead of uniform grey",
)
args = parser.parse_args()

W, H = args.width, args.win_height


# ── build chunk index ─────────────────────────────────────────────────────
def split_mesh_to_cells(mesh, cell_size):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    has_vc = mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors) if has_vc else None

    face_xy = verts[faces].mean(axis=1)[:, :2]
    cell_keys = np.floor(face_xy / cell_size).astype(int)

    cells = []
    for key in {tuple(k) for k in cell_keys}:
        mask = np.all(cell_keys == np.array(key), axis=1)
        sub_faces = faces[mask]
        used = np.unique(sub_faces)
        remap = np.full(len(verts), -1, dtype=int)
        remap[used] = np.arange(len(used))

        sub = o3d.geometry.TriangleMesh()
        sub.vertices = o3d.utility.Vector3dVector(verts[used])
        sub.triangles = o3d.utility.Vector3iVector(remap[sub_faces])
        if has_vc:
            sub.vertex_colors = o3d.utility.Vector3dVector(colors[used])
        sub.compute_vertex_normals()
        if not (args.color and has_vc):
            sub.paint_uniform_color([0.82, 0.82, 0.82])

        cells.append(
            {
                "key": f"cell_{key[0]}_{key[1]}",
                "centroid": verts[used].mean(axis=0),
                "source": sub,
            }
        )
    return cells


chunk_index = []
sample_z = []

if args.mesh:
    print(f"Loading {args.mesh} …")
    full = o3d.io.read_triangle_mesh(args.mesh)
    print(
        f"  {len(full.vertices):,} vertices — splitting into {args.cell_size} m cells…"
    )
    chunk_index = split_mesh_to_cells(full, args.cell_size)
    del full
    print(f"  {len(chunk_index)} cells.")
    for c in chunk_index:
        v = np.asarray(c["source"].vertices)
        sample_z.extend(v[::50, 2].tolist())
else:
    files = sorted(glob.glob(os.path.join(args.chunks_dir, args.pattern)))
    if not files:
        print(f"No chunks found in {args.chunks_dir}")
        sys.exit(1)
    print(f"Indexing {len(files)} chunk files…")
    for path in files:
        m = o3d.io.read_triangle_mesh(path)
        if len(m.vertices) == 0:
            continue
        v = np.asarray(m.vertices)
        chunk_index.append(
            {
                "key": os.path.basename(path),
                "centroid": v.mean(axis=0),
                "source": path,
            }
        )
        sample_z.extend(v[::50, 2].tolist())
        del m
    print(f"  {len(chunk_index)} chunks indexed.")

# ── auto-detect ground ────────────────────────────────────────────────────
ground_z = float(np.percentile(sample_z, 5))
ground_normal = np.array([0.0, 0.0, 1.0])
print(f"  Auto ground Z = {ground_z:.3f} m  (nudge with + / -)")

# ── Open3D Visualizer (visible=False uses WGL on Windows, no EGL needed) ─
vis3d = o3d.visualization.Visualizer()
vis3d.create_window(visible=False, width=W, height=H)

ropt = vis3d.get_render_option()
ropt.background_color = np.array([0.07, 0.07, 0.09])
ropt.light_on = True
ropt.mesh_show_back_face = True

# ── chunk streaming ───────────────────────────────────────────────────────
loaded = {}  # key → TriangleMesh (needed to call remove_geometry)
raycast_mesh = None
scene_has_geom = [False]


def get_mesh(chunk):
    src = chunk["source"]
    if isinstance(src, str):
        m = o3d.io.read_triangle_mesh(src)
        m.compute_vertex_normals()
        if not (args.color and m.has_vertex_colors()):
            m.paint_uniform_color([0.82, 0.82, 0.82])
        return m
    return src


def update_chunks(eye):
    changed = False
    for c in chunk_index:
        d = np.linalg.norm(c["centroid"] - eye)
        key = c["key"]
        if d < args.load_radius and key not in loaded:
            m = get_mesh(c)
            if len(m.triangles) > 0:
                vis3d.add_geometry(m, reset_bounding_box=not scene_has_geom[0])
                scene_has_geom[0] = True
                loaded[key] = m
                changed = True
        elif d > args.unload_radius and key in loaded:
            vis3d.remove_geometry(loaded[key], reset_bounding_box=False)
            del loaded[key]
            changed = True
    return changed


def rebuild_raycast():
    global raycast_mesh
    rc = o3d.t.geometry.RaycastingScene()
    for key, m in loaded.items():
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    raycast_mesh = rc


def cast_ray(eye, direction):
    if raycast_mesh is None or len(loaded) == 0:
        return None
    d = direction / np.linalg.norm(direction)
    rays = o3d.core.Tensor([[*eye, *d]], dtype=o3d.core.Dtype.Float32)
    t = raycast_mesh.cast_rays(rays)["t_hit"][0].item()
    return None if (t == float("inf") or t > 200.0) else eye + t * d


def pixel_to_ray(px, py, front, right, up):
    aspect = W / H
    half_tan = np.tan(np.radians(args.fov) / 2)
    nx = (2.0 * px / W - 1.0) * aspect * half_tan
    ny = (1.0 - 2.0 * py / H) * half_tan
    return front + nx * right + ny * up


# ── camera ────────────────────────────────────────────────────────────────
all_centroids = np.array([c["centroid"] for c in chunk_index])
scene_center = all_centroids.mean(axis=0)

pos = scene_center.copy()
pos[2] = ground_z
yaw = 0.0
pitch = 0.0


def get_eye():
    return pos + ground_normal * args.height


def get_vectors():
    front = np.array(
        [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]
    )
    right = np.cross(front, ground_normal)
    r_len = np.linalg.norm(right)
    right = right / r_len if r_len > 1e-6 else np.array([1.0, 0.0, 0.0])
    return front, right, ground_normal


def set_camera():
    if not scene_has_geom[0]:
        return
    eye = get_eye()
    front, right, up = get_vectors()

    # Build OpenCV-convention extrinsic (X=right, Y=down, Z=forward)
    f = front / np.linalg.norm(front)
    r = right / np.linalg.norm(right)
    d = -up  # Y axis points down in camera frame
    R = np.stack([r, d, f], axis=0)  # rows: right, down, forward
    t = -R @ eye

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    fov_rad = np.radians(args.fov)
    fx = fy = W / (2.0 * np.tan(fov_rad / 2.0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W / 2.0, H / 2.0)

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extrinsic
    vis3d.get_view_control().convert_from_pinhole_camera_parameters(
        params, allow_arbitrary=True
    )


def capture_frame():
    vis3d.poll_events()
    vis3d.update_renderer()
    img = np.asarray(vis3d.capture_screen_float_buffer(do_render=True))
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ── pygame ────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Hallway Walker")
font = pygame.font.SysFont("monospace", 15)
clock = pygame.time.Clock()

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
pygame.mouse.get_rel()
mouse_captured = True

update_chunks(get_eye())
rebuild_raycast()
set_camera()

move_target = None

# ── main loop ─────────────────────────────────────────────────────────────
running = True
while running:
    dt = min(clock.tick(60) / 1000.0, 0.1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if mouse_captured:
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    mouse_captured = False
                else:
                    running = False
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                ground_z += 0.1
                print(f"Ground Z → {ground_z:.2f} m")
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                ground_z -= 0.1
                print(f"Ground Z → {ground_z:.2f} m")

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not mouse_captured:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                pygame.mouse.get_rel()
                mouse_captured = True
            else:
                px, py = event.pos
                front, right, up = get_vectors()
                hit = cast_ray(get_eye(), pixel_to_ray(px, py, front, right, up))
                if hit is not None:
                    target = hit.copy()
                    target[2] = ground_z
                    move_target = target

    if mouse_captured:
        mx, my = pygame.mouse.get_rel()
        yaw += mx * args.mouse_sens
        pitch -= my * args.mouse_sens
        pitch = float(np.clip(pitch, -1.3, 1.3))

        front, right, _ = get_vectors()
        move = np.zeros(3)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            move += front
        if keys[pygame.K_s]:
            move -= front
        if keys[pygame.K_a]:
            move -= right
        if keys[pygame.K_d]:
            move += right

        if np.linalg.norm(move) > 1e-6:
            move -= np.dot(move, ground_normal) * ground_normal
            m_len = np.linalg.norm(move)
            if m_len > 1e-6:
                pos += move / m_len * args.move_speed * dt
            move_target = None
        elif move_target is not None:
            diff = move_target - pos
            diff[2] = 0.0
            dist = np.linalg.norm(diff)
            if dist < 0.15:
                move_target = None
            else:
                pos += diff / dist * min(args.move_speed * dt, dist)

        pos[2] = ground_z

        if update_chunks(get_eye()):
            rebuild_raycast()
        set_camera()

    # ── render ────────────────────────────────────────────────────────────
    img_np = capture_frame()
    screen.blit(pygame.surfarray.make_surface(img_np.transpose(1, 0, 2)), (0, 0))

    eye = get_eye()
    hud = [
        f"pos ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.2f} m)  "
        f"yaw {np.degrees(yaw):.0f}°  pitch {np.degrees(pitch):.0f}°",
        f"ground Z {ground_z:.2f} m  (+/- to adjust)   "
        f"chunks {len(loaded)}/{len(chunk_index)}",
        "W/S/A/D move   Mouse look   LClick go-to   Esc release/quit",
    ]
    for i, line in enumerate(hud):
        screen.blit(
            font.render(line, True, (220, 220, 60), (20, 20, 20)), (8, 6 + i * 18)
        )

    if not mouse_captured:
        msg = font.render("MOUSE RELEASED — click to re-capture", True, (255, 80, 80))
        screen.blit(msg, (W // 2 - msg.get_width() // 2, H - 30))

    pygame.display.flip()

vis3d.destroy_window()
pygame.quit()
