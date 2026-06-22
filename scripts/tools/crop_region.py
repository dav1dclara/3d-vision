"""Crop an axis-aligned box from a large PLY and save it for inspection.

Reads the file as a mesh first, falls back to a point cloud if it has no faces;
normals and colors are preserved. The crop is a cube of side 2*--radius centred
on --center (default: the geometry's centroid). This is non-destructive — the
source file is never modified. Use --info to just print the bounding box and
centroid without cropping (handy for picking valid --center coordinates).

Usage:
    # print bounding box / centroid, no crop
    python scripts/tools/crop_region.py --input scene.ply --output unused.ply --info

    # crop a 10 m box around the scene centroid
    python scripts/tools/crop_region.py --input scene.ply --output crop.ply

    # crop an 8 m box (radius 4) around a specific point
    python scripts/tools/crop_region.py --input scene.ply --output crop.ply \
        --center 123.4 456.7 463.5 --radius 4

Flags:
    --input FILE    Source mesh or point cloud (required)
    --output FILE   Output PLY (required; ignored with --info)
    --center X Y Z  Crop-box centre in world coords (default: centroid)
    --radius FLOAT  Half the box side length in metres (default: 10)
    --info          Print bounding box / centroid and exit without cropping
"""

import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--center",
    nargs=3,
    type=float,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Crop centre in world coords (default: scene centroid)",
)
parser.add_argument(
    "--radius",
    type=float,
    default=10.0,
    help="Half-size of the crop box in metres (default: 10)",
)
parser.add_argument(
    "--info",
    action="store_true",
    help="Just print bounding box and centroid, do not crop",
)
args = parser.parse_args()

print(f"Loading {args.input} …")
geom = o3d.io.read_triangle_mesh(args.input)
is_mesh = len(geom.triangles) > 0

if not is_mesh:
    print("  No triangles found — loading as point cloud.")
    geom = o3d.io.read_point_cloud(args.input)
    pts = np.asarray(geom.points)
else:
    pts = np.asarray(geom.vertices)

print(
    f"  {'Mesh' if is_mesh else 'PointCloud'}: {len(pts):,} {'vertices' if is_mesh else 'points'}"
)
print("  Bounding box:")
print(f"    min: {pts.min(axis=0)}")
print(f"    max: {pts.max(axis=0)}")
print(f"    centroid: {pts.mean(axis=0)}")

if args.info:
    raise SystemExit(0)

centre = np.array(args.center) if args.center else pts.mean(axis=0)
print(f"  Crop centre: {centre}  radius: {args.radius} m")

bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=centre - args.radius,
    max_bound=centre + args.radius,
)
cropped = geom.crop(bbox)

if is_mesh:
    n = len(np.asarray(cropped.vertices))
    print(
        f"  Cropped: {n:,} vertices, {len(np.asarray(cropped.triangles)):,} triangles"
    )
    o3d.io.write_triangle_mesh(args.output, cropped)
else:
    n = len(np.asarray(cropped.points))
    print(f"  Cropped: {n:,} points")
    o3d.io.write_point_cloud(args.output, cropped)

print(f"Saved → {args.output}")
print("\nOpen with:")
print(f"  python scripts/visualization/viewer.py --input {args.output} --interactive")
