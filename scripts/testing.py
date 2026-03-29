import os
import glob
import open3d as o3d
import numpy as np

# ── Decimate planar chunks only ───────────────────────────────────────────
print("Decimating planar chunks...")
planar_chunks  = sorted(glob.glob("outputs/chunks/planar_*.ply"))
complex_chunks = sorted(glob.glob("outputs/chunks/complex_*.ply"))
print(f"Planar chunks: {len(planar_chunks)}")
print(f"Complex chunks: {len(complex_chunks)}")

# Check total triangle counts
planar_total  = sum(len(o3d.io.read_triangle_mesh(c).triangles)
                    for c in planar_chunks)
complex_total = sum(len(o3d.io.read_triangle_mesh(c).triangles)
                    for c in complex_chunks)
print(f"Planar triangles:  {planar_total:,}")
print(f"Complex triangles: {complex_total:,}")

# Merge and decimate planar only
print("Merging planar chunks...")
planar_merged = o3d.geometry.TriangleMesh()
for path in planar_chunks:
    chunk = o3d.io.read_triangle_mesh(path)
    planar_merged += chunk
    del chunk

# Decimate planar to 20% of original — flat surfaces survive this well
target = max(100_000, len(planar_merged.triangles) // 5)
print(f"Decimating planar: {len(planar_merged.triangles):,} → {target:,}")
planar_dec = planar_merged.simplify_quadric_decimation(target)
del planar_merged

# Merge complex chunks at full resolution
print("Merging complex chunks...")
final = planar_dec
for path in complex_chunks:
    chunk = o3d.io.read_triangle_mesh(path)
    final += chunk
    del chunk

print(f"Final: {len(final.vertices):,} vertices, "
      f"{len(final.triangles):,} faces")

o3d.io.write_triangle_mesh("outputs/reconstruction_optimized.ply", final)
print("Saved → outputs/reconstruction_optimized.ply")