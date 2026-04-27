import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("outputs/nksr_reconstruction.ply")

# Find boundary edges — edges belonging to only one triangle
triangles = np.asarray(mesh.triangles)
edges = {}
for tri in triangles:
    for i in range(3):
        e = tuple(sorted([tri[i], tri[(i+1)%3]]))
        edges[e] = edges.get(e, 0) + 1

# Boundary vertices — vertices on edges with only one triangle
boundary_verts = set()
for (v0, v1), count in edges.items():
    if count == 1:
        boundary_verts.add(v0)
        boundary_verts.add(v1)

print(f"Boundary vertices: {len(boundary_verts):,}")
print(f"Total vertices: {len(mesh.vertices):,}")
print(f"Boundary ratio: {len(boundary_verts)/len(mesh.vertices)*100:.1f}%")