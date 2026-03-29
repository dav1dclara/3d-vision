import rerun as rr
import numpy as np
import open3d as o3d

rr.init("mesh_viewer", spawn=True)

mesh = o3d.io.read_triangle_mesh("reconstruction.ply")
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
faces    = np.asarray(mesh.triangles)
normals  = np.asarray(mesh.vertex_normals)

rr.log("mesh", rr.Mesh3D(
    vertex_positions=vertices,
    triangle_indices=faces,
    vertex_normals=normals,
))

input("Press Enter to exit...")