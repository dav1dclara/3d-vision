import rerun as rr
import numpy as np
import open3d as o3d

rr.init("mesh_viewer", spawn=True)

nksr = o3d.io.read_triangle_mesh("outputs/nksr_reconstruction.ply")
nksr.compute_vertex_normals()

nksr_vertices = np.asarray(nksr.vertices)
nksr_faces    = np.asarray(nksr.triangles)
nksr_normals  = np.asarray(nksr.vertex_normals)

poisson = o3d.io.read_triangle_mesh("outputs/possoin_reconstruction.ply")
poisson.compute_vertex_normals()

poisson_vertices = np.asarray(poisson.vertices)
poisson_faces    = np.asarray(poisson.triangles)
poisson_normals  = np.asarray(poisson.vertex_normals)

pcd = o3d.io.read_triangle_mesh("outputs/pcd_reconstruction.ply")
pcd.compute_vertex_normals()

pcd_vertices = np.asarray(pcd.vertices)
pcd_faces    = np.asarray(pcd.triangles)
pcd_normals  = np.asarray(pcd.vertex_normals)

rr.log("nksr", rr.Mesh3D(
    vertex_positions=nksr_vertices,
    triangle_indices=nksr_faces,
    vertex_normals=nksr_normals,
))

rr.log("pcd", rr.Mesh3D(
    vertex_positions=pcd_vertices,
    triangle_indices=pcd_faces,
    vertex_normals=pcd_normals,
))

rr.log("poisson", rr.Mesh3D(
    vertex_positions=poisson_vertices,
    triangle_indices=poisson_faces,
    vertex_normals=poisson_normals,
))

input("Press Enter to exit...")