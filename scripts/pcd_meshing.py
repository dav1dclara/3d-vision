import open3d as o3d
from pathlib import Path
from pcdmeshing import run_block_meshing
from datetime import datetime

start = datetime.now()

# Load your PLY (has xyz + normals + RGB)
print("Loading point cloud...")
pcd = o3d.io.read_point_cloud("/work/scratch/oscipal/2026-03-09_16.19.44/pointcloud.ply")
print(f"Loaded {len(pcd.points):,} points")

# Run reconstruction
print("Meshing...")
mesh, _ = run_block_meshing(
    pcd,
    voxel_size=20,          # 20m blocks — good for your 36x64m scene
    margin_seam=0.2,
    margin_discard=0.2,
    num_parallel=10,
    opts={
        "max_edge_length": 0.5,   # reject triangles with edges > 0.5m
        "max_visibility": 10,
    },
    use_visibility=False,
)

print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

# Save
out_path = "outputs/pcd_reconstruction.ply"
o3d.io.write_triangle_mesh(out_path, mesh)
print(f"Saved to {out_path}")

print(f"\nTotal time: {datetime.now() - start}")