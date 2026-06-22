import os
import yaml
import open3d as o3d
from pcdmeshing import run_block_meshing
from datetime import datetime

start = datetime.now()

with open("configs/pcd_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
meshing = cfg["meshing"]

out_path = os.path.join(paths["output_dir"], paths["output_mesh"])

# Load your PLY (has xyz + normals + RGB)
print("Loading point cloud...")
pcd = o3d.io.read_point_cloud(paths["pointcloud_ply"])
print(f"Loaded {len(pcd.points):,} points")

# Run reconstruction
print("Meshing...")
mesh, _ = run_block_meshing(
    pcd,
    voxel_size=meshing["voxel_size"],
    margin_seam=meshing["margin_seam"],
    margin_discard=meshing["margin_discard"],
    num_parallel=meshing["num_parallel"],
    opts={
        "max_edge_length": meshing["max_edge_length"],
        "max_visibility": meshing["max_visibility"],
    },
    use_visibility=meshing["use_visibility"],
)

print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

os.makedirs(paths["output_dir"], exist_ok=True)
o3d.io.write_triangle_mesh(out_path, mesh)
print(f"Saved to {out_path}")

print(f"\nTotal time: {datetime.now() - start}")
