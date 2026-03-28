import numpy as np
import laspy
import open3d as o3d
import torch
import fvdb

las = laspy.read("/work/scratch/oscipal/2026-03-09_16.19.44/pointcloud.las")
xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
centroid = xyz.mean(axis=0)
xyz -= centroid
xyz_t = torch.from_numpy(xyz).float().cuda()

VOXEL_SIZE   = 0.1
CHUNK_VOXELS = 80
OVERLAP      = 32

grid = fvdb.GridBatch.from_points(
    fvdb.JaggedTensor([xyz_t]),
    voxel_sizes=[VOXEL_SIZE] * 3,
    origins=[0.0, 0.0, 0.0],
)
bbox    = grid.bbox_at(0).cpu().numpy()
ijk_min = bbox[0].astype(int)
ijk_max = bbox[1].astype(int)

def compute_planarity(pts_np):
    centered = pts_np - pts_np.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return s[2] / (s[0] + 1e-8)

for ix in range(int(ijk_min[0]), int(ijk_max[0]), CHUNK_VOXELS):
    for iy in range(int(ijk_min[1]), int(ijk_max[1]), CHUNK_VOXELS):
        for iz in range(int(ijk_min[2]), int(ijk_max[2]), CHUNK_VOXELS):
            cmin = [ix - OVERLAP, iy - OVERLAP, iz - OVERLAP]
            cmax = [ix + CHUNK_VOXELS + OVERLAP,
                    iy + CHUNK_VOXELS + OVERLAP,
                    iz + CHUNK_VOXELS + OVERLAP]

            wmin = torch.tensor(cmin, device="cuda").float() * VOXEL_SIZE
            wmax = torch.tensor(cmax, device="cuda").float() * VOXEL_SIZE
            mask = ((xyz_t >= wmin) & (xyz_t <= wmax)).all(dim=1)
            chunk_pts = xyz_t[mask]

            if chunk_pts.shape[0] < 2000:
                continue

            pts_np    = chunk_pts.cpu().numpy()
            planarity = compute_planarity(pts_np)
            print(f"Chunk [{ix:4d},{iy:4d},{iz:4d}]: "
                  f"{chunk_pts.shape[0]:>8,} pts | planarity={planarity:.4f}")