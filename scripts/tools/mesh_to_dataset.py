#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2023 Ignacio Vizzo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# coding: utf-8
import os

import argh
import mesh_to_sdf
import numpy as np
import open3d as o3d
import trimesh


# Original bunny resolution is: 512x400 = 204800 points
def mesh_to_tsdf(filename, scan_count=6, scan_resolution=2048):
    scans = mesh_to_sdf.get_surface_point_cloud(
        trimesh.load(filename),
        scan_count=scan_count,
        scan_resolution=scan_resolution,
    ).scans

    os.makedirs("results", exist_ok=True)
    poses = []
    for idx, scan in enumerate(scans):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan.points))
        print(cloud)
        poses.append(scan.camera_transform)
        o3d.io.write_point_cloud(f"results/{str(idx).zfill(6)}.ply", cloud)
    poses = np.asarray(poses).reshape(scan_count, -1)
    np.savetxt("results/poses.txt", poses)


if __name__ == "__main__":
    argh.dispatch_command(mesh_to_tsdf)
