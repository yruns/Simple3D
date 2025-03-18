"""
File: real3d.py
Date: 2025/3/18
Author: yruns

Description: This file contains ...
"""
from datasets.real3d import Real3DDataset

import numpy as np
import open3d as o3d

def main():
    dataset = Real3DDataset("/Users/yruns/Downloads/Real3D-AD-PCD", "airplane", split="test")

    for idx in range(len(dataset)):
        item = dataset[idx]
        pointcloud, mask, label, sample_path = item
        if label == 0: continue
        if sample_path != '/Users/yruns/Downloads/Real3D-AD-PCD/airplane/test/482_bulge_cut.pcd': continue

        print("Path:", sample_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        colors = np.zeros((len(pointcloud), 3))
        colors[mask == 1] = [1, 0, 0]
        colors[mask == 0] = [0, 1, 0]

        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
