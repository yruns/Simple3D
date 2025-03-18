"""
File: visual_real3d.py
Date: 2025/3/11
Author: yruns

Description: This file contains ...
"""
from visualize.point_ops import simulate_realistic_industrial_anomaly, voxel_downsample_with_anomalies

import numpy as np
import open3d as o3d


def norm_pcd(point_cloud):
    center = np.average(point_cloud, axis=0)
    # print(center.shape)
    new_points = point_cloud - np.expand_dims(center, axis=0)
    return new_points

def main():
    pcd = o3d.io.read_point_cloud("/Users/yruns/Downloads/Real3D-AD-PCD/airplane/test/149_good.pcd")
    points = np.array(pcd.points)
    points = norm_pcd(points)

    points, modified_indices = simulate_realistic_industrial_anomaly(points, noise_radius_range=(0.05, 0.10))

    print("Modified indices:", (modified_indices == 0).sum() / modified_indices.shape[0])
    pcd.points = o3d.utility.Vector3dVector(points)

    # 按modifed_indices进行着色
    # 0: 无异常
    # >0: 异常类型
    # 假设 points 和 modified_indices 已处理完毕
    colors = np.zeros((len(points), 3))

    # 定义最多八种异常类型的颜色映射
    color_map = {
        0: [0.8, 0.8, 0.8],  # 无异常：浅灰色
        1: [1.0, 0.0, 0.0],  # 异常类型1：红色
        2: [0.0, 1.0, 0.0],  # 异常类型2：绿色
        3: [0.0, 0.0, 1.0],  # 异常类型3：蓝色
        4: [1.0, 1.0, 0.0],  # 异常类型4：黄色
        5: [1.0, 0.0, 1.0],  # 异常类型5：洋红色
        6: [0.0, 1.0, 1.0],  # 异常类型6：青色
        7: [1.0, 0.5, 0.0],  # 异常类型7：橙色
        8: [0.5, 0.0, 1.0],  # 异常类型8：紫色
    }

    # 根据modified_indices着色
    for idx, anomaly_type in enumerate(modified_indices):
        colors[idx] = color_map.get(anomaly_type, [0.0, 0.0, 0.0])  # 默认未知类型为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # voxel_points, voxel_representative_indices, voxel_labels = voxel_downsample_with_anomalies(points, modified_indices, voxel_size=1.0)
    #
    # pcd.colors = o3d.utility.Vector3dVector(colors[voxel_representative_indices])
    # pcd.points = o3d.utility.Vector3dVector(points[voxel_representative_indices])
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()