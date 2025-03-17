"""
File: point_ops.py
Date: 2025/3/9
Author: yruns

Description: This file contains ...
"""
import numpy as np
import open3d as o3d


def augment_point_cloud(points, scale_range=(0.8, 1.2), rotation_range=(0, 2 * np.pi)):
    """
    对点云进行随机缩放和旋转的增强。

    参数:
        points: ndarray of shape (N, 3), 点云坐标。
        scale_range: tuple (min_scale, max_scale), 缩放比例的范围。
        rotation_range: tuple (min_angle, max_angle), 旋转角度范围 (弧度制)。

    返回:
        增强后的点云，shape为(N, 3)。
    """
    # 随机缩放
    scale = np.random.uniform(scale_range[0], scale_range[1])
    scaled_points = points * scale

    # 随机旋转（绕Z轴）
    theta = np.random.uniform(rotation_range[0], rotation_range[1])

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    rotated_points = scaled_points.dot(rotation_matrix)

    return rotated_points


def add_nonuniform_noise(points, center, radius, scale_factor=0.02, decay="linear"):
    # 计算点到中心的距离
    distances = np.linalg.norm(points - center, axis=1)

    # 找到受影响的点
    region_indices = np.where(distances < radius)[0]
    if len(region_indices) == 0:
        return points, region_indices  # 无点受影响

    # 计算点云整体尺度，使噪声标准差自适应
    point_cloud_scale = points.max() - points.min()
    max_std = scale_factor * point_cloud_scale  # 使噪声强度随点云大小变化

    # 计算噪声衰减系数
    normalized_distances = distances[region_indices] / radius  # 归一化 [0,1]

    if decay == "linear":
        noise_scale = 1 - normalized_distances  # 线性衰减
    elif decay == "cosine":
        noise_scale = np.cos(normalized_distances * np.pi / 2)  # 余弦衰减
    elif decay == "exponential":
        noise_scale = np.exp(-4 * normalized_distances)  # 指数衰减
    else:
        raise ValueError("Unsupported decay type. Use 'linear', 'cosine', or 'exponential'.")

    # 生成非均匀噪声
    noise = np.random.normal(0, max_std, size=(len(region_indices), 3)) * noise_scale[:, np.newaxis]

    # 施加噪声
    points[region_indices] += noise

    return points, region_indices


def add_structured_anomaly(points, center, radius, scale_factor=0.02, outward=True):
    # 计算所有点到中心点的距离
    distances = np.linalg.norm(points - center, axis=1)

    # 找出位于影响范围内的点
    region_indices = np.where(distances < radius)[0]

    if len(region_indices) == 0:
        return points, region_indices  # 没有点受影响

    # 计算点云整体尺寸，决定 magnitude 的大小
    point_cloud_scale = points.max() - points.min()
    magnitude = scale_factor * point_cloud_scale  # 使异常变形随点云大小变化

    # 计算每个点的径向方向向量
    directions = points[region_indices] - center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6  # 归一化

    # 选择向外 (outward=True) 还是向内 (outward=False)
    if not outward:
        directions = -directions

    # 施加异常位移
    points[region_indices] += directions * magnitude

    return points, region_indices


def add_shape_anomaly_with_normals(points, center, radius, scale_factor=0.02, outward=True):
    # 将点云转换为Open3D对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.normalize_normals()

    normals = np.asarray(pcd.normals)

    # 计算所有点到中心点的距离
    distances = np.linalg.norm(points - center, axis=1)

    # 找到落入异常区域的点
    region_indices = np.where(distances < radius)[0]

    if len(region_indices) == 0:
        return points, region_indices  # 没有点受到影响

    # 根据点云整体尺寸动态调整变形幅度
    point_cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    deform_strength = scale_factor * point_cloud_scale

    # 计算变形幅度（在 radius 内逐渐衰减）
    region_distances = distances[region_indices]
    deform_scale = deform_strength * np.exp(-(region_distances ** 2) / (2 * (radius / 2) ** 2))

    # 调整变形方向
    direction = normals[region_indices]
    if not outward:
        direction = -direction

    # 计算最终位移
    displacement = direction * deform_scale[:, np.newaxis]

    # 施加变形
    points[region_indices] += displacement

    return points, region_indices


def simulate_realistic_industrial_anomaly(points, max_num_region=8, noise_radius_range=(0.05, 0.10)):
    anomaly_types = ['noise', 'structure', 'shape']

    num_regions = np.random.randint(1, max_num_region + 1)
    # 选择num_regions个中心点
    selected_indices = np.random.choice(len(points), num_regions, replace=False)
    selected_centers = points[selected_indices]

    # 生成num_regions个半径
    max_ = points.max()
    min_ = points.min()
    scale = max_ - min_
    selected_radii = np.random.uniform(noise_radius_range[0] * scale, noise_radius_range[1] * scale, num_regions)

    modified_indices = np.zeros(len(points), dtype=np.int32)
    for idx, center in enumerate(selected_centers):
        anomaly_type = np.random.choice(anomaly_types)
        radii = selected_radii[idx]

        if anomaly_type == 'noise':
            points, region_indices = add_nonuniform_noise(points, center, radii)
        elif anomaly_type == 'structure':
            points, region_indices = add_shape_anomaly_with_normals(points, center, radii, outward=False)
        else:
            points, region_indices = add_shape_anomaly_with_normals(points, center, radii)
        modified_indices[region_indices] = idx + 1

    return points, modified_indices


def voxel_downsample_with_anomalies(points, modified_mask=None, voxel_size=0.5):
    if modified_mask is None:
        modified_mask = np.zeros(len(points))

    # 计算每个点所属的voxel坐标
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # 获取唯一的voxel坐标及其索引
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)

    # 计算每个voxel的中心点坐标（所有点的平均值）
    voxel_points = np.zeros((unique_voxels.shape[0], 3))
    np.add.at(voxel_points, inverse_indices, points)
    counts = np.bincount(inverse_indices)
    voxel_points /= counts[:, np.newaxis]

    # 计算每个点到其对应voxel中心的欧几里得距离
    distances = np.linalg.norm(points - voxel_points[inverse_indices], axis=1)

    # 获取每个voxel的代表点索引（距离voxel中心最近的点）
    voxel_representative_indices = np.zeros(len(unique_voxels), dtype=int)
    unique_inverse_argmin = np.argsort(inverse_indices, kind='stable')
    sorted_inverse_indices = inverse_indices[unique_inverse_argmin]

    first_occurrences = np.concatenate(([True], sorted_inverse_indices[1:] != sorted_inverse_indices[:-1]))
    voxel_representative_indices[sorted_inverse_indices[first_occurrences]] = unique_inverse_argmin[first_occurrences]

    # 确定每个voxel是否含有异常点（modified_mask中不为0）
    voxel_labels = np.zeros(len(unique_voxels), dtype=int)
    np.maximum.at(voxel_labels, inverse_indices, modified_mask.astype(int))

    return voxel_points, voxel_representative_indices, voxel_labels
