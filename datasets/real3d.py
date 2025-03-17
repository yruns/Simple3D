"""
File: real3d.py
Date: 2025/3/17
Author: yruns

Description: This file contains dataset implementation for Real3D anomaly detection.
"""

import glob
import os
import pathlib

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

REAL3D_CLASS_NAME = [
    'airplane', 'car', 'candybar', 'chicken',
    'diamond', 'duck', 'fish', 'gemstone',
    'seahorse', 'shell', 'starfish', 'toffees'
]


class Real3DDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            cls_name,
            split='train',
            norm=True,
    ):
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.split = split
        self.norm = norm

        # Build sample list
        if self.split == 'train':
            self.sample_list = glob.glob(
                os.path.join(dataset_dir, cls_name, split, '*template*.pcd')
            )
        else:
            self.sample_list = glob.glob(
                os.path.join(dataset_dir, cls_name, split, '*.pcd')
            )
            self.sample_list = [s for s in self.sample_list if 'temp' not in s]
            self.gt_path = os.path.join(dataset_dir, cls_name, 'gt')

        self.sample_list.sort()

    @staticmethod
    def norm_pcd(point_cloud):
        center = np.average(point_cloud, axis=0)
        new_points = point_cloud - np.expand_dims(center, axis=0)
        return new_points

    def __getitem__(self, idx):
        sample_path = self.sample_list[idx]
        label = 0  # Default for normal samples

        if self.split == 'train':
            # Process training sample (all normal)
            pcd = o3d.io.read_point_cloud(sample_path)
            pointcloud = np.array(pcd.points)
            mask = np.zeros(pointcloud.shape[0])
        else:
            # Process testing sample
            if 'good' in sample_path:
                # Normal test sample
                pcd = o3d.io.read_point_cloud(sample_path)
                pointcloud = np.array(pcd.points)
                mask = np.zeros(pointcloud.shape[0])
            else:
                # Anomalous sample with ground truth
                filename = pathlib.Path(sample_path).stem
                txt_path = os.path.join(self.gt_path, filename + '.txt')
                pcd_data = np.genfromtxt(txt_path, delimiter=' ')   # TODO: preprocess to speed up
                pointcloud = pcd_data[:, :3]
                mask = pcd_data[:, 3].astype(np.float32)
                label = 1  # Anomalous label

        # Normalize point cloud
        if self.norm:
            pointcloud = self.norm_pcd(pointcloud)

        return pointcloud.astype(np.float32), mask, label, sample_path

    def __len__(self):
        return len(self.sample_list)
