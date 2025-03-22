"""
File: analyze.py
Date: 2025/3/18
Author: yruns

Description: This file contains ...
"""
import torch
import numpy as np
import pickle
import os
import open3d as o3d
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import trimesh


PATH = "./assets/eval/0_best.pkl"

def compute_metrics(pred, gt):
    precision, recall, thresholds_pr = precision_recall_curve(gt, pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx_f1 = np.argmax(f1_scores)
    best_threshold_f1 = thresholds_pr[best_idx_f1]
    pred = (pred >= best_threshold_f1).astype(np.int32)
    return pred

def visualize_with_title(pcd, window_title="Point Cloud Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)  # Set the window title here
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def save_pcd(pcd, path):
    # Ensure that the directory exists before saving
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    o3d.io.write_point_cloud(path, pcd)
    obj_file = path.replace(".ply", ".obj")

    mesh = trimesh.load_mesh(path)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")
    mesh.export(obj_file)

    # remove the ply file
    os.remove(path)

def main():
    result_dict = pickle.load(open(PATH, 'rb'))
    print(result_dict.keys())

    metrics = result_dict['metrics']
    print(metrics)
    datas = result_dict['data']

    count = 0
    for data in datas:
        pred = data['pred']
        gt = data['mask'].astype(np.int64)
        full_gt = data['mask_full'].astype(np.int64)
        ori_path = data['path']
        downsampled_points = data['pointcloud']
        threshold = metrics['threshold']

        ori_path = ori_path.replace("./data", "/Users/yruns/Downloads/Real3D-AD-PCD")
        if 'good' in ori_path:
            continue

        pcd = o3d.io.read_point_cloud(ori_path)
        points = np.array(pcd.points)
        colors = np.zeros((len(points), 3))
        colors[full_gt == 1] = [1, 0, 0]
        colors[full_gt == 0] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])

        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        colors = np.zeros((len(downsampled_points), 3))
        colors[gt == 1] = [1, 0, 0]
        colors[gt == 0] = [0, 1, 0]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([downsampled_pcd])
        # visualize_with_title(downsampled_pcd, "Downsampled GT")
        save_pcd(downsampled_pcd, f"assets/pcds/{count}_downsampled_gt.ply")

        preds = (pred >= threshold).astype(np.int32)
        true_index = (gt == 0)
        fake_index = (gt == 1)
        p_true = (preds[true_index] == 0).mean()
        p_fake = (preds[fake_index] == 1).mean()
        print(f"p_true: {p_true}, p_fake: {p_fake}")

        colors = np.zeros((len(downsampled_points), 3))
        colors[preds == 1] = [1, 0, 0]
        colors[preds == 0] = [0, 1, 0]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(colors)
        # visualize_with_title(downsampled_pcd, "Downsampled Pred")
        save_pcd(downsampled_pcd, f"assets/pcds/{count}_downsampled_pred.ply")
        count += 1

        if count > 8:
            break


if __name__ == '__main__':
    main()
