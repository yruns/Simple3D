import os

import h5py
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'modelnet_c')


def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


class ModelNetC(Dataset):
    def __init__(self, split):
        h5_path = os.path.join(DATA_DIR, split + '.h5')
        self.data, self.label = load_h5(h5_path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label[0]

    def __len__(self):
        return self.data.shape[0]


class ModelNetCDataset(Dataset):
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(
            self,
            split='clean',
            transform=None
    ):
        h5_path = os.path.join(DATA_DIR, split + '.h5')
        self.data, self.label = load_h5(h5_path)
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item][0]

        data = {'pos': pointcloud, 'y': label}
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1


def eval_corrupt_wrapper(trainer, fn_test_corrupt):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param trainer: Trainer
    :param fn_test_corrupt: Original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :return:
    """
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    DGCNN_OA = {
        'clean': 0.926,
        'scale': 0.906,
        'jitter': 0.684,
        'rotate': 0.785,
        'dropout_global': 0.752,
        'dropout_local': 0.793,
        'add_global': 0.705,
        'add_local': 0.725
    }
    OA_clean = None
    perf_all = {'OA': [], 'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, trainer=trainer)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            print(test_perf)
            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break
        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)
        if corruption_type != 'clean':
            perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - DGCNN_OA[corruption_type])
            perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (DGCNN_OA['clean'] - DGCNN_OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        print(perf_corrupt)
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    print(perf_all)

    return perf_all
