"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

import numpy as np
import timm
import torch
from sklearn import random_projection
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds

from M3DM.cpu_knn import fill_missing_values


class PC_Features(torch.nn.Module):

    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(device=self.device)
        self.deep_feature_extractor.to(self.device)
        self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)

        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        # self.gts = []
        # self.predictions = []
        self.image_rocauc = 0
        self.image_aupr = 0
        self.pixel_rocauc = 0
        self.pixel_aupr = 0
        self.au_pro = 0

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, origin_data, sample_idx, source_path, target_path):
        # print(type(patch)) <class 'torch.Tensor'>
        # print(patch.shape) torch.Size([784, 1536])
        # print(type(feature_map_dims)) <class 'torch.Size'>
        # print(feature_map_dims.shape)
        # print(type(mask))
        # print(mask.shape)
        # print(type(label))
        # print(label.shape)
        # print(len(self.patch_lib)) 1 & 1903
        # print(type(self.patch_lib[0])) <class 'torch.Tensor'>
        if (len(self.patch_lib) == 1):
            cur_patch_lib = self.patch_lib[0]
        else:
            cur_patch_lib = torch.cat(self.patch_lib)
        dist = torch.cdist(patch, cur_patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = cur_patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, cur_patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - cur_patch_lib[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # print(min_val.shape)torch.Size([3668])
        s_map = min_val
        # print("s_map shape")
        # print(s_map.shape) torch 718
        # segmentation map
        # s_map = min_val.view(1, 1, *feature_map_dims)
        # s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        # s_map = self.blur(s_map)

        # self.image_preds.append(s.cpu().numpy())
        # self.pixel_preds.extend(s_map.cpu().flatten().numpy())
        sample_data = origin_data[sample_idx]
        s_map = s_map.cpu().numpy()
        # print(sample_data.shape)
        # print(s_map.shape)
        # print(origin_data.shape)
        full_s_map = fill_missing_values(sample_data, s_map, origin_data, k=1)
        '''
        #save visualization and anomaly map numpy
        save_anomalymap(source_path,full_s_map,target_path)
        np.save(target_path.replace('.pcd','.npy'),full_s_map)
        '''
        self.image_preds.append(s.numpy())
        # self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_preds.extend(full_s_map.flatten())
        self.image_labels.append(label)
        self.pixel_labels.extend(mask.flatten().numpy())
        # self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        # self.gts.append(mask.detach().cpu().squeeze().numpy())

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.pixel_aupr = average_precision_score(self.pixel_labels, self.pixel_preds)
        self.image_aupr = average_precision_score(self.image_labels, self.image_preds)
        # print(len(self.gts)) 4
        # print(len(self.predictions)) 4
        # print(self.gts[1].shape) #366
        # print(self.predictions[1].shape) #366
        # self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)


class Model(torch.nn.Module):

    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(2, 3), checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        return features

    def freeze_parameters(self, layers, freeze_bn=False):
        """ Freeze resent parameters. The layers which are not indicated in the layers list are freeze. """

        layers = [str(layer) for layer in layers]
        # Freeze first block.
        if '1' not in layers:
            if hasattr(self.backbone, 'conv1'):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'bn1'):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'layer1'):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False

        # Freeze second block.
        if '2' not in layers:
            if hasattr(self.backbone, 'layer2'):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False

        # Freeze third block.
        if '3' not in layers:
            if hasattr(self.backbone, 'layer3'):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False

        # Freeze fourth block.
        if '4' not in layers:
            if hasattr(self.backbone, 'layer4'):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False

        # Freeze last FC layer.
        if '-1' not in layers:
            if hasattr(self.backbone, 'fc'):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False

        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
