import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from torch.utils.data import Dataset
from utils.augmentation import augmentation_pipeline

class IVUS_OCT_Dataset(Dataset):
    def __init__(self, data_root, subset, augmentation, config):

        self.data_root = data_root + subset
        self.augmentation = augmentation
        self.subset = subset
        self.config = config

        # load list of ivus and oct pkl file paths
        self.ivus_sb_path_list = sorted(glob.glob(data_root + subset + '/*/ivus_superglue_clusters.pkl'))
        self.oct_sb_path_list = sorted(glob.glob(data_root + subset + '/*/oct_superglue_clusters.pkl'))
        # load list of gt assignment file paths
        self.gt_list = sorted(glob.glob(data_root + subset + '/*/gt_superglue_clusters.npy'))

        assert len(self.ivus_sb_path_list) == len(self.oct_sb_path_list)

        print('{} set has {} images'.format(subset, len(self.ivus_sb_path_list)))

    def __len__(self):
        return len(self.ivus_sb_path_list)

    def load_pkl(self, path):
        with open(path, "rb") as f:
            pkl = pickle.load(f)
        return pkl

    def __getitem__(self, index):

        # load path names
        ivus_sb_path = self.ivus_sb_path_list[index]
        oct_sb_path = self.oct_sb_path_list[index]
        gt_assignment_path = self.gt_list[index]

        # load files
        ivus_sb = self.load_pkl(ivus_sb_path)
        oct_sb = self.load_pkl(oct_sb_path)
        gt_assignment = np.load(gt_assignment_path).astype(np.int64)
        gt_matching0, gt_matching1 = self.gt_assignment_2_gt_matches(gt_assignment)
        id_ = gt_assignment_path.split('/')[-2]

        # Each cluster is a k x f array where k is the number of sb in cluster and f is the number of features
        # during training we randomly sample points from each cluster to increase diversity 
        # the mean is taken of the sampled points to get a single sb representation (1 x f)
        # and stacked to (N x f) where N is the number of clusters in the modality

        # first we do this for IVUS
        ivus_sb_instances = []
        ivus_sb_instances_count = []
        for cluster_id, ivus_cluster in enumerate(ivus_sb):
            num_pts = np.array([ivus_cluster.shape[0]]) 
            if self.augmentation is not None:
                # random sample of points in cluster during training
                size = int(np.ceil(ivus_cluster.shape[0] * 0.7))
                sampled_idx = np.random.choice(list(range(ivus_cluster.shape[0])), size=size, replace=True)
                ivus_cluster = ivus_cluster[sampled_idx]
            elif (self.subset=='val' or self.subset=='test'):
                pass
            else:
                pass
            # average points to final sb representation
            ivus_cluster_mean = np.mean(ivus_cluster, axis=0)
            ivus_sb_instances.append(np.concatenate([ivus_cluster_mean, num_pts]))
        ivus_sb_instances = np.stack(ivus_sb_instances)

        # repeat for OCT
        oct_sb_instances = []
        oct_sb_instances_count = []
        for cluster_id, oct_cluster in enumerate(oct_sb):
            num_pts = np.array([oct_cluster.shape[0]]) #/ 10
            if self.augmentation is not None:
                size = int(np.ceil(oct_cluster.shape[0] * 0.7))
                sampled_idx = np.random.choice(list(range(oct_cluster.shape[0])), size=size, replace=True)
                oct_cluster = oct_cluster[sampled_idx]
            elif self.subset=='val' or self.subset=='test':
                pass
            else:
                pass
            oct_cluster_mean = np.mean(oct_cluster, axis=0)
            oct_sb_instances.append(np.concatenate([oct_cluster_mean, num_pts]))
        oct_sb_instances = np.stack(oct_sb_instances)

        # normalize the size of the cluster. 
        ivus_sb_instances[:, 10] /= ivus_sb_instances[:, 10].max()
        oct_sb_instances[:, 10] /= oct_sb_instances[:, 10].max()

        # re-order oct in increasing OCT frame order
        oct_sort_indices = np.argsort(oct_sb_instances[:, 0])
        oct_sb_instances = oct_sb_instances[oct_sort_indices]
        oct_sb = [oct_sb[i] for i in oct_sort_indices]
        ivus_sort_indices = np.argsort(ivus_sb_instances[:, 0])
        ivus_sb_instances = ivus_sb_instances[ivus_sort_indices]
        ivus_sb = [ivus_sb[i] for i in ivus_sort_indices]
        gt_assignment = gt_assignment[:, oct_sort_indices]
        gt_assignment = gt_assignment[ivus_sort_indices, :]
        gt_matching0 = gt_matching0[ivus_sort_indices]
        gt_matching1 = gt_matching1[oct_sort_indices]

        # ---- Features ---- # 
        # [0] frame_id, [1] sb_angle, [2] sb_angle circular x, [3] sb_angle_circular y, [4] sb volume, [5] sb_ecc, [6] lumen_vol, 
        # [7] intersection_score, [8] box sigmoid score, [9] original frame ids [10] num points in clusterg

        # pos descriptors
        pos_features_select = []
        if 'frame_id' in self.config['POS_FEAT']: pos_features_select.append(0)
        if 'angle' in self.config['POS_FEAT']: pos_features_select.extend([2,3])
        ivus_pos = ivus_sb_instances[:, pos_features_select]
        oct_pos = oct_sb_instances[:, pos_features_select]

        # context descriptors
        ctx_features_select = []
        if 'sb_vol' in self.config['CTX_FEAT']: ctx_features_select.append(4)
        if 'sb_ecc' in self.config['CTX_FEAT']: ctx_features_select.append(5)
        if 'lum_vol' in self.config['CTX_FEAT']: ctx_features_select.append(6)
        if 'int_score' in self.config['CTX_FEAT']: ctx_features_select.append(7)
        if 'bbox_score' in self.config['CTX_FEAT']: ctx_features_select.append(8)
        if 'calc_arc' in self.config['CTX_FEAT']: ctx_features_select.append(9)
        if 'num_pts' in self.config['CTX_FEAT']: ctx_features_select.append(11)
        ivus_ctx = ivus_sb_instances[:, ctx_features_select]
        oct_ctx = oct_sb_instances[:, ctx_features_select]

        # needed for evaluations
        ivus_frame_id_unnormalised = ivus_sb_instances[:, 10]
        oct_frame_id_unnormalised = oct_sb_instances[:, 10]

        # perform augmentations
        if self.augmentation is not None:
            ivus_pos, oct_pos, ivus_ctx, oct_ctx, gt_assignment = augmentation_pipeline(self.augmentation, 
                ivus_pos, oct_pos, ivus_ctx, oct_ctx, gt_matching0, gt_matching1, gt_assignment)

        # reorder sequence based on frame id
        oct_sort_indices = np.argsort(oct_pos[:, 0])
        oct_pos = oct_pos[oct_sort_indices]
        oct_ctx = oct_ctx[oct_sort_indices]
        oct_frame_id_unnormalised = oct_frame_id_unnormalised[oct_sort_indices]
        oct_sb = [oct_sb[i] for i in oct_sort_indices]

        ivus_sort_indices = np.argsort(ivus_pos[:, 0])
        ivus_pos = ivus_pos[ivus_sort_indices]
        ivus_ctx = ivus_ctx[ivus_sort_indices]
        ivus_frame_id_unnormalised = ivus_frame_id_unnormalised[ivus_sort_indices]
        ivus_sb = [ivus_sb[i] for i in ivus_sort_indices]

        gt_assignment = gt_assignment[:, oct_sort_indices]
        gt_assignment = gt_assignment[ivus_sort_indices, :]
        gt_matching0 = gt_matching0[ivus_sort_indices]
        gt_matching1 = gt_matching1[oct_sort_indices]

        ivus_ctx = torch.from_numpy(ivus_ctx).float() 
        oct_ctx = torch.from_numpy(oct_ctx).float()
        gt_assignment = torch.tensor(gt_assignment).float()
        gt_matching0 = torch.tensor(gt_matching0).long()
        gt_matching1 = torch.tensor(gt_matching1).long()
        ivus_frame_id_unnormalised = torch.from_numpy(ivus_frame_id_unnormalised).float() #
        oct_frame_id_unnormalised = torch.from_numpy(oct_frame_id_unnormalised).float() #
        ivus_pos = torch.from_numpy(ivus_pos).float() #
        oct_pos = torch.from_numpy(oct_pos).float() #

        data = {}
        data['keypoints0'] = ivus_pos
        data['keypoints1'] = oct_pos
        data['original0'] = ivus_sb
        data['original1'] = oct_sb
        data['context0'] = ivus_ctx
        data['context1'] = oct_ctx
        data['gt_assignment'] = gt_assignment
        data['gt_matches0'] = gt_matching0
        data['gt_matches1'] = gt_matching1
        data['pos_unnorm0'] = ivus_frame_id_unnormalised
        data['pos_unnorm1'] = oct_frame_id_unnormalised
        data['ids'] = id_

        return data

    def gt_assignment_2_gt_matches(self, gt_assignment):

        gt_matches0 = gt_assignment.sum(axis=1) 
        gt_matches0[gt_matches0 == 0] = -1
        gt_matches0[gt_matches0 > 0] = 1
        gt_matches1 = gt_assignment.sum(axis=0) 
        gt_matches1[gt_matches1 == 0] = -1
        gt_matches1[gt_matches1 > 0] = 1

        return gt_matches0, gt_matches1

def collate(batch):
    return batch

