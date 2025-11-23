import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm

def load_list_from_txt(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

class Calcium_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root, subset, transforms, modality):

        self.data_root = data_root
        self.transforms = transforms
        self.modality = modality
        self.subset = subset

        if subset == 'train':
            self.frame_list = load_list_from_txt(data_root  + f'{modality}_train_frames_10k_subset.txt')
        elif subset == 'val':
            self.frame_list = load_list_from_txt(data_root  + f'{modality}_val_frames.txt')
        elif subset == 'test':
            self.frame_list = load_list_from_txt(data_root  + f'{modality}_test_frames.txt')
        else:
            print('WARNING: No set list found')

        self.all_mask_paths = self.get_mask_paths()

        print('{} set has {} ca masks'.format(subset, len(self.frame_list)))

    def __len__(self):
        return len(self.frame_list)

    def cart_to_polar(self, image, center):
        polar_image = cv2.linearPolar(image,
                                      center,
                                      image.shape[0] // 2,
                                      cv2.WARP_FILL_OUTLIERS)
        return polar_image

    def get_mask_paths(self):
        all_mask_paths = []
        for vessel_frame_id in tqdm(self.frame_list):
            all_mask_paths.append(f'{self.data_root}Calcium Annotations/{self.subset}/{vessel_frame_id[:-5]}/{self.modality}/{vessel_frame_id[-4:]}_calc_mask.npy')
        print('Total: ', len(all_mask_paths), all_mask_paths[:5])
        return all_mask_paths

    def __getitem__(self, idx):

        vessel_frame_id = self.frame_list[idx]
        img_name =  f'{self.data_root}/Frame Dataset/{self.subset}/{vessel_frame_id[:-5]}/{self.modality}_frames/{vessel_frame_id[-4:]}.npy'
        centre_name = f'{self.data_root}/Calcium Annotations/{self.subset}/{vessel_frame_id[:-5]}/{self.modality}/{vessel_frame_id[-4:]}_lumen_centre.npy'
        mask_name = f'{self.data_root}/Calcium Annotations/{self.subset}/{vessel_frame_id[:-5]}/{self.modality}/{vessel_frame_id[-4:]}_mask.npy'

        image = np.load(img_name)
        mask = np.load(mask_name)
        centre = np.load(centre_name)

        id_ = '{}_{}'.format(img_name.split('/')[-3], img_name.split('/')[-1].split('.')[0])

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, keypoints=[(centre[0], centre[1])])
            image, mask, centre = transformed['image'], transformed['mask'], transformed['keypoints']
            centre = np.array(centre[0], dtype=np.uint16)

        centre = centre.astype(np.uint16)

        mask_polar = self.cart_to_polar(mask, centre)
        image_polar = self.cart_to_polar(image, centre)

        label = mask_polar.sum(axis=1)
        label[label > 0] = 1
        label[label < 1] = 0
        label = label.astype(np.uint8)

        # To Tensor
        image_polar = np.expand_dims(image_polar, axis=0)
        image_polar = np.clip(image_polar, 0, 255)
        image_polar = image_polar / 255
        image_polar = torch.from_numpy(image_polar).float()
        label = torch.from_numpy(label).long()

        return image_polar, label, id_