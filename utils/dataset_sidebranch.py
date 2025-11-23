import torch
from torch.utils.data import Dataset
import numpy as np
import json


class SB_Dataset(Dataset):

    def __init__(self, data_root, modality, subset, transforms, resolution):
        self.transforms = transforms
        self.subset = subset
        self.data_root = data_root
        self.resolution = resolution
        self.modality = modality

        # load all annotations into memory
        with open(data_root + '{}/sb_{}.json'.format(modality, subset), 'r') as j:
            self.annotations = json.loads(j.read())
        # we need to be able to index the annotations in __getitem__ so convert to list
        self.annotations = list(self.annotations.items())

        print('{} set: {} examples'.format(self.subset, len(self.annotations)))

        # classes: 0 index is reserved for background
        self.classes = ['none', 'sidebranch']

    def __getitem__(self, idx):

        img_path, anno = self.annotations[idx]

        vessel_name = img_path[:-5]
        frame_id = img_path[-4:]

        if self.modality == 'ivus':
            image_path = self.data_root + '../Frame Dataset/{}/{}/{}_frames/{}.npy'.format(self.subset, vessel_name, self.modality, frame_id)
            image = np.load(image_path)

        elif self.modality == 'oct':
            image_path = self.data_root + '../Frame Dataset/{}/{}/{}_frames/{}.npy'.format(self.subset, vessel_name,self.modality, frame_id)
            image = np.load(image_path)

        else:
            print('No modality selected')

        image = image / 255.0 # [0,1] just like the pretrained weights.
        image = np.expand_dims(image, -1)

        # uses pascal_voc notations (x1,y1,x2,y2)
        if len(anno) > 0:

            boxes = [l[0] for l in anno]
            # convert boxes into a torch.Tensor
            boxes = np.array(boxes, dtype=np.float32)
            # scale bounding boxes to new image size. i.e 1024 to 224.
            if self.modality == 'oct':
                boxes = boxes * (480 / 1024)
            boxes = np.rint(boxes).astype(np.int64)
            # getting the areas of the boxes
            if len(boxes.shape) > 1:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                area = np.zeros((boxes.shape[0],))

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            area = torch.from_numpy(area)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = vessel_name + '_' + frame_id

        else:

            iscrowd = torch.empty((0), dtype=torch.int64)
            labels = torch.empty((0), dtype=torch.int64)
            area = torch.empty((0), dtype=torch.int64)

            target = {}
            target["boxes"] = np.empty((0,4), dtype=np.float32)
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = vessel_name + '_' + frame_id


        if self.transforms:

            sample = self.transforms(image=image.copy(),
                                     bboxes=target['boxes'],
                                     labels=labels)
            image = sample['image']
            boxes = sample['bboxes']

        image = torch.from_numpy(image).to(torch.float32)
        image = torch.permute(image, (2,0,1))

        if len(anno) > 0:
            target['boxes'] = torch.tensor(boxes)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0), dtype=torch.int64)
        if self.subset != 'test':
            if len(target['boxes'].shape) < 2:
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                print('ERROR: bbox of shape {} in exampe {}'.format(target['boxes'].shape, target["image_id"]))
        return image, target

    def __len__(self):
        return len(self.annotations)
