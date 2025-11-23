
from torch.utils.data import DataLoader
from utils.dataset_ivus_oct import IVUS_OCT_Dataset, collate
from utils.dataset_sidebranch import SB_Dataset, collate_fn_inc_negatives
from utils.augmentation import get_transform_calcium, get_transform_sidebranch
from utils.dataset_oct_calcium import Calcium_Dataset

def load_dataset(config):

    data_root = config['DATA_ROOT']

    if config['DATASET'] == 'ivus_oct_dataset':
        train_dataset = IVUS_OCT_Dataset(
            data_root=data_root + 'Registration Dataset v2/', 
            subset='train', 
            augmentation=config['AUG_LIST'], 
            config=config)
        val_dataset = IVUS_OCT_Dataset(
            data_root=data_root + 'Registration Dataset v2/', 
            subset='val', 
            augmentation=None, 
            config=config)
        test_dataset = IVUS_OCT_Dataset(
            data_root=data_root + 'Registration Dataset v2/', 
            subset='test', 
            augmentation=None, 
            config=config)
        collate_fn = collate
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['TRAIN_BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=config['NUM_WORKERS'],
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True,
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['VAL_BATCH_SIZE'],
                                 num_workers=config['NUM_WORKERS'],
                                 pin_memory=True,
                                 collate_fn=collate_fn)
        return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)
    
    elif config['DATASET'] == 'sidebranch':
        train_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                   modality=config['MODALITY'],
                                   subset='train',
                                   transforms=get_transform_sidebranch(train=config['AUGMENTATION']),
                                   resolution=config['RESOLUTION'])
        val_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                 modality=config['MODALITY'],
                                 subset='val',
                                 transforms=False,
                                 resolution=config['RESOLUTION'])
        test_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                  modality=config['MODALITY'],
                                  subset='test',
                                  transforms=False,
                                  resolution=config['RESOLUTION'])

        train_loader = DataLoader(train_dataset,
                                  batch_size=config['TRAIN_BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=config['NUM_WORKERS'],
                                  collate_fn=collate_fn_inc_negatives)
        val_loader = DataLoader(val_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True,
                                collate_fn=collate_fn_inc_negatives)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['VAL_BATCH_SIZE'],
                                 num_workers=config['NUM_WORKERS'],
                                 pin_memory=True,
                                 collate_fn=collate_fn_inc_negatives)

        return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)
    
    elif config['DATASET'] == 'calcium':
        train_dataset = Calcium_Dataset(data_root=data_root,
                                   modality=config['MODALITY'],
                                   subset='train',
                                   transforms=get_transform_calcium())
        val_dataset = Calcium_Dataset(data_root=data_root,
                                 modality=config['MODALITY'],
                                 subset='val',
                                 transforms=False)
        test_dataset = Calcium_Dataset(data_root=data_root,
                                  modality=config['MODALITY'],
                                  subset='test',
                                  transforms=False)
        
        train_loader = DataLoader(train_dataset,
                                batch_size=config['TRAIN_BATCH_SIZE'],
                                shuffle=True,
                                num_workers=config['NUM_WORKERS'])
        val_loader = DataLoader(val_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True)
        
        return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)

    else:
        print('WARNING - No dataset selected..')


