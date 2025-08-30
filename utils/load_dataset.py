
from torch.utils.data import DataLoader
from utils.dataset_ivus_oct import IVUS_OCT_Dataset, collate


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

    else:
        print('WARNING - No dataset selected..')


