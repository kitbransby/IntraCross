import argparse
import os
import yaml
import datetime
import time
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from utils.train_utils import validation, plot_losses_longi_rot
from models.load_model import load_model
from utils.load_dataset import load_dataset

from utils.load_dataset import trash 

def main(config):

    torch.autograd.set_detect_anomaly(True)

    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    model = load_model(config, device)

    summary(model, input_size=[
        (config['TRAIN_BATCH_SIZE'], 20, config['POS_DIM']),
        (config['TRAIN_BATCH_SIZE'], 15, config['POS_DIM']), 
        (config['TRAIN_BATCH_SIZE'], 20, config['CTX_DIM']),
        (config['TRAIN_BATCH_SIZE'], 15, config['CTX_DIM']), 
    ])

    optimizer = torch.optim.Adam(model.parameters(), config['LR'])
    max_epochs = config['EPOCHS']

    # create save folder
    save_folder = os.path.join(config['RESULTS_ROOT'], config['RUN_ID'])
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)

    # initialise tracking variables
    best_loss = np.inf
    best_comb = np.inf
    train_loss, val_loss = [], []
    val_longi_r1, val_rot_r1 = [], []

    # ------------------ #
    # ---- TRAINING ---- #
    # ------------------ #

    print('Starting Training...')
    for epoch in range(max_epochs):
        start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        probs_all = []
        gt_all = []

        # train epoch
        for batch in train_loader:
            optimizer.zero_grad()
            total_loss = 0
            for data in batch:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.unsqueeze(0).to(device)
                pred = model(
                    data['keypoints0'], data['keypoints1'], 
                    data['context0'], data['context1'], 
                    )
                losses = model.loss(pred, data)
                total_loss += losses['total']
            average_loss = total_loss / len(batch)
            average_loss.backward()
            optimizer.step()
            epoch_loss += average_loss.item()
            step += 1
        epoch_loss /= step
        train_loss.append(epoch_loss)
        print(f"Train epoch {epoch + 1}: avg loss {epoch_loss:.4f}")

        # validation epoch
        model.eval()
        results = validation(model, val_dataset, device, save_folder, postprocessing=config['OUTLIER_REMOVAL'], save_viz=None, save_pred=False, verbose=True)
        epoch_loss, longi_r1_computer_diff, rot_r1_computer_diff = results
        val_loss.append(epoch_loss)
        val_longi_r1.append(longi_r1_computer_diff)
        val_rot_r1.append(rot_r1_computer_diff)
        print(f"Val epoch {epoch + 1}: avg loss {epoch_loss:.4f} longi vs R1 {longi_r1_computer_diff:.1f} rot vs R1 {rot_r1_computer_diff:.1f}" )
        end = time.time()
        epoch_time = end - start
        print('Epoch time: {:.2f}s'.format(epoch_time))

        # roughly normalise each to [0,1]
        avg_comb = (longi_r1_computer_diff / 6) + (rot_r1_computer_diff / 20) 

        # save best weights
        if avg_comb <= best_comb:
            best_comb = avg_comb
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_comb.pt'))
            print("saved model new best overall")

        # plot metrics over epochs
        plot_losses_longi_rot(train_loss, val_loss, val_longi_r1, val_rot_r1, save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../Data/')
    parser.add_argument('--RESULTS_ROOT', type=str, default='Results/')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=4)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=4)
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--RUN_ID', type=str, default='local')
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)