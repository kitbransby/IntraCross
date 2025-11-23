import argparse
import yaml
import torch
import numpy as np
import os
from torchinfo import summary
from models.load_model import load_model
from utils.load_dataset import load_dataset
from utils.train_utils import plot_sb_detector_loss
from detection_utils.engine import train_one_epoch, evaluate, evaluate_loss


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    # load dataset
    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    # get the model using our helper function
    model = load_model(config, device)

    summary(model, (config['TRAIN_BATCH_SIZE'], config['INPUT_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config['LR'])

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs -- DISABLED
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    save_folder = os.path.join(config['DATA_ROOT'], '..', 'Side_Branch_Detection', "results", config['RUN_ID'])
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)

    best_val_loss = np.inf
    best_val_map = 0
    train_losses = []
    val_losses = []
    val_map_all = []

    for epoch in range(config['EPOCHS']):
        # training for one epoch
        model, metrics, train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=config['TRAIN_PRINT'])
        # update the learning rate
        lr_scheduler.step()
        # validation loss
        val_loss = evaluate_loss(model, val_loader, device=device)

        # evaluate on the test dataset
        evaluator = evaluate(model, val_loader, device=device)

        val_map = evaluator.coco_eval['bbox'].stats[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss.cpu().numpy())
        val_map_all.append(val_map)

        print('Val mAP: {:.3f}, loss: {:.3f}'.format(val_map, val_loss))

        if val_map > best_val_map:
            print('Val mAP improved from {:.3f} to {:.3f}. Saving model to..{}'.format(best_val_map, val_map, save_folder))
            best_val_map = val_map
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_map.pt'))
            print("saved model new best mAP")

        # plot loss and metrics
        plot_sb_detector_loss(train_losses, val_losses, val_map_all, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../Data/')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=64)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=64)
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--RUN_ID', type=str, default='local')
    parser.add_argument('--INFERENCE', action=argparse.BooleanOptionalAction, default=False)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)