import argparse
import yaml
import time
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import CrossEntropyLoss
from models.load_model import load_model
from utils.load_dataset import load_dataset
from torchinfo import summary
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def multiclass_dice_loss(logits, targets, eps=1e-6):
    """
    logits: Tensor of shape (B, C=2, H)
    targets: LongTensor of shape (B, H) with values in {0, 1}
    """
    num_classes = logits.shape[1]  # should be 2

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)  # shape: (B, 2, H)

    # One-hot encode targets to shape (B, 2, H)
    targets_onehot = F.one_hot(targets, num_classes=num_classes)  # (B, H, 2)
    targets_onehot = targets_onehot.permute(0, 2, 1).float()       # (B, 2, H)

    # Compute per-class Dice
    intersection = (probs * targets_onehot).sum(dim=2)
    union = probs.sum(dim=2) + targets_onehot.sum(dim=2)
    dice = (2 * intersection + eps) / (union + eps)

    # Average over classes and batch
    return 1 - dice.mean()

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')  # shape (B, H)
    probs = F.softmax(logits, dim=1)  # shape (B, 2, H)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    # load dataset
    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    # get the model using our helper function
    model = load_model(config, device)

    model = model.to(device)
    summary(model, (config['TRAIN_BATCH_SIZE'], config['INPUT_DIM'], config['RESOLUTION'], config['RESOLUTION']))
    
    optimizer = torch.optim.Adam(model.parameters(), config['LR'])
    max_epochs = config['EPOCHS']

    class_weights = torch.tensor([1.0 , 1.0], dtype=torch.float32)
    print('Class weighting in CE: ', class_weights)

    #dice_loss = DiceLoss().to(device)
    ce_loss = CrossEntropyLoss(weight=class_weights).to(device)

    save_folder = os.path.join(config['RESULTS_ROOT'], config['RUN_ID'])

    train_loss, val_loss = [], []
    train_macro_f1_all, val_macro_f1_all = [], []
    train_pos_f1_all, val_pos_f1_all = [], []
    train_auc_all, val_auc_all = [], []
    train_auc_pr_all, val_auc_pr_all = [], []
    train_pos_rate_all, val_pos_rate_all = [], []
    best_pos_f1 = 0
    best_loss = np.inf
    best_auc = 0

    for epoch in range(max_epochs):
        start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []

        
        for batch in train_loader:
            image, label, id_ = batch
            image, label = image.to(device), label.to(device)

            logits = model(image)
            optimizer.zero_grad()
            loss = ce_loss(logits.transpose(1,2), label)

            running_loss += loss.item()

            logits = logits.reshape(-1,2)
            label = label.reshape(-1)
            probs = F.softmax(logits, dim=-1)

            preds = logits.argmax(dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_probs.extend(probs[:,1].detach().cpu().numpy())
            all_train_labels.extend(label.cpu().numpy())

            loss.backward()
            optimizer.step()

        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_pr_auc = average_precision_score(all_train_labels, all_train_probs)
        positive_rate = (np.array(all_train_probs) > 0.5).astype(np.float32).mean()

        results = classification_report(all_train_labels, all_train_preds, target_names=['no calcium', 'calcium'], output_dict=True)
        train_macro_f1 = results['macro avg']['f1-score']
        train_pos_f1 = results['calcium']['f1-score']

        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        train_macro_f1_all.append(train_macro_f1)
        train_pos_f1_all.append(train_pos_f1)
        train_auc_all.append(train_auc)
        train_auc_pr_all.append(train_pr_auc)
        train_pos_rate_all.append(positive_rate)

        print(f"Train epoch {epoch + 1}: avg loss {avg_train_loss:.4f}, pos rate {positive_rate:.5f}" )

        model.eval()
        with torch.no_grad():
            running_loss = 0
            all_val_preds = []
            all_val_labels = []
            all_val_probs = []
 
            for batch in val_loader:
                image, label, id_ = batch
                image, label = image.to(device), label.to(device)

                logits = model(image)
                loss = ce_loss(logits.transpose(1,2), label)
                running_loss += loss.item()

                logits = logits.reshape(-1, 2)
                label = label.reshape(-1)
                probs = F.softmax(logits, dim=-1)

                preds = logits.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs[:,1].cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())

            val_auc = roc_auc_score(all_val_labels, all_val_probs)
            val_pr_auc = average_precision_score(all_val_labels, all_val_probs)
            positive_rate = (np.array(all_val_probs) > 0.5).astype(np.float32).mean()

            results = classification_report(all_val_labels, all_val_preds, target_names=['no calcium', 'calcium'], output_dict=True)
            val_macro_f1 = results['macro avg']['f1-score']
            val_pos_f1 = results['calcium']['f1-score']

            avg_val_loss = running_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            val_macro_f1_all.append(val_macro_f1)
            val_pos_f1_all.append(val_pos_f1)
            val_auc_all.append(val_auc)
            val_auc_pr_all.append(val_pr_auc)
            val_pos_rate_all.append(positive_rate)

            print(f"Val epoch {epoch + 1}: avg loss {avg_val_loss:.4f}, pos rate {positive_rate:.5f}" )

        end = time.time()
        epoch_time = end - start

        print('Epoch time: {:.2f}s'.format(epoch_time))

        if val_pos_f1 >= best_pos_f1:
            best_pos_f1 = val_pos_f1
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_pos_f1.pt'))
            print("saved model new best pos f1")
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_auc.pt'))
            print("saved model new best auc")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../Data/')
    parser.add_argument('--RESULTS_ROOT', type=str, default='results/')
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=10)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=10)
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