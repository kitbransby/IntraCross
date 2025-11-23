import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.load_model import load_model
from utils.load_dataset import load_dataset
from torchinfo import summary
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
import seaborn as sns

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    # load dataset
    (_, _, _), (_, _, test_dataset) = load_dataset(config)

    # get the model using our helper function
    model = load_model(config, device)

    model = model.to(device)
    summary(model, (2, config['INPUT_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    weight_dir = os.path.join(config['RESULTS_ROOT'], config['RUN_ID'], 'best_pos_f1.pt')
    print('Loading weights from...', weight_dir)
    model.load_state_dict(torch.load(weight_dir))
    model.eval()

    save_folder = os.path.join(config['RESULTS_ROOT'], config['RUN_ID'], 'pred')
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print(e)

    all_test_preds = []
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset))):

            sample = test_dataset[i]
            image, label, id_ = sample
            image = image.to(device)
            image = torch.unsqueeze(image, 0)
            label = label.numpy()

            logits = model(image)

            logits = logits.reshape(-1, 2)
            label = label.reshape(-1)

            pred = logits.argmax(dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_test_probs.append(probs)
            all_test_preds.extend(pred)
            all_test_labels.extend(label)

    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)

    AUPRC = average_precision_score(all_test_labels.flatten(), all_test_probs.flatten())
    print(f'Average Precision: {AUPRC:.3f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Calcium', 'Calcium'],
                    yticklabels=['No Calcium', 'Calcium'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('{} Calcium'.format(config['MODALITY']))
    plt.savefig(save_folder + '/{}_confusion.png'.format(config['MODALITY']), dpi=300)

    report = classification_report(all_test_labels, all_test_preds)
    print("Classification Report:\n", report)
    with open(save_folder + '/classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../Data/')
    parser.add_argument('--RESULTS_ROOT', type=str, default='results/')
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