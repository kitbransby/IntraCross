import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torchvision
import torch
from torchvision import transforms as torchtrans
from models.load_model import load_model
from utils.load_dataset import load_dataset
from utils.eval_utils import plot_img_bbox
from tqdm import tqdm
from detection_utils.mean_avg_precision import mean_average_precision
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
import matplotlib.patches as patches
from torchvision.ops import nms

def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def calc_intersection(bbox, lumen):
    # Ensure that the input arrays are binary (i.e., contain only 0s and 1s)
    assert np.array_equal(bbox, bbox.astype(bool)) and np.array_equal(lumen,lumen.astype(bool)), "Both inputs must be binary arrays."
    # Calculate the intersection: element-wise logical AND
    intersection = np.logical_and(bbox, lumen)
    intersection = np.sum(intersection) / np.sum(lumen)

    return intersection


def bbox_pascal_2_mask(bbox, resolution):
    # Create a blank (black) image
    mask = Image.new("L", (resolution, resolution), 0)

    # Draw the bounding box on the mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=1)

    mask = np.array(mask)

    return mask

def rotate_img_bbox(img, angle):
    """ Rotate the image and return it. """
    rotated_img = F.rotate(img, angle)
    return rotated_img

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

def unrotate_bbox(bbox, angle, img_width=224, img_height=224):
    if angle == 90:
        return torch.stack([img_height - bbox[:, 3], bbox[:, 0], img_height - bbox[:, 1], bbox[:, 2]], dim=1)
    elif angle == 180:
        return torch.stack([img_width - bbox[:, 2], img_height - bbox[:, 3], img_width - bbox[:, 0], img_height - bbox[:, 1]], dim=1)
    elif angle == 270:
        return torch.stack([bbox[:, 1], img_width - bbox[:, 2], bbox[:, 3], img_width - bbox[:, 0]], dim=1)
    return bbox

def combine_predictions(predictions, iou_threshold=0.5, suppression_iou_threshold=0.2, min_votes=2):
    final_boxes = []
    final_scores = []
    final_labels = []

    # Filter out empty predictions
    predictions = [p for p in predictions if p['boxes'].numel() > 0]

    if not predictions:
        # If no valid predictions, return empty tensors
        return {
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'scores': torch.empty((0,), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64)
        }

    all_boxes = torch.cat([p['boxes'] for p in predictions], dim=0)
    all_scores = torch.cat([p['scores'] for p in predictions], dim=0)
    all_labels = torch.cat([p['labels'] for p in predictions], dim=0)

    used_boxes = set()

    for i in range(len(all_boxes)):
        if i in used_boxes:
            continue

        current_box_group = [i]
        vote_count = 1  # Start with 1 vote (the current box)

        for j in range(i + 1, len(all_boxes)):
            if j in used_boxes:
                continue

            if box_iou(all_boxes[i:i + 1], all_boxes[j:j + 1]).item() >= iou_threshold and all_labels[i] == all_labels[
                j]:
                current_box_group.append(j)
                vote_count += 1

        if vote_count >= min_votes:  # Only combine if at least 'min_votes' support it
            used_boxes.update(current_box_group)

            # Combine these boxes by averaging their coordinates, scores, and taking the mode of the labels
            combined_box = all_boxes[current_box_group].mean(dim=0)
            combined_score = all_scores[current_box_group].mean()
            combined_label = all_labels[current_box_group].mode()[0]

            final_boxes.append(combined_box)
            final_scores.append(combined_score)
            final_labels.append(combined_label)

    if len(final_boxes) == 0:
        # If no combined boxes, return empty tensors
        return {
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'scores': torch.empty((0,), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64)
        }

    final_boxes = torch.stack(final_boxes)
    final_scores = torch.stack(final_scores)
    final_labels = torch.stack(final_labels)

    # Now apply NMS to the combined predictions to remove overlapping boxes
    keep_indices = nms(final_boxes, final_scores, suppression_iou_threshold)

    return {
        'boxes': final_boxes[keep_indices],
        'scores': final_scores[keep_indices],
        'labels': final_labels[keep_indices]
    }

def main(config):

    save_folder = os.path.join(config['DATA_ROOT'], '..', 'Side_Branch_Detection', "results", config['RUN_ID'])
    try:
        os.mkdir(save_folder + '/pred')
    except Exception as e:
        print('Warning: ', e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    # load dataset
    (_, _, test_loader), (_, _, test_dataset) = load_dataset(config)

    # load model and pretrained weights.
    model = load_model(config, device)
    weights = torch.load(save_folder + '/best_map.pt')
    model.load_state_dict(weights)

    pred_boxes = []
    true_boxes = []

    pred_binary_all = []
    true_binary_all = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            img, target = test_dataset[i]

            if config['TTA']:

                img_rot90 = rotate_img_bbox(img, 90)
                img_rot180 = rotate_img_bbox(img, 180)
                img_rot270 = rotate_img_bbox(img, 270)
                img_cat = torch.stack([img, img_rot90, img_rot180, img_rot270], dim=0).to(device)

                # f, axes = plt.subplots(1,4, figsize=(20,5))
                # for i, ax in enumerate(axes.flatten()):
                #     ax.imshow(img_cat[i][0].cpu().numpy())
                # plt.show()

                pred_all = model(img_cat)

                # f, axes = plt.subplots(1, 4, figsize=(20, 5))
                # for i, ax in enumerate(axes.flatten()):
                #     ax.imshow(img_cat[i][0].cpu().numpy())
                #     for box_i, box_pred in enumerate(pred_all[i]['boxes']):
                #         bbox = box_pred.cpu().numpy()
                #         x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                #         rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                #         ax.add_patch(rect)
                # plt.show()

                pred_original = pred_all[0]
                pred_rot90 = pred_all[1]
                pred_rot90['boxes'] = unrotate_bbox(pred_rot90['boxes'], 90)
                pred_rot180 = pred_all[2]
                pred_rot180['boxes'] = unrotate_bbox(pred_rot180['boxes'], 180)
                pred_rot270 = pred_all[3]
                pred_rot270['boxes'] = unrotate_bbox(pred_rot270['boxes'], 270)

                predictions = [pred_original, pred_rot90, pred_rot180, pred_rot270]

                pred = combine_predictions(predictions)

                f, axes = plt.subplots(1, 5, figsize=(20, 5))
                for i, ax in enumerate(axes.flatten()[:4]):
                    ax.imshow(img_cat[0][0].cpu().numpy())
                    for box_i, box_pred in enumerate(predictions[i]['boxes']):
                        bbox = box_pred.cpu().numpy()
                        x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.set_title('prediction ' + str(i))
                axes[4].imshow(img_cat[0][0].cpu().numpy())
                for box_i, box_pred in enumerate(pred['boxes']):
                    bbox = box_pred.cpu().numpy()
                    x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    axes[4].add_patch(rect)
                    axes[4].set_title('avg pred')
                plt.savefig(save_folder+'/pred/'+ target["image_id"] + '_tta.jpg')
                plt.close()

            else:
                pred = model(img.unsqueeze(0).to(device))[0]

            plot_img_bbox(torch_to_pil(img), target, pred, save_folder+'/pred/'+ target["image_id"] + '.jpg', low_resolution=False)

            # post processing to remove boxes with 50% overlap with lumen
            idx_to_rm = []
            if config['MODALITY'] == 'oct':
                vessel_wal_path = config['DATA_ROOT'] + 'Vessel Wall Annotations/test/{}/{}/{}_mask.npy'.format(target['image_id'][:-5], config['MODALITY'], target['image_id'][-4:])
                vessel_wall = np.load(vessel_wal_path)
                if config['MODALITY'] == 'oct':
                    lumen_mask = vessel_wall == 1
                else:
                    lumen_mask = vessel_wall == 2

                for box_i, box_pred in enumerate(pred['boxes']):
                    #print(box_pred)
                    bbox_mask = bbox_pascal_2_mask(box_pred.cpu().numpy().tolist(), config['RESOLUTION'])
                    intersection = calc_intersection(bbox_mask, lumen_mask)
                    #print(intersection)
                    if intersection > 0.8 or intersection == 0:
                        #print('removing box for ', target['image_id'], 'intersection with lumen: ', intersection)
                        idx_to_rm.append(box_i)

            pred_binary = 0
            for box_i in range(len(pred['boxes'])):
                if box_i in idx_to_rm:
                    continue
                pred_boxes.append([target['image_id'],
                                   pred['labels'][box_i].cpu(),
                                   pred['scores'][box_i].cpu(),
                                   *pred['boxes'][box_i].cpu()])
                if pred['labels'][box_i].cpu() > 0:
                    pred_binary = 1

            true_binary = 0
            for box_i in range(len(target['boxes'])):
                true_boxes.append([target['image_id'],
                                   target['labels'][box_i],
                                   None,
                                   *target['boxes'][box_i]])
                if target['labels'][box_i].cpu() > 0:
                    true_binary = 1

            if true_binary == 1 or pred_binary == 1:
                plot_img_bbox(torch_to_pil(img), target, pred, save_folder+'/pred/'+ target["image_id"] + '.jpg', low_resolution=False)
                

            true_binary_all.append(true_binary)
            pred_binary_all.append(pred_binary)
            # if i == 50:
            #  break

    AP, precision, recall = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='corners', num_classes=2)
    f1 = (2 * precision * recall) / (precision + recall)
    scores = {'AP': AP, 'Precision': precision, 'Recall': recall, 'F1': f1}
    print(scores)
    with open(save_folder+'/test_scores.txt', 'w') as f:
        print(scores, file=f)

    # Compute confusion matrix
    cm = confusion_matrix(true_binary_all, pred_binary_all)


    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No SB', 'SB'],
                yticklabels=['No SB', 'SB'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_folder+'/confusion.png', dpi=300)

    report = classification_report(true_binary_all, pred_binary_all)
    print("Classification Report:\n", report)
    with open(save_folder+'/classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../Data/')
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--RUN_ID', type=str)
    parser.add_argument('--INFERENCE', action=argparse.BooleanOptionalAction, default=True)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)