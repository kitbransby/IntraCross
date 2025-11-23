import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from utils.preprocess import circular_to_angle, angle_to_circular
import albumentations as A


def augmentation_pipeline(aug_list, ivus_pos, oct_pos, ivus_ctx, oct_ctx, gt_matching0, gt_matching1, gt_assignment):
    
    # For each unmatched points, move them to a random position (angle and pos).
    if 'UNMATCHED_SHIFT' in aug_list.keys() and np.random.rand() < aug_list['UNMATCHED_SHIFT']:
        unmatched_idx0 = np.where(gt_matching0 == -1)[0]
        for idx in unmatched_idx0:
            unmatched0 = ivus_pos[idx, :]
            new_frame_id = np.array([np.random.uniform(low=-0.1, high=1.1)]).reshape(1, 1)
            new_angle = np.random.randint(0, 359)
            new_circular = angle_to_circular(np.expand_dims(new_angle, 0))
            ivus_pos[idx, :] = np.concatenate([new_frame_id, new_circular], axis=-1)

        unmatched_idx1 = np.where(gt_matching1 == -1)[0]
        for idx in unmatched_idx1:
            unmatched1 = oct_pos[idx, :]
            new_frame_id = np.array([np.random.uniform(low=-0.1, high=1.1)], dtype=np.float32).reshape(1, 1)
            new_angle = np.random.randint(0, 359)
            new_circular = angle_to_circular(np.expand_dims(new_angle, 0))
            oct_pos[idx, :] = np.concatenate([new_frame_id, new_circular], axis=-1)

    # For all the matched points, move the pair to a random position and angle then add some shift to oct. sample from spine. 
    if 'MATCHED_SHIFT' in aug_list.keys() and np.random.rand() < aug_list['MATCHED_SHIFT']:

        _, theta_shift = generate_random_function(
            V=np.random.randint(100, 150), 
            num_control_points=np.random.randint(3, 6), 
            num_samples=100)

        _, pos_shift = generate_random_function(
            V=np.random.uniform(0.10, 0.18), 
            num_control_points=np.random.randint(3, 6), 
            num_samples=100)

        idxs0, idxs1 = np.where(gt_assignment == 1)

        for idx0, idx1 in zip(idxs0, idxs1):

            new_pos_ivus = np.array([np.random.uniform(low=-0.1, high=1.1)], dtype=np.float32).reshape(1, 1)
            new_theta_ivus = np.random.randint(0, 359)
            if new_pos_ivus <= 0:
                new_pos_oct = new_pos_ivus + pos_shift[0] 
                theta_shift_oct = theta_shift[0]

            elif new_pos_ivus >= 1:
                new_pos_oct = new_pos_ivus + pos_shift[-1]
                theta_shift_oct = theta_shift[-1]
                
            else:
                idx = np.floor(new_pos_ivus * 100).astype(np.int32)
                new_pos_oct = new_pos_ivus + pos_shift[idx]
                idx = np.clip(np.floor(new_pos_oct * 100).astype(np.int32), 0, 99)
                theta_shift_oct = theta_shift[idx]

            new_theta_oct = new_theta_ivus + theta_shift_oct

            ivus_pos[idx0, 1:] = angle_to_circular((new_theta_ivus + np.random.uniform(-3, 3)) % 360) 
            oct_pos[idx1, 1:] = angle_to_circular((new_theta_oct + np.random.uniform(-3, 3)) % 360)
            ivus_pos[idx0, 0] = new_pos_ivus + np.random.uniform(-0.02, 0.02)
            oct_pos[idx1, 0] = new_pos_oct + np.random.uniform(-0.02, 0.02)


    if 'NOISE' in aug_list.keys() and np.random.rand() < aug_list['NOISE']:
        # add some noise to the context features. 
        ivus_ctx *= np.random.uniform(0.9, 1.1, size=ivus_ctx.shape)
        oct_ctx *= np.random.uniform(0.9, 1.1, size=oct_ctx.shape)

    return ivus_pos, oct_pos, ivus_ctx, oct_ctx, gt_assignment

def generate_random_function(V, num_control_points=10, num_samples=1000):
    """
    Generate a smooth random function y(x) with:
    - x in [0, 100]
    - y in [-V, +V]
    - y(0) = 0 and y(100) = 0
    
    Parameters:
    - V: Maximum absolute y-value.
    - num_control_points: Number of random control points.
    - num_samples: Number of x samples to generate.
    
    Returns:
    - x (numpy array): x-values
    - y (numpy array): smooth y-values
    """
    x_control = np.linspace(0, 100, num_control_points)
    y_control = np.random.uniform(-V, V, size=num_control_points)
    
    # Ensure start and end points are zero
    y_control[0] = 0
    y_control[-1] = 0

    # Interpolate with a cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')

    x = np.linspace(0, 100, num_samples)
    y = spline(x)
    
    return x, y

def get_transform_sidebranch(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(p=0.5, rotate_limit=(0,0), border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            #ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
def get_transform_calcium():
    return A.Compose([
            A.Rotate(180, p=1, interpolation=2, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05)],
        keypoint_params=A.KeypointParams(format='xy')
        )