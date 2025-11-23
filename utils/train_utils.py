import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.eval_utils import interpolate_longitudinal, interpolate_frame_angles, angle_difference, consistent_normalize, find_min_distance_configuration, extract_keypt_rots
from utils.preprocess import extract_matching_labels, circular_to_angle

def plot_losses_longi_rot(train_loss, val_loss, val_longi_r1, val_rot_r1, folder):

    epochs = len(train_loss)
    f, axes = plt.subplots(1,3, figsize=(20,10))

    axes[0].plot(list(range(epochs)), train_loss, label='train_loss')
    axes[0].plot(list(range(epochs)), val_loss, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0, max(max(val_loss), max(train_loss)) * 1.2)
    axes[1].plot(list(range(epochs)), val_longi_r1, label='longi vs r1')
    axes[1].set_title('OCT frame dist')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Frame Dist')
    axes[1].grid()
    axes[1].set_ylim(0, max(val_longi_r1) * 1.2)
    axes[2].plot(list(range(epochs)), val_rot_r1, label='rot vs r1')
    axes[2].set_title('OCT frame angle diff')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Angle (degrees)')
    axes[2].grid()
    axes[2].set_ylim(0, max(val_rot_r1) * 1.1)
    plt.savefig(folder + '/progress.png', dpi=200)
    plt.close('all')



def viz_superglue_matching(pred, data, save_folder, set_, epoch):
    sinkhorn_cost = pred['sinkhorn_cost'].detach().cpu().numpy()[0]
    gt_matching = data['gt_assignment'].cpu().numpy()[0]
    m0 = pred['matches0'].detach().cpu().numpy()[0]
    m1 = pred['matches1'].detach().cpu().numpy()[0]
    gt_m0 = data['gt_matches0'].cpu().numpy()[0]
    gt_m1 = data['gt_matches1'].cpu().numpy()[0]
    kpts0 = data['keypoints0'].cpu().numpy()[0]
    kpts1 = data['keypoints1'].cpu().numpy()[0]
    matching_scores0 = pred['matching_scores0'].detach().cpu().numpy()[0]
    matching_scores1 = pred['matching_scores1'].detach().cpu().numpy()[0]
    original0 = data['original0']
    original1 = data['original1']

    pred_matching = np.zeros_like(gt_matching)
    pred_matching_soft = np.zeros_like(gt_matching).astype(np.float32)
    for i, match_m1 in enumerate(m0):
        if match_m1 != -1:
            pred_matching[i, match_m1] = 1
            pred_matching_soft[i, match_m1] = matching_scores0[i]
    for i, match_m0 in enumerate(m1):
        if match_m0 != -1:
            pred_matching[match_m0, i] = 1
            pred_matching_soft[match_m0, i] = matching_scores1[i]

    m0 = [x for x in m0 if x != -1]
    m1 = [x for x in m1 if x != -1]
    gt_m0 = [x for x in gt_m0 if x != -1]
    gt_m1 = [x for x in gt_m1 if x != -1]

    f, axes = plt.subplots(1, 3, figsize=(10, 5))
    im1 = axes[0].imshow(sinkhorn_cost)
    f.colorbar(im1, ax=axes[0], shrink=0.6)
    axes[1].imshow(pred_matching)
    #axes[1].set_title('pred: \nm0: {} \n m1: {}'.format(str(m0), str(m1)))
    axes[1].set_title('pred')
    axes[2].imshow(gt_matching)
    axes[2].set_title('gt')
    #axes[2].set_title('gt: \nm0: {} \n m1: {}'.format(str(gt_m0), str(gt_m1)))
    plt.savefig(save_folder + '/{}/{}_{}_cost.jpg'.format(set_, str(epoch), data['ids']), dpi=100)
    plt.close('all')

    ivus_frame_id = kpts0[:,0]
    ivus_angle_cir = kpts0[:,1:3]
    ivus_angle = circular_to_angle(ivus_angle_cir)
    ivus = np.stack([ivus_frame_id, ivus_angle], axis=-1)

    oct_frame_id = kpts1[:,0]
    oct_angle_cir = kpts1[:,1:3]
    oct_angle = circular_to_angle(oct_angle_cir)
    oct = np.stack([oct_frame_id, oct_angle], axis=-1)


    TP = (pred_matching == 1) & (gt_matching == 1)  # True Positives
    TN = (pred_matching == 0) & (gt_matching == 0)  # True Negatives
    FP = (pred_matching == 1) & (gt_matching == 0)  # False Positives
    FN = (pred_matching == 0) & (gt_matching == 1)  # False Negatives

    H, W = pred_matching.shape

    f, axes = plt.subplots(2,1,figsize=(15,5), gridspec_kw={'hspace': 0.5})
    
    for sb_cluster in original0:
        axes[0].scatter(sb_cluster[:,0], sb_cluster[:,1] * 360, s=5, c='blue', alpha=0.3)
    axes[0].scatter(ivus_frame_id, ivus_angle, s=10, c='black')
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(0, 360)
    axes[0].set_title('IVUS SBs')
    
    for sb_cluster in original1:
        axes[1].scatter(sb_cluster[:,0], sb_cluster[:,1] * 360, s=5, c='blue', alpha=0.3)
    axes[1].scatter(oct_frame_id, oct_angle, s=10, c='black')
    axes[1].set_xlim(-0.1, 1.1)
    axes[1].set_ylim(0, 360)
    axes[1].set_title('OCT SBs')

    # Get transformation objects for aligning lines across subplots
    transFigure = f.transFigure  # Figure-wide transformation
    ax0_bbox = axes[0].get_position()  # Position of the first subplot
    ax1_bbox = axes[1].get_position()  # Position of the second subplot

    # Normalize the coordinates for the lines
    for i in range(H):
        for j in range(W):
                
            if (pred_matching[i, j] == 1) or (gt_matching[i, j] == 1):  # If binary mask value is 1
                # Normalize the x and y coordinates
                x_start = ax0_bbox.x0 + (ivus[i, 0] + 0.1) * (ax0_bbox.x1 - ax0_bbox.x0) / 1.2
                y_start = ax0_bbox.y0 + (ivus[i, 1] / 360) * (ax0_bbox.y1 - ax0_bbox.y0)
                x_end = ax1_bbox.x0 + (oct[j, 0] + 0.1) * (ax1_bbox.x1 - ax1_bbox.x0) / 1.2
                y_end = ax1_bbox.y0 + (oct[j, 1] / 360) * (ax1_bbox.y1 - ax1_bbox.y0)

                if TP[i,j] == 1:
                    color = 'green'
                elif TN[i,j] == 1:
                    color = 'blue'
                elif FP[i,j] == 1:
                    color = 'red'
                elif FN[i,j] == 1:
                    color = 'orange'
                else:
                    print('no color')
    
                # Plot the line across subplots
                line = plt.Line2D([x_start, x_end], [y_start, y_end], transform=f.transFigure,
                                  color=color, linewidth=2, alpha=0.6, zorder=1)
                f.add_artist(line)

                text = '{:.2f}'.format(pred_matching_soft[i,j])

                # Add text in the middle of the line
                x_mid = (x_start + x_end) / 2
                y_mid = (y_start + y_end) / 2
                plt.text(x_mid, y_mid, text, transform=f.transFigure, 
                         color='black', fontsize=8, ha='center', va='center', alpha=1, zorder=10)
    
    plt.savefig(save_folder + '/{}/{}_{}_match.jpg'.format(set_, str(epoch), data['ids']), dpi=100)
    plt.close('all')



def load_gt_val(id_):
    # --- LOAD GT ---- #
    r1_keypt_ids, r1_keypt_rot, r1_ivus_oct_rough_match, r1_ivus_oct_keypt_match, _, _, _ = extract_matching_labels(
        '../Data/Observer Variability/R1/{}.txt'.format(id_[5:]))
    start = r1_ivus_oct_keypt_match[0,:].astype(np.int32)
    end = r1_ivus_oct_keypt_match[-1,:].astype(np.int32)

    r1_longi_seg_interpolated = interpolate_longitudinal(r1_ivus_oct_keypt_match, start, end)

    r1_oct_rot = extract_keypt_rots(r1_ivus_oct_keypt_match, r1_keypt_ids, r1_keypt_rot)
    r1_oct_rot[:,1] += 180 # normalize from [-180, +180] to [0, 360]
    r1_oct_rot = find_min_distance_configuration(r1_oct_rot)
    r1_oct_rot[:,1] = consistent_normalize(r1_oct_rot[:,1], 360, 720)

    rot_start = r1_oct_rot[0,1]
    rot_end = r1_oct_rot[-1,1]

    r1_oct_rot_interpolated = interpolate_frame_angles(r1_oct_rot, start[1], end[1])
    r1_oct_rot_interpolated[:,1] = consistent_normalize(r1_oct_rot_interpolated[:,1], 360, 720)

    r1_oct_rot[:,1] = consistent_normalize(r1_oct_rot[:,1], 360, 720)

    longi = [r1_longi_seg_interpolated, r1_ivus_oct_keypt_match]
    rot = [r1_oct_rot_interpolated, r1_oct_rot]
    
    return longi, rot

def validation(model, val_dataset, device, save_folder, postprocessing, save_viz=None, save_pred=False, verbose=True):

    longi_r1_pred = []
    longi_computer_pred = []
    longi_r1_pred_interpolated = []
    longi_computer_pred_interpolated = []

    rot_r1_pred = []
    rot_computer_pred = []
    rot_r1_pred_interpolated = []
    rot_computer_pred_interpolated = []

    total_loss = []


    with torch.no_grad():
        
        for i in range(len(val_dataset)):

            # --- LOAD DATA + GT ---- #
            
            data = val_dataset[i]
            id_ = data['ids']

            longi_gt, rot_gt = load_gt_val(id_)
            r1_longi_interpolated, r1_longi_keypt = longi_gt
            r1_rot_interpolated, r1_rot_keypt = rot_gt
            ivus_ed = np.load('../Data/Registration Dataset/val/{}/ivus_ids.npy'.format(id_))
            oct_ed = np.load('../Data/Registration Dataset/val/{}/oct_ids.npy'.format(id_))
            rot_interpolated = np.load('../Data/Registration Dataset/val/{}/rot_interpolated.npy'.format(id_))
            start = r1_longi_keypt[0,:].astype(np.int32)
            end = r1_longi_keypt[-1,:].astype(np.int32)

            # --- FORWARD PASS ---- #
            
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.unsqueeze(0).to(device)

            if 'img0' in data.keys():
                pred = model(
                            data['keypoints0'], data['keypoints1'], 
                            data['context0'], data['context1'], 
                            data['img0'], data['img1']
                            )
            else:
                pred = model(
                            data['keypoints0'], data['keypoints1'], 
                            data['context0'], data['context1'], 
                            )
            loss = model.loss(pred, data)['total']
            total_loss.append(loss.cpu().numpy())


            m0 = pred['matches0'].detach().cpu().numpy()[0]
            m1 = pred['matches1'].detach().cpu().numpy()[0]
            matching_scores0 = pred['matching_scores0'].detach().cpu().numpy()[0]
            matching_scores1 = pred['matching_scores1'].detach().cpu().numpy()[0]
            log_assignment = pred['log_assignment'].detach().cpu().numpy()[0]
            frame_id0 = data['pos_unnorm0'].cpu().numpy()[0]
            frame_id1 = data['pos_unnorm1'].cpu().numpy()[0]
            keypoints0 = data['keypoints0'].cpu().numpy()[0]
            keypoints1 = data['keypoints1'].cpu().numpy()[0]
            original0 = data['original0']
            original1 = data['original1']

            # --- VIZUALIZE ---- #

            viz_superglue_matching(pred, data, save_folder, 'val', 0)

            # --- EXTRACT FINAL PREDICTION ---- #

            r1_rot_keypt[:, 1] = r1_rot_keypt[:, 1] % 360

            final_matching = []
            final_angles = []
            for m0_idx, m1_idx in enumerate(m0):
                if m1_idx != -1:
                    norm_pos0 = original0[m0_idx][:,0].mean(axis=0)
                    norm_pos1 = original1[m1_idx][:,0].mean(axis=0)
                    pos0 = original0[m0_idx][:,-1].mean(axis=0)
                    pos1 = original1[m1_idx][:,-1].mean(axis=0)
                    angle0 = original0[m0_idx][:,2:4].mean(axis=0)
                    angle1 = original1[m1_idx][:,2:4].mean(axis=0)
                    angle0 = circular_to_angle(np.expand_dims(angle0, axis=0))[0]
                    angle1 = circular_to_angle(np.expand_dims(angle1, axis=0))[0]


                    # find the original angle1 (before alignment) 
                    oct_frame_id = np.rint(pos1).astype(np.int32)
                    if oct_frame_id < start[1]:
                        rot = rot_interpolated[0, 1]
                    elif oct_frame_id > end[1]:
                        rot = rot_interpolated[-1, 1]
                    else:
                        idx = np.where(rot_interpolated[:,0] == oct_frame_id)
                        rot = rot_interpolated[idx, 1][0][0]

                    angle1 = (angle1 - rot) % 360

                    # normalise to same range as GT. 
                    rotation_oct_to_ivus = angle0 - angle1
                    rotation_oct_to_ivus += 180
                    rotation_oct_to_ivus = rotation_oct_to_ivus % 360

                    final_matching.append([pos0, pos1])
                    final_angles.append([pos1, rotation_oct_to_ivus])
            final_matching = np.array(final_matching)
            final_angles = np.array(final_angles)

            # post processing. 
            final_matching_cleaned = []
            final_angles_cleaned = []
            for (iv_frame_id, oc_frame_id), (oc_frame_id, oc_angle) in zip(final_matching, final_angles):
                # find closest ed frames. 
                ed_dists = []
                for ivus_ed_id in ivus_ed:
                    ed_dists.append(np.abs(ivus_ed_id - iv_frame_id))
                ed_argmin = np.argmin(ed_dists)
                iv_frame_id = ivus_ed[ed_argmin]
                #print('new frame id: ', iv_frame_id)
                ed_dists = []
                for oct_ed_id in oct_ed:
                    ed_dists.append(np.abs(oct_ed_id - oc_frame_id))
                ed_argmin = np.argmin(ed_dists)
                oc_frame_id = oct_ed[ed_argmin]
                # remove any points before or after end points. and add seg start and end points. 
                if (iv_frame_id <= r1_longi_keypt[0,0] or oc_frame_id <= r1_longi_keypt[0,1]) or (iv_frame_id >= r1_longi_keypt[-1,0] or oc_frame_id >= r1_longi_keypt[-1,1]):
                    #print('No points outside start and end points. Skipping...')
                    pass
                else:
                    #print('Adding points: IVUS frame id: {}, OCT frame id: {}'.format(iv_frame_id, oc_frame_id))
                    final_matching_cleaned.append([iv_frame_id, oc_frame_id])
                    final_angles_cleaned.append([oc_frame_id, oc_angle])
            final_matching_cleaned = np.array(final_matching_cleaned)
            final_angles_cleaned = np.array(final_angles_cleaned)
            if final_matching_cleaned.shape[0] > 0:
                final_matching_cleaned = np.concatenate([np.expand_dims(r1_longi_keypt[0,:], 0), final_matching_cleaned, np.expand_dims(r1_longi_keypt[-1,:], 0)])
                final_angles_cleaned = np.concatenate([np.expand_dims(r1_rot_keypt[0,:], 0), final_angles_cleaned, np.expand_dims(r1_rot_keypt[-1,:], 0)])  
                #print('Appending start and end points: {} and {}'.format(r1_longi_keypt[0,:], r1_longi_keypt[-1,:]))
                #print('Appending start and end angles: {} and {}'.format(r1_rot_keypt[0,:], r1_rot_keypt[-1,:]))
            else:
                final_matching_cleaned = r1_longi_keypt[[0, -1],:]
                final_angles_cleaned = r1_rot_keypt[[0, -1],:]

            # --- EVALUATE ---- #

            computer_longi_keypt = final_matching_cleaned
            computer_longi_interpolated = interpolate_longitudinal(computer_longi_keypt, start, end)

            #print(computer_longi_interpolated.sum())

            computer_rot = find_min_distance_configuration(final_angles_cleaned)
            computer_rot[:,1] = consistent_normalize(computer_rot[:,1], 360, 720)
            computer_rot_interpolated = interpolate_frame_angles(computer_rot, start[1], end[1])
            computer_rot_interpolated[:,1] = consistent_normalize(computer_rot_interpolated[:,1], 360, 720)

            computer_rot_interpolated[:,1] = computer_rot_interpolated[:,1] % 360
            r1_rot_interpolated[:,1] = r1_rot_interpolated[:,1] % 360
            computer_rot[:,1] = computer_rot[:,1] % 360
            r1_rot_keypt[:,1] = r1_rot_keypt[:,1] % 360

            # save predictions for later
            longi_r1_pred.extend(r1_longi_interpolated[(r1_longi_keypt[1:-1,0] - start[0]).astype(np.int32), 1])
            longi_computer_pred.extend(computer_longi_interpolated[(r1_longi_keypt[1:-1,0] - start[0]).astype(np.int32), 1])
            longi_r1_pred_interpolated.extend(r1_longi_interpolated[:,1])
            longi_computer_pred_interpolated.extend(computer_longi_interpolated[:,1])

            rot_r1_pred.extend(r1_rot_interpolated[(r1_rot_keypt[1:-1,0] - start[1]).astype(np.int32), 1])
            rot_computer_pred.extend(computer_rot_interpolated[(r1_rot_keypt[1:-1,0] - start[1]).astype(np.int32), 1])
            rot_r1_pred_interpolated.extend(r1_rot_interpolated[:,1])
            rot_computer_pred_interpolated.extend(computer_rot_interpolated[:,1])

            if save_viz is not None:
                f, axes = plt.subplots(1,2,figsize=(10,5))
                axes[0].scatter(*r1_longi_interpolated.T, c='lime', alpha=0.3, s=1)
                axes[0].scatter(*r1_longi_keypt.T, label='R1', c='lime', alpha=1, s=20)
                axes[0].scatter(*computer_longi_keypt.T, label='Computer', c='r', alpha=1, s=20)
                axes[0].scatter(*computer_longi_interpolated.T, c='r', s=1, alpha=0.3)
                axes[0].legend()
                axes[0].set_xlabel('IVUS frame id')
                axes[0].set_ylabel('OCT frame id')
                #axes[0].set_title("Longitudinal: \nR1 vs Computer (all): {:.1f}, R1 vs Computer (keypt): {:.1f}, \nR2 vs Computer (all): {:.1f}, R2 vs Computer (keypt): {:.1f}, \nR1 vs R2 (all): {:.1f}, R1 vs R2 (keypt): {:.1f} ".format(
                #         longi_computer_r1_all_mean, longi_computer_r1_keypt_mean, longi_computer_r2_all_mean, longi_computer_r2_keypt_mean, longi_r1_r2_all_mean, longi_r1_r2_keypt_mean), fontsize =10)
                axes[1].scatter(*computer_rot_interpolated.T, c='r', s=1)
                axes[1].scatter(*r1_rot_interpolated.T, s=1, c='lime')
                axes[1].scatter(*r1_rot_keypt.T, s=20, c='lime', label='R1')
                axes[1].scatter(*computer_rot.T, s=20, c='r', label='Computer')
                axes[1].set_ylim(0, 360)
                axes[1].set_ylabel('Angle (degrees)')
                axes[1].set_xlabel('OCT frame id')
                axes[1].legend()
                #axes[1].set_title("Rotational: \nR1 vs Computer (all): {:.1f}, R1 vs Computer (keypt): {:.1f}, \nR2 vs Computer (all): {:.1f}, R2 vs Computer (keypt): {:.1f}, \nR1 vs R2 (all): {:.1f}, R1 vs R2 (keypt): {:.1f} ".format(
                #        angle_computer_R1_all_mean, angle_computer_R1_keypt_mean, angle_computer_R2_all_mean, angle_computer_R2_keypt_mean, angle_R1_R2_all_mean, angle_R1_R2_keypt_mean), fontsize =10)
                plt.suptitle(id_)
                plt.tight_layout()
                plt.savefig(save_folder + '/val/{}_registration.jpg'.format(id_), dpi=100)
                plt.close('all')

    
    longi_r1_computer_diff = np.mean(np.abs(np.array(longi_r1_pred) - np.array(longi_computer_pred)))
    longi_r1_computer_interpolated_diff = np.mean(np.abs(np.array(longi_r1_pred_interpolated) - np.array(longi_computer_pred_interpolated)))

    rot_r1_computer_diff = np.mean(angle_difference(np.array(rot_r1_pred), np.array(rot_computer_pred)))
    rot_r1_computer_interpolated_diff = np.mean(angle_difference(np.array(rot_r1_pred_interpolated), np.array(rot_computer_pred_interpolated)))

    loss = np.mean(total_loss)

    return loss, longi_r1_computer_diff, rot_r1_computer_diff

def plot_sb_detector_loss(train_loss, val_loss, val_map, folder):
    epochs = len(val_loss)
    f, axes = plt.subplots(1,2, figsize=(20,10))
    axes[0].plot(list(range(epochs)), train_loss, label='train_loss')
    axes[0].plot(list(range(epochs)), val_loss, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0, 0.4)
    axes[1].plot(list(range(epochs)), val_map, label='val mAP')
    axes[1].set_title('mAP')
    axes[1].legend()
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('mAP')
    axes[1].set_ylim(0, 0.8)
    plt.savefig(folder + '/progress.png', dpi=200)
    plt.close('all')

