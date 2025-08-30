import numpy as np
from scipy.interpolate import interp1d
from utils.preprocess import extract_matching_labels
from itertools import product

def load_gt_test(id_):
    # --- LOAD GT ---- #
    xingwei_keypt_ids, xingwei_keypt_rot, xingwei_ivus_oct_rough_match, xingwei_ivus_oct_keypt_match, _, _, _ = extract_matching_labels('../Data/Observer Variability/Xingwei V2 (no anno)/{}.txt'.format(id_[5:]))
    emir_keypt_ids, emir_keypt_rot, emir_ivus_oct_rough_match, emir_ivus_oct_keypt_match, _, _, _ = extract_matching_labels('../Data/Observer Variability/Emir V4 (no anno)/{}.txt'.format(id_[5:]))
    emir_repeat_keypt_ids, emir_repeat_keypt_rot, emir_repeat_ivus_oct_rough_match, emir_repeat_ivus_oct_keypt_match, _, _, _ = extract_matching_labels('../Data/Observer Variability/Emir V5 (no anno)/{}.txt'.format(id_[5:]))
    start = emir_ivus_oct_keypt_match[0,:].astype(np.int32)
    end = emir_ivus_oct_keypt_match[-1,:].astype(np.int32)

    xingwei_longi_seg_interpolated = interpolate_longitudinal(xingwei_ivus_oct_keypt_match, start, end)
    emir_longi_seg_interpolated = interpolate_longitudinal(emir_ivus_oct_keypt_match, start, end)
    emir_repeat_longi_seg_interpolated = interpolate_longitudinal(emir_repeat_ivus_oct_keypt_match, start, end)

    xingwei_oct_rot = extract_keypt_rots(xingwei_ivus_oct_keypt_match, xingwei_keypt_ids, xingwei_keypt_rot)
    xingwei_oct_rot[:,1] += 180 # normalize from [-180, +180] to [0, 360]
    xingwei_oct_rot = find_min_distance_configuration(xingwei_oct_rot)
    xingwei_oct_rot[:,1] = consistent_normalize(xingwei_oct_rot[:,1], 360, 720)
    emir_oct_rot = extract_keypt_rots(emir_ivus_oct_keypt_match, emir_keypt_ids, emir_keypt_rot)
    emir_oct_rot[:,1] += 180
    emir_oct_rot = find_min_distance_configuration(emir_oct_rot)
    emir_oct_rot[:,1] = consistent_normalize(emir_oct_rot[:,1], 360, 720)
    emir_repeat_oct_rot = extract_keypt_rots(emir_repeat_ivus_oct_keypt_match, emir_repeat_keypt_ids, emir_repeat_keypt_rot)
    emir_repeat_oct_rot[:,1] += 180
    emir_repeat_oct_rot = find_min_distance_configuration(emir_repeat_oct_rot)
    emir_repeat_oct_rot[:,1] = consistent_normalize(emir_repeat_oct_rot[:,1], 360, 720)

    rot_start = ((xingwei_oct_rot[0,1]) + (emir_oct_rot[0,1]) + (emir_repeat_oct_rot[0,1])) / 3
    rot_end = ((xingwei_oct_rot[-1,1]) + (emir_oct_rot[-1,1]) + (emir_repeat_oct_rot[-1,1])) / 3
    xingwei_oct_rot[0,1] = rot_start
    xingwei_oct_rot[-1,1] = rot_end
    emir_oct_rot[0,1] = rot_start
    emir_oct_rot[-1,1] = rot_end
    emir_repeat_oct_rot[0,1] = rot_start
    emir_repeat_oct_rot[-1,1] = rot_end

    # Normalise so they are all in same range (0, 360) and always takes the shortest path around the circle. 
    xingwei_oct_rot_interpolated = interpolate_frame_angles(xingwei_oct_rot, start[1], end[1])
    xingwei_oct_rot_interpolated[:,1] = consistent_normalize(xingwei_oct_rot_interpolated[:,1], 360, 720)
    emir_oct_rot_interpolated = interpolate_frame_angles(emir_oct_rot, start[1], end[1])
    emir_oct_rot_interpolated[:,1] = consistent_normalize(emir_oct_rot_interpolated[:,1], 360, 720)
    emir_repeat_oct_rot_interpolated = interpolate_frame_angles(emir_repeat_oct_rot, start[1], end[1])
    emir_repeat_oct_rot_interpolated[:,1] = consistent_normalize(emir_repeat_oct_rot_interpolated[:,1], 360, 720)

    xingwei_oct_rot[:,1] = consistent_normalize(xingwei_oct_rot[:,1], 360, 720)
    emir_repeat_oct_rot[:,1] = consistent_normalize(emir_repeat_oct_rot[:,1], 360, 720)
    emir_oct_rot[:,1] = consistent_normalize(emir_oct_rot[:,1], 360, 720)

    longi = [emir_longi_seg_interpolated, xingwei_longi_seg_interpolated, emir_repeat_longi_seg_interpolated, emir_ivus_oct_keypt_match, xingwei_ivus_oct_keypt_match, emir_repeat_ivus_oct_keypt_match]
    rot = [emir_oct_rot_interpolated, xingwei_oct_rot_interpolated, emir_repeat_oct_rot_interpolated, emir_oct_rot, xingwei_oct_rot, emir_repeat_oct_rot]
    
    return longi, rot

def interpolate_longitudinal(points, start_point, max_point):
    """
    Interpolates missing values between given points in a 2D array,
    and also interpolates to (0,0) and the (i, j) max point.

    Args:
    - points (np.ndarray): 2D array of shape (N, 2) representing x and y positions.
    - start_point (tuple): Starting point (x, y).
    - max_point (tuple): The (i, j) point representing the maximum length of the sequence.

    Returns:
    - interpolated_points (np.ndarray): 2D array with interpolated points.
    """
    # Adding start_point and max_point to the array
    #points = np.vstack((start_point, points, max_point))
    
    # Sorting points by x-values
    points = points[np.argsort(points[:, 0])]
    
    # Ensuring unique x-values by aggregating duplicate x-values
    unique_x = {}
    for x, y in points:
        if x in unique_x:
            if x == start_point[0]:
                unique_x[x].append(start_point[1]) # 1st value in seq make sure its the start pt
            elif x == max_point[0]:
                unique_x[x].append(max_point[1]) # last value in seq make sure its the last pt
            else:
                unique_x[x].append(y)
        else:
            if x == start_point[0]:
                unique_x[x] = [start_point[1]] #.append(start_point[1])
            elif x == max_point[0]:
                unique_x[x] = [max_point[1]] #.append(end_point[1])
            else:
                unique_x[x] = [y]
    
    # Replace duplicate x-values with the mean of their y-values
    aggregated_points = np.array([[x, np.mean(y_list)] for x, y_list in unique_x.items()])
    
    # Extract x and y values
    x_values = aggregated_points[:, 0]
    y_values = aggregated_points[:, 1]
    
    # Create interpolation function for y values over x
    interp_func = interp1d(x_values, y_values, kind='linear', fill_value="extrapolate")
    
    # Generate a sequence of x values from start_point[0] to max_point[0] to interpolate over
    interpolated_x = np.arange(start_point[0], max_point[0] + 1)
    
    # Use the interpolation function to get corresponding y values
    interpolated_y = interp_func(interpolated_x)
    
    # Combine the x and y into a single array
    interpolated_points = np.vstack((interpolated_x, interpolated_y)).T
    
    return interpolated_points

import numpy as np

def interpolate_frame_angles(data, start, end):
    """
    Interpolate angles for missing frame IDs to ensure all consecutive frame IDs have angles.
    Accounts for circular angles by choosing the shortest path for interpolation.

    Parameters:
        data (numpy.ndarray): An N x 2 array where the first column contains frame IDs
                              (not necessarily consecutive), and the second column contains angles in degrees.
        start (int): The starting frame ID.
        end (int): The ending frame ID.

    Returns:
        numpy.ndarray: A new array with consecutive frame IDs and interpolated angles.
    """
    # Add start and end points of the sequence
    data = np.concatenate(
        [np.array([[start, data[0, 1]]]), 
         data, 
         np.array([[end, data[-1, 1]]])], axis=0
    )
    
    # Extract frame IDs and angles
    frame_ids = data[:, 0]
    angles = data[:, 1]

    # Normalize angles to the range [0, 360)
    angles = angles % 360

    # Adjust for circular interpolation
    adjusted_angles = [angles[0]]
    for i in range(1, len(angles)):
        prev_angle = adjusted_angles[-1]
        current_angle = angles[i]
        
        # Calculate difference in both clockwise and counterclockwise directions
        diff_cw = (current_angle - prev_angle) % 360
        diff_ccw = (prev_angle - current_angle) % 360
        
        # Choose the smaller of the two and adjust the current angle
        if diff_cw <= diff_ccw:
            adjusted_angle = prev_angle + diff_cw
        else:
            adjusted_angle = prev_angle - diff_ccw

        adjusted_angles.append(adjusted_angle)

    # Interpolate adjusted angles for the full range of frame IDs
    full_frame_ids = np.arange(frame_ids[0], frame_ids[-1] + 1)
    interpolated_adjusted_angles = np.interp(full_frame_ids, frame_ids, adjusted_angles)

    # Normalize back to [0, 360) for the final angles
    interpolated_angles = interpolated_adjusted_angles % 360

    # Combine the full frame IDs and interpolated angles into a single array
    interpolated_data = np.column_stack((full_frame_ids, interpolated_angles))

    return interpolated_data


def angle_difference(angle1, angle2, norm='0_360', absolute=True):
    """
    Compute the difference between two angles, considering circular wrap-around.
    
    Parameters:
        angle1 (float or numpy.ndarray): First angle(s) in degrees.
        angle2 (float or numpy.ndarray): Second angle(s) in degrees.

    Returns:
        numpy.ndarray: The difference between the angles, in the range [-180, 180].
    """

    angle1 = np.array(angle1)
    angle2 = np.array(angle2)
    # Compute the raw difference
    diff = np.abs(angle1 % 360 - angle2 % 360) % 360
    diff = np.minimum(diff, 360 - diff)
            
    return diff

def extract_keypt_rots(ivus_oct_keypt_match, keypt_ids, keypt_rot): 
    oct_rot = []
    for oct_frame_id in ivus_oct_keypt_match[:,1]:
        if oct_frame_id in keypt_ids:
            idx = np.where(keypt_ids == oct_frame_id)[0]
            deg = np.rad2deg(keypt_rot[idx])[0]
            oct_rot.append([oct_frame_id, deg])
    oct_rot = np.array(oct_rot)
    return oct_rot

def consistent_normalize(sequence, target_min=360, target_max=720):
    """
    Normalize a sequence of angles to the range [target_min, target_max].
    Ensures consistent normalization for all points in the sequence.
    
    Args:
        sequence (array-like): Input sequence of angles.
        target_min (int): Lower bound of the target range.
        target_max (int): Upper bound of the target range.
    
    Returns:
        numpy.ndarray: Consistently normalized angles within the target range.
    """
    sequence = np.array(sequence)
    range_width = target_max - target_min

    # Calculate the overall shift to align the sequence with the target range
    avg_value = np.mean(sequence)
    shift = ((avg_value - target_min) // range_width) * range_width
    
    # Normalize the sequence using the consistent shift
    normalized_sequence = sequence - shift
    
    # If any values are still out of range, shift further to align
    while np.any(normalized_sequence < target_min):
        normalized_sequence += range_width
    while np.any(normalized_sequence >= target_max):
        normalized_sequence -= range_width
    
    return normalized_sequence



def find_min_distance_configuration(data):
    """
    Find the configuration of angles that minimizes total absolute distance.

    Parameters:
        data (np.ndarray): Array of shape (N, 2), where each row contains [frame_id, angle].

    Returns:
        tuple: (minimum_distance, best_configuration)
    """
    frame_ids = data[:, 0]
    original_angles = data[:, 1]

    # Generate all possible configurations by adding 360 to each angle (or keeping it as is)
    configurations = product(*[[angle, angle + 360] for angle in original_angles])

    def total_distance(angles):
        """
        Compute the total absolute distance for adjacent angles.
        """
        return sum(absolute_distance(angles[i], angles[i+1]) for i in range(len(angles) - 1))

    def absolute_distance(angle1, angle2):
        """
        Compute the absolute distance between two angles.
        """
        return abs(angle1 - angle2)

    min_distance = float('inf')
    best_configuration = None

    for config in configurations:
        distance = total_distance(config)
        if distance < min_distance:
            min_distance = distance
            best_configuration = config

    result = np.array([frame_ids, best_configuration]).T

    if (result[:,1] > 360).all():
        result[:,1] -= 360

    return result

