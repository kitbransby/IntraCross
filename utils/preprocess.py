import numpy as np
import re
import os
from itertools import islice
import re
import os

def extract_matching_labels(file_path):
    """
    Function to extract expert's matching labels from an XML file.
    """
 
    # Open the file with the appropriate encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines in the file
        lines = file.readlines()

    # Define a regular expression pattern to match the lines with <Point ... />
    pattern = re.compile(r'<Point (\d+)\s+(\d+)\s+([-+]?\d+\.\d+) \/>')

    # List to hold all extracted points
    oct_keypt_ids = []
    oct_keypt_rot = []
    for line in lines:
        match = pattern.search(line)
        if match:
            # Extract the three groups of numbers
            oct_keypt_ids.append(int(match.group(2)))
            oct_keypt_rot.append(float(match.group(3)))

    # Initialize variables to hold the second numbers from ResampleParameters1 and ResampleParameters2
    ivus_len = None
    oct_len = None

    # Define a regular expression pattern to match the ResampleParameters lines
    resample_pattern = re.compile(r'<ResampleParameters\d+>\s*(\d+)\s+(\d+)\s+\d+\s*</ResampleParameters\d+>')

    for line in lines:
        match = resample_pattern.search(line)
        if match:
            if ivus_len is None:
                ivus_len = int(match.group(2)) + 1
            else:
                oct_len = int(match.group(2)) + 1
                break  # Once both parameters are found, we can stop searching

    assert ivus_len is not None
    assert oct_len is not None
    assert ivus_len > 10
    assert oct_len > 10

    ivus_oct_rough_match = []
    ivus_oct_keypt_match = []
    # Initialize a flag to detect when to start processing lines
    process_lines = False
    # Define a regular expression pattern to match lines with two numbers
    pattern = re.compile(r'^(\d+)\s+(\d+)\s+\d+$')
    for line in lines:
        line = line.strip()  # Trim whitespace from the start and end of the line
        if line == '</ImageRotationPoints>':
            process_lines = True  # Set the flag to start processing the following lines
            continue  # Skip to the next iteration of the loop
        if process_lines:
            # Check if the line matches the pattern
            match = pattern.match(line)
            if match:
                first_num, second_num = match.groups()
                ivus_oct_rough_match.append([int(first_num), int(second_num)])
                
                if line.split(' ')[-1] == '1':
                    ivus_oct_keypt_match.append([int(first_num), int(second_num)])
            elif '<' in line and '>' in line:
                break  # Optional: Stop processing if another XML-like tag is encountered
                
    
    ivus_oct_keypt_match_many_to_one = []
    for oct_keypt in oct_keypt_ids:
        for ivus_id, oct_id in ivus_oct_rough_match:
            if oct_keypt == oct_id:
                ivus_oct_keypt_match_many_to_one.append([ivus_id, oct_id])
                #print(ivus_id, oct_id, oct_keypt)
            
    return np.array(oct_keypt_ids), np.array(oct_keypt_rot), np.array(ivus_oct_rough_match), np.array(ivus_oct_keypt_match), np.array(ivus_oct_keypt_match_many_to_one), ivus_len, oct_len

def int_to_4_digit_string(num):
    return f"{num:04d}"

def circular_to_angle(circular_coords):
    # Convert the normalized circular coordinates back to their original range
    x = 2 * circular_coords[:, 0] - 1
    y = 2 * circular_coords[:, 1] - 1
    # Compute the angle in radians using arctan2
    angle_radians = np.arctan2(y, x)
    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)
    # Ensure the angle is in the range [0, 360)
    angle_degrees = np.mod(angle_degrees, 360)
    return angle_degrees

def angle_to_circular(angle):
    return (np.column_stack((np.cos(np.radians(angle)), np.sin(np.radians(angle)))) + 1) / 2


