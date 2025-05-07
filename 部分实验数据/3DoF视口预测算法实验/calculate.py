import re
import numpy as np
import matplotlib.pyplot as plt

pattern = re.compile(
    r"position: \{ x: (?P<x>-?\d+\.\d+), y: (?P<y>-?\d+\.\d+), z: (?P<z>-?\d+\.\d+) \},"
    r"\s*rotation: \{ x: (?P<rx>-?\d+\.\d+), y: (?P<ry>-?\d+\.\d+), z: (?P<rz>-?\d+\.\d+) \}\s*\}"
)

def rotation_matrix(theta_x, theta_y, theta_z):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotate_object_center(rotation, object_center):
    theta_x = np.deg2rad(float(rotation['x']))
    theta_y = np.deg2rad(float(rotation['y']))
    theta_z = np.deg2rad(float(rotation['z']))
    R = rotation_matrix(theta_x, theta_y, theta_z)
    rotated_center = np.dot(R.T, object_center) 
    return rotated_center

def get_rotated_dimensions(rotation, obj_width, obj_height, obj_depth):
    half_extents = np.array([obj_width/2, obj_height/2, obj_depth/2])
    theta_x = np.deg2rad(rotation['x'])
    theta_y = np.deg2rad(rotation['y'])
    theta_z = np.deg2rad(rotation['z'])
    R = rotation_matrix(theta_x, theta_y, theta_z)
    rotated_extents = np.abs(R) @ half_extents  
    return {
        'width': 2 * np.sqrt(rotated_extents[0]**2 + rotated_extents[2]**2), 
        'height': 2 * rotated_extents[1] 
    }

def is_object_visible(rotation,position, rotated_center, fov_horizontal, fov_vertical, distance_to_object):
    x_position = position['x']
    y_position = position['y']
    width = 2 * np.tan(np.deg2rad(fov_horizontal / 2)) * distance_to_object
    height = 2 * np.tan(np.deg2rad(fov_vertical / 2)) * distance_to_object
    x, y, z = rotated_center
    result = get_rotated_dimensions(rotation, 0.4, 1.55, 0.3)
    distance_to_object = 1
    object_width = result['width']
    object_height = result['height']

    segments = [
        {'y_min': 0.875, 'y_max': 1.0, 'width_ratio': 0.4},  
        {'y_min': 0.5, 'y_max': 0.875, 'width_ratio': 0.9}, 
        {'y_min': 0.0, 'y_max': 0.5, 'width_ratio': 0.4}  
    ]
    total_visible = 0.0
    obj_y_min = y - object_height/2
    obj_y_max = y + object_height/2
    view_x_min = x_position - width/2
    view_x_max = x_position + width/2
    view_y_min = y_position - height/2
    view_y_max = y_position + height/2
    
    for segment in segments:
        seg_obj_y_min = obj_y_min + segment['y_min'] * object_height
        seg_obj_y_max = obj_y_min + segment['y_max'] * object_height
        seg_width = object_width * segment['width_ratio']
        seg_x_min = x - seg_width/2
        seg_x_max = x + seg_width/2
        overlap_y_min = max(seg_obj_y_min, view_y_min)
        overlap_y_max = min(seg_obj_y_max, view_y_max)
        if overlap_y_min >= overlap_y_max:
            continue  
        overlap_x_min = max(seg_x_min, view_x_min)
        overlap_x_max = min(seg_x_max, view_x_max)
        if overlap_x_min >= overlap_x_max:
            continue  
        seg_area = (seg_obj_y_max - seg_obj_y_min) * seg_width
        overlap_area = (overlap_y_max - overlap_y_min) * (overlap_x_max - overlap_x_min)
        visible_ratio = overlap_area / seg_area
        segment_height_ratio = (segment['y_max'] - segment['y_min'])
        total_visible += visible_ratio * segment_height_ratio    
    return total_visible

def process_vr_data(file_path, object_center, fov_horizontal, fov_vertical, distance_to_object):
    visible_percentages = []  
    with open(file_path, 'r') as file:
        content = file.read()
        matches = pattern.findall(content)
        for match in matches:
            position = {'x': float(match[0]), 'y': float(match[1]), 'z': float(match[2])}
            rotation = {'x': float(match[3]), 'y': float(match[4]), 'z': float(match[5])}
            rotated_center = rotate_object_center(rotation, object_center)
            visible_percentage = is_object_visible(rotation,position, rotated_center, fov_horizontal, fov_vertical, distance_to_object)
            visible_percentages.append(visible_percentage)   
    return visible_percentages

def plot_cdf(visible_percentages, title="PICO vis"):

    visible_percentages = np.array(visible_percentages)
    sorted_percentages = np.sort(visible_percentages)
    cdf = np.arange(1, len(sorted_percentages)+1) / len(sorted_percentages)
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    plt.plot(sorted_percentages, cdf, 
             color='#1f77b4', 
             linewidth=2.5,
             marker='o',
             markersize=4,
             markerfacecolor='white',
             markeredgewidth=1,
             alpha=0.8,
             label='Visibility CDF')
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        idx = int(len(sorted_percentages) * p / 100)
        plt.scatter(sorted_percentages[idx], cdf[idx], 
                   color='red', 
                   s=60,
                   zorder=5)
        plt.text(sorted_percentages[idx], cdf[idx]+0.03, 
                f'{p}%: {sorted_percentages[idx]:.2f}',
                ha='center',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.xlabel('Visible Percentage', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
file_path = '../location_record/record_1.txt'
object_center = np.array([0, 1.1, -1])
fov_horizontal = 105
fov_vertical = 105
distance_to_object = 1
visible_percentages = process_vr_data(file_path, object_center, fov_horizontal, fov_vertical, distance_to_object)
plot_cdf(visible_percentages)
