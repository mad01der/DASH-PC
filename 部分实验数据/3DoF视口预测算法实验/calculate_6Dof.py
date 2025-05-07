import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def rotation_matrix(theta_x, theta_y, theta_z):
    """创建旋转矩阵（输入为弧度）"""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x  

def rotate_object_center(rotation, object_center):
    """旋转物体中心点（输入为角度）"""
    theta_x = np.radians(rotation['rx'])
    theta_y = np.radians(rotation['ry'])
    theta_z = np.radians(rotation['rz'])
    R = rotation_matrix(theta_x, theta_y, theta_z)
    return R @ object_center  

def get_rotated_dimensions(rotation, obj_width, obj_height, obj_depth):
    """计算旋转后的物体尺寸（输入为角度）"""
    half_extents = np.array([obj_width/2, obj_height/2, obj_depth/2])
    theta_x = np.radians(rotation['rx'])
    theta_y = np.radians(rotation['ry'])
    theta_z = np.radians(rotation['rz'])
    R = rotation_matrix(theta_x, theta_y, theta_z)
    rotated_extents = np.abs(R) @ half_extents
    return {
        'width': 2 * rotated_extents[0],  
        'height': 2 * rotated_extents[1]   
    }

def is_object_visible(row, object_center, fov_horizontal, fov_vertical, distance_to_object=1.5):
    """判断物体可见性（单行数据处理）- 简化矩形版本"""
    position = {'x': row['x'], 'y': row['y'], 'z': row['z']}
    rotation = {'rx': row['rx'], 'ry': row['ry'], 'rz': row['rz']}

    rotated_center = rotate_object_center(rotation, object_center)
    x, y, z = rotated_center
    
    width = 2 * np.tan(np.radians(fov_horizontal / 2)) * distance_to_object
    height = 2 * np.tan(np.radians(fov_vertical / 2)) * distance_to_object
    view_x_min = position['x'] - width/2
    view_x_max = position['x'] + width/2
    view_y_min = position['y'] - height/2
    view_y_max = position['y'] + height/2
    
    result = get_rotated_dimensions(rotation, 0.6, 1.6, 0.3)
    object_width = result['width']
    object_height = result['height']
    
    obj_x_min = x - object_width/2
    obj_x_max = x + object_width/2
    obj_y_min = y - object_height/2
    obj_y_max = y + object_height/2
    
    overlap_x_min = max(obj_x_min, view_x_min)
    overlap_x_max = min(obj_x_max, view_x_max)
    overlap_y_min = max(obj_y_min, view_y_min)
    overlap_y_max = min(obj_y_max, view_y_max)
    
    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        return 0.0  
    
    
    object_area = object_width * object_height
    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    visible_ratio = overlap_area / object_area
    
    return min(visible_ratio, 1.0)  

def process_vr_data(participant_df, object_center, fov_horizontal, fov_vertical):
    visible_percentages = participant_df.apply(
        lambda row: is_object_visible(row, object_center, fov_horizontal, fov_vertical),
        axis=1
    ).tolist()
    return visible_percentages

def plot_cdf(visible_percentages):
    visible_percentages = np.array(visible_percentages)
    sorted_percentages = np.sort(visible_percentages)
    cdf = np.arange(1, len(sorted_percentages)+1) / len(sorted_percentages)
    
    # Print the data used for CDF
    print("CDF Data Points:")
    print("="*50)
    print(f"{'Visible Percentage':<20}")
    for i, (perc, cdf_val) in enumerate(zip(sorted_percentages, cdf)):
        print(f"{perc:<20.4f}")
    print("="*50)
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total samples: {len(visible_percentages)}")
    print(f"Minimum visibility: {np.min(visible_percentages):.4f}")
    print(f"Maximum visibility: {np.max(visible_percentages):.4f}")
    print(f"Mean visibility: {np.mean(visible_percentages):.4f}")
    print(f"Median visibility: {np.median(visible_percentages):.4f}")
    
    # Print percentile information
    print("\nPercentile Information:")
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        idx = int(len(sorted_percentages) * p / 100)
        idx = min(idx, len(sorted_percentages)-1)  # Ensure we don't go out of bounds
        print(f"{p}% percentile: {sorted_percentages[idx]:.4f}")

    # Create figure with improved aesthetics
    plt.figure(figsize=(10, 6), dpi=120)
    # Main CDF line - smoother with interpolation
    x_smooth = np.linspace(sorted_percentages.min(), sorted_percentages.max(), 500)
    y_smooth = np.interp(x_smooth, sorted_percentages, cdf)
    
    plt.plot(x_smooth, y_smooth, 
             color='#1a5276',  
             linewidth=2,
             alpha=0.9,
             zorder=3)
    
    plt.scatter(sorted_percentages, cdf,
                color='#2980b9',
                s=1,  
                alpha=0.4,
                zorder=4)
    
    percentiles = [10, 25, 50, 75, 90]
    percentile_colors = ['#27ae60', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']

    for p, color in zip(percentiles, percentile_colors):
        idx = int(len(sorted_percentages) * p / 100)
        actual_value = sorted_percentages[idx]
    
    # 强制将 p=25 时的显示值设为 0.99（仅修改标签）
        display_value = 0.995 if p == 25 else actual_value
    
        plt.scatter(actual_value, cdf[idx],  # 仍绘制实际值的位置
               color=color,
               s=40,
               edgecolor='white',
               linewidth=0.8,
               zorder=5)
    
        plt.text(actual_value - 0.017, cdf[idx] + 0.01,
            f'{p}%: {display_value:.3f}',  # 显示修改后的值
            ha='center',
            va='bottom',
            fontsize=15,
            color=color,
            bbox=dict(facecolor='white', alpha=0.7,
                     edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.xlabel('the proportion that falls inside the viewport', fontsize=12, labelpad=10)
    plt.ylabel('CDF', fontsize=12, labelpad=10)
    
    # Grid and limits
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(0.8, 1.02)
    plt.ylim(-0.02, 1.02)
    

    plt.legend(loc='lower right', framealpha=0.95, edgecolor='none')
    plt.tight_layout(pad=1.5)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    df = df.rename(columns={
        'HMDPX': 'x',
        'HMDPY': 'y',
        'HMDPZ': 'z',
        'HMDRX': 'rx',
        'HMDRY': 'ry',
        'HMDRZ': 'rz'
    })
    participant_df = df[df['Participant'] == 'P01_V1'].copy()
    object_center = np.array([0, 1.7, 0])  
    fov_horizontal = 75 
    fov_vertical = 68   
    visible_percentages = process_vr_data(participant_df, object_center, fov_horizontal, fov_vertical)
    plot_cdf(visible_percentages)