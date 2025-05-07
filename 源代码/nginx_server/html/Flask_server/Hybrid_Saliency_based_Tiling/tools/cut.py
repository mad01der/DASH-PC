import open3d as o3d
import numpy as np

def cut(file_path, blocks_num):
    point_cloud = o3d.io.read_point_cloud(file_path)
    if point_cloud.is_empty():
        print("Error!")
        return None
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)  
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    x_intervals = np.linspace(x_min, x_max, num=blocks_num)
    y_intervals = np.linspace(y_min, y_max, num=blocks_num)
    z_intervals = np.linspace(z_min, z_max, num=blocks_num)
    blocks = {}
    for z_idx in range(blocks_num-1):
        for x_idx in range(blocks_num-1):
            for y_idx in range(blocks_num-1):
                x_min_block, x_max_block = x_intervals[x_idx], x_intervals[x_idx + 1]
                y_min_block, y_max_block = y_intervals[y_idx], y_intervals[y_idx + 1]
                z_min_block, z_max_block = z_intervals[z_idx], z_intervals[z_idx + 1]
                
                mask = (points[:, 0] >= x_min_block) & (points[:, 0] < x_max_block) & \
                       (points[:, 1] >= y_min_block) & (points[:, 1] < y_max_block) & \
                       (points[:, 2] >= z_min_block) & (points[:, 2] < z_max_block) 
                block_points = points[mask]
                block_colors = colors[mask]
                blocks[(x_idx, y_idx, z_idx)] = {
                    "x_range": (x_min_block, x_max_block),
                    "y_range": (y_min_block, y_max_block),
                    "z_range": (z_min_block, z_max_block),
                    "points": block_points,
                    "colors": block_colors
                }
    
    return blocks

