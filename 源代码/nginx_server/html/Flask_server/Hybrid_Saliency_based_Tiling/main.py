import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import tools.cut as cut
import yaml
import open3d as o3d
import render_result as rr
import os
import json
import pickle
import random
import shutil
import time
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_points_to_ply(points, colors, output_file):
    points_array = np.array(points)
    colors_array = np.array(colors) 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_array) 
    o3d.io.write_point_cloud(output_file, point_cloud, write_ascii=True)

def hierarchical_clustering(saliency_values, num_clusters):
    data = saliency_values.reshape(-1, 1)
    linkage_matrix = linkage(data, method='ward')
    cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust') 
    return cluster_labels, linkage_matrix

def downsample_points_and_colors(points, colors, opt_s_value):
    sample_ratio = max(0, min(1, opt_s_value))  
    total_points = len(points)
    num_samples = int(total_points * sample_ratio)
    indices_to_keep = random.sample(range(total_points), num_samples)
    indices_to_keep_set = set(indices_to_keep)
    sampled_points = [point for idx, point in enumerate(points) if idx in indices_to_keep_set]
    sampled_colors = [color for idx, color in enumerate(colors) if idx in indices_to_keep_set]
    return sampled_points, sampled_colors

def generate_blocks_file():
    config = load_config('./config.yml')  
    blocks_num = config['blocks_num']
    N = (blocks_num-1) * (blocks_num - 1) * (blocks_num - 1)
    file_folder = config['file_folder']
    file_list = [f for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
    file_list.sort() 
    group_size = config['group_size']
    for i in range(0, len(file_list), group_size):
        group = file_list[i:i + group_size]
        group_number = i // group_size + 1 
        file_path = os.path.join(file_folder, group[0])
        file_second_path = os.path.join(file_folder, group[1])
        file_path_later = os.path.join(file_folder, group[-1])
        print(f"Group {group_number}:")
        blocks = cut.cut(file_path, blocks_num)
        blocks_second = cut.cut(file_second_path, blocks_num)
        blocks_later = cut.cut(file_path_later, blocks_num)
        keys = list(blocks.keys())
        keys_second = list(blocks_second.keys())
        keys_later = list(blocks_later.keys())
        optimized_s = rr.get_optimized_s(group_number,N,blocks,blocks_later)
        print(f"optimized_s for {group_number} is",optimized_s)
        points_with_saliency = rr.assign_saliency_to_points(blocks, optimized_s)
        rr.visualize_saliency_with_colorbar(group_number,points_with_saliency) 
        num_clusters = 10
        cluster_labels, linkage_matrix = hierarchical_clustering(optimized_s, num_clusters)
        print("Cluster labels for each tile:", cluster_labels)
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix, labels=np.arange(1, N + 1), leaf_rotation=90, leaf_font_size=10)
        plt.title(f"Hierarchical Clustering Dendrogram (Clusters = {num_clusters})")
        plt.xlabel("Tile Index")
        plt.ylabel("Distance")
        plt.legend([f"Number of Clusters = {num_clusters}"], loc="upper right")
        plt.savefig(f"./tile_result_pictures_2/聚类_{group_number}.png", bbox_inches='tight', pad_inches=0.1)
        len_clusters = max(cluster_labels)
        blocks_result = {}
        blocks_result_second = {}
        blocks_result_later = {}
        for i in range(len_clusters):
            block_name = i
            blocks_result[block_name] = {
                "points": [],
                "colors": [],
                "optimized_s": [],
                "index":[]
            }
        for i in range(len_clusters):
            block_second_name = i
            blocks_result_second[block_second_name] = {
                "points": [],
                "colors": [],
                "optimized_s": [],
                "index":[]
            }
        for i in range(len_clusters):
            block_later_name = i
            blocks_result_later[block_later_name] = {
                "points": [],
                "colors": [],
                "optimized_s": [],
                "index":[]
            }
        for i in range(N):
            block = blocks[keys[i]]
            block_second = blocks_second[keys_second[i]]
            block_later = blocks_later[keys_later[i]]
            points_list = block["points"].tolist()
            color_list = block["colors"].tolist()
            points_list_second = block_second["points"].tolist()
            color_list_second = block_second["colors"].tolist()
            points_list_later = block_later["points"].tolist()
            color_list_later = block_later["colors"].tolist()
            blocks_result[cluster_labels[i] - 1]['optimized_s'].append(optimized_s[i])
            blocks_result[cluster_labels[i] - 1]['index'].append(i)
            blocks_result[cluster_labels[i] - 1]["points"].extend(points_list)
            blocks_result[cluster_labels[i] - 1]["colors"].extend(color_list)
            blocks_result_second[cluster_labels[i] - 1]['optimized_s'].append(optimized_s[i])
            blocks_result_second[cluster_labels[i] - 1]['index'].append(i)
            blocks_result_second[cluster_labels[i] - 1]["points"].extend(points_list_second)
            blocks_result_second[cluster_labels[i] - 1]["colors"].extend(color_list_second)
            blocks_result_later[cluster_labels[i] - 1]['optimized_s'].append(optimized_s[i])
            blocks_result_later[cluster_labels[i] - 1]['index'].append(i)
            blocks_result_later[cluster_labels[i] - 1]["points"].extend(points_list_later)
            blocks_result_later[cluster_labels[i] - 1]["colors"].extend(color_list_later)
        for block_name, block_data in blocks_result.items():
            optimized_s = block_data.get('optimized_s', [])
            mean_optimized_s = np.mean(optimized_s)
            blocks_result[block_name]['optimized_s'] = mean_optimized_s
        for block_name, block_data in blocks_result.items():
            print(f"块 {block_name} 的显著性值为: {block_data['optimized_s']}")
        file_path_save_2 = f"./tile_results/blocks_result_{(group_number - 1) * 3 + 1}.pkl"
        file_path_second_save_2 = f"./tile_results/blocks_result_{(group_number - 1) * 3 + 2}.pkl"
        file_path_later_save_2 = f"./tile_results/blocks_result_{(group_number - 1) * 3 + 3}.pkl"
        with open(file_path_save_2, 'wb') as f:
            pickle.dump(blocks_result, f)
        with open(file_path_second_save_2, 'wb') as f:
            pickle.dump(blocks_result_second, f)
        with open(file_path_later_save_2, 'wb') as f:
            pickle.dump(blocks_result_later, f)
        all_points = []
        all_colors = []
        color_map = [
            [0.75, 0.75, 0.75],  # 灰色
            [1.0, 0.75, 0.75],  # 淡红色
            [0.5, 1.0, 0.5],    # 浅绿色
            [0.0, 1.0, 1.0],    # 青色
            [0.0, 1.0, 0.0],    # 绿色
            [0.0, 0.0, 1.0],    # 蓝色
            [0.5, 0.0, 0.5],    # 紫色
            [1.0, 1.0, 0.0],    # 黄色
            [1.0, 0.0, 0.0],    # 红色
            [0.5, 0.0, 0.0],    # 深红色
        ]
        for block_name, block_data in blocks_result.items():
            points = np.array(block_data["points"])
            optimized_s = block_data.get('optimized_s', 0)
            if points.shape[0] > 0:
                optimized_s = float(optimized_s)
                color_index = int(optimized_s * (10 - 1))  
                color = color_map[color_index]
                block_colors = np.array([color] * points.shape[0])
                all_points.append(points)
                all_colors.append(block_colors)
        if all_points:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
            point_cloud = o3d.geometry.PointCloud()
            sample_size = int(len(all_points) * 0.3)  
            sample_indices = np.random.choice(len(all_points), size=sample_size, replace=False)
            sampled_points = all_points[sample_indices]
            sampled_colors = all_colors[sample_indices]
            point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            point_cloud.colors = o3d.utility.Vector3dVector(sampled_colors)
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(visible=True)  # 设置为不可见
            # vis.add_geometry(point_cloud)
            # vis.poll_events()
            # vis.update_renderer()
            # time.sleep(5)
            # vis.capture_screen_image(f"./tile_result_pictures/tile_result_{group_number}.png")
            # vis.destroy_window()
   
# def generate_blocked_ply():
#     config = load_config('./Hybrid_Saliency_based_Tiling/config.yml')  
#     group_size = config["group_size"]
#     file_folder = config['file_folder']
#     blocks_num = config['blocks_num']
#     file_list = [f for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
#     file_list.sort() 
#     target_folder = config["target_folder"]
#     for i in range(0, len(file_list), group_size):
#         group = file_list[i:i + group_size]
#         group_number = i // group_size + 1
#         file_path_1 = os.path.join(file_folder, group[0])
#         file_path_2 = os.path.join(file_folder, group[-2]) 
#         file_path_3 = os.path.join(file_folder, group[-1])
#         blocks_1 = cut.cut(file_path_1, blocks_num)
#         blocks_2 = cut.cut(file_path_2, blocks_num)
#         blocks_3 = cut.cut(file_path_3, blocks_num)
#         keys = list(blocks_1.keys())
#         json_filename = f'blocks_result_{group_number}.json'
#         json_filepath = os.path.join(target_folder, json_filename)
#         if os.path.exists(json_filepath):
#             with open(json_filepath, 'r') as file:
#                 blocks_data = json.load(file)
#                 optimized_s = blocks_data.get('optimized_s', {})
#                 indices = blocks_data.get('index', {})
#                 total_points = [[], [], []]
#                 total_colors = [[], [], []]
#                 for key in optimized_s:
#                     opt_s_value = optimized_s[key] 
#                     index_value = indices.get(key, [])
#                     combined_points = [[], [], []]
#                     combined_colors = [[], [], []]
#                     for i in range(len(index_value)):
#                         for j, block in enumerate([blocks_1, blocks_2, blocks_3]):
#                            points = block[keys[index_value[i]]]['points']
#                            colors = block[keys[index_value[i]]]['colors']
#                            combined_points[j].extend([point.tolist() for point in points])
#                            combined_colors[j].extend([color.tolist() for color in colors])
#                     for j in range(3):
#                         sampled_points, sampled_colors = downsample_points_and_colors(combined_points[j], combined_colors[j], opt_s_value)
#                         total_points[j].extend(sampled_points)
#                         total_colors[j].extend(sampled_colors)
#                 for j in range(3):
#                     ply_filename = f'redandblack_vox10_{(group_number-1) * 3 + j + 1450}.ply'
#                     ply_filepath = os.path.join('./Hybrid_Saliency_based_Tiling/result_ply/', ply_filename)
#                     save_points_to_ply(total_points[j], total_colors[j], ply_filepath)  
#         else:
#             print(f"{json_filename} does not exist in the target folder.")

# def copy_files(src_folder, dst_folder):
#     if not os.path.exists(src_folder):
#         print(f"源文件夹 {src_folder} 不存在！")
#         return
#     if os.path.exists(dst_folder):
#         shutil.rmtree(dst_folder)
#         print(f"已清空目标文件夹 {dst_folder}")
#     os.makedirs(dst_folder, exist_ok=True)
#     for filename in os.listdir(src_folder):
#         src_file = os.path.join(src_folder, filename)
#         dst_file = os.path.join(dst_folder, filename)
#         if os.path.isfile(src_file):
#             shutil.copy2(src_file, dst_file) 
#     print(f"已完成从 {src_folder} 到 {dst_folder} 的文件拷贝！")

# def copy_files(src_folder, dst_folder):
#     if not os.path.exists(src_folder):
#         print(f"源文件夹 {src_folder} 不存在！")
#         return
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#         for filename in os.listdir(src_folder):
#             src_file = os.path.join(src_folder, filename)
#             dst_file = os.path.join(dst_folder, filename)
#             if os.path.isfile(src_file):
#                  shutil.copy2(src_file, dst_file) 
#     else:
#         print("文件夹已经存在")
#     print(f"已完成从 {src_folder} 到 {dst_folder} 的文件拷贝！")

if __name__ == "__main__":
    print("Start..........") 
    print("Calculating........")
    generate_blocks_file()
    # generate_blocked_ply()
    # src_folder = "./Hybrid_Saliency_based_Tiling/result_ply"
    # dst_folder = "./source_redgirl/origin_girl"
    # copy_files(src_folder, dst_folder)
    print("Down!Check your result")









