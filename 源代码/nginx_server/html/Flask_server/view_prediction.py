import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import open3d as o3d
import os
import subprocess
import shutil
def train_and_predict(data_x, data_y, data_z): 
    X_train = np.array([[1], [2], [3], [4], [5]])
    model_x = LinearRegression().fit(X_train, data_x)
    model_y = LinearRegression().fit(X_train, data_y)
    model_z = LinearRegression().fit(X_train, data_z)
    X_test =  np.array([[6], [7], [8], [9], [10] , [11] ,[12] ,[13], [14], [15]])
    y_pred_x = np.round(model_x.predict(X_test), 2).tolist()
    y_pred_y = np.round(model_y.predict(X_test), 2).tolist()
    y_pred_z = np.round(model_z.predict(X_test), 2).tolist()
    predictions = [[x, y, z] for x, y, z in zip(y_pred_x, y_pred_y, y_pred_z)]
    return predictions

def plane_equation(A, B):
    x1, y1, z1 = A
    x2, y2, z2 = B
    A_coeff = x2 - x1
    B_coeff = y2 - y1
    C_coeff = z2 - z1
    D = -(A_coeff * x2 + B_coeff * y2 + C_coeff * z2)
    return A_coeff, B_coeff, C_coeff, D

def get_block_side_of_plane(block_center, plane_coeffs):
    x, y, z = block_center
    A_coeff, B_coeff, C_coeff, D = plane_coeffs
    value = A_coeff * x + B_coeff * y + C_coeff * z + D
    return value

def get_points_centers(file_path):
    with open(file_path, 'rb') as f:
        blocks_result = pickle.load(f)
    for block_name, block_data in blocks_result.items():
        points = block_data["points"]
        index = block_data["index"]
        if points:  
            center = np.round([sum(coord) / len(points) * 0.05 for coord in zip(*points)],2) 
        else:
            center = [0, 0, 0] 
        print(f"index {index}: point-center {center}")

def convert_ply_to_int(input_ply, output_ply):
    with open(input_ply, 'r') as f:
        lines = f.readlines()
    header_end_index = next(i for i, line in enumerate(lines) if line.startswith("end_header"))
    for i in range(header_end_index):
        if "property double x" in lines[i]:
            lines[i] = lines[i].replace("double", "float")
        if "property double y" in lines[i]:
            lines[i] = lines[i].replace("double", "float")
        if "property double z" in lines[i]:
            lines[i] = lines[i].replace("double", "float")
    with open(output_ply, 'w') as f:
        f.writelines(lines)

def compress_ply(input_ply_file, output_folder_1, output_folder_2, output_folder_3):
    for folder in [output_folder_1, output_folder_2, output_folder_3]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    filename = os.path.basename(input_ply_file)
    output_drc_filename = filename.replace(".ply", ".drc")
    output_path_1 = os.path.join(output_folder_1, output_drc_filename)
    output_path_2 = os.path.join(output_folder_2, output_drc_filename)
    output_path_3 = os.path.join(output_folder_3, output_drc_filename)
    command = [
        "../../../../../home/exit/Graduate_design/draco/build/draco_encoder",
        "-point_cloud",
        "-i", input_ply_file,
        "-o", output_path_1,
        "-cl", "10",
        "-qp", "10",
    ]
    try:
        subprocess.run(command, check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        shutil.copy(output_path_1, output_path_2)
        shutil.copy(output_path_1, output_path_3)
    except subprocess.CalledProcessError as e:
        print(f"错误信息: {e}")
    except Exception as e:
        print(f"文件复制失败: {e}")

def main(data_x, data_y, data_z, data_number):  
    logs = []
    # 分块信息，然后依据分块信息和预测信息得到重组后的点云文件，再进行传输
    Basic_point = [18, 26, 9] # 物体中心原点，具体的数据需要测量
    predictions = train_and_predict(data_x, data_y, data_z)
    for i in range(len(predictions)):  
        frame = predictions[i]
        A_coeff, B_coeff, C_coeff, D = plane_equation(frame, Basic_point)
        frame_side_value = A_coeff * frame[0] + B_coeff * frame[1] + C_coeff * frame[2] + D
        file_path = f'./Hybrid_Saliency_based_Tiling/tile_results/blocks_result_{data_number[4] - 1450 + i + 2}.pkl' 
        with open(file_path, 'rb') as f:
            blocks_result = pickle.load(f)
        count = 0  
        all_points = []
        all_colors = []
        for block_name, block_data in blocks_result.items():
            points = block_data["points"]
            colors = block_data["colors"]
            index = block_data["index"]
            significance_values = block_data["optimized_s"]
            if isinstance(block_data["optimized_s"], (np.float64, float)):
              average_significance = block_data["optimized_s"]  
            elif isinstance(block_data["optimized_s"], (list, np.ndarray)):
              significance_values = block_data["optimized_s"]
              average_significance = np.mean(significance_values)  
            else:
              average_significance = 0  
            if points:
                center = np.round([sum(coord) / len(points) * 0.05 for coord in zip(*points)], 2)
                axes = ['x', 'y', 'z']
                extremes = {
                   f"{axis}_max": max(points, key=lambda p: p[i])
                   for i, axis in enumerate(axes)
                }
                extremes.update({
                   f"{axis}_min": min(points, key=lambda p: p[i])
                   for i, axis in enumerate(axes)
                })
                plane_params = (A_coeff, B_coeff, C_coeff, D)
                side_values = {
                   key: get_block_side_of_plane(point, plane_params) * 0.05
                   for key, point in extremes.items()
                }
                max_points_condition = any(
                   (s > 0 and frame_side_value > 0) or (s < 0 and frame_side_value < 0)
                   for s in side_values.values()
                )
            else:
                center = [0, 0, 0] 
            side_value = get_block_side_of_plane(center, (A_coeff, B_coeff, C_coeff, D))
            if (side_value > 0 and frame_side_value > 0) or (side_value < 0 and frame_side_value < 0) or max_points_condition or (average_significance > 0.9): 
                all_points.extend(points)
                all_colors.extend(colors)
                count += 1
        log_message = f"the {data_number[4] - 1450 + i + 2} frame's predictions are {predictions[i]} and need to transfer {count} blocks with origin block's count is 10."
        print(log_message)
        logs.append(log_message)  
        points_array = np.array(all_points)
        colors_array = np.array(all_colors)
        if len(points_array) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)
            if colors_array.dtype == np.uint8:
               colors_normalized = colors_array.astype(np.float64) / 255.0
            else:
               colors_normalized = colors_array
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            o3d.io.write_point_cloud(
               f"./view_prediction_transfer/origin/redandblack_vox10_{data_number[4] + i + 1}.ply",
               pcd,
               write_ascii=True 
            )
            convert_ply_to_int(f"./view_prediction_transfer/origin/redandblack_vox10_{data_number[4] + i + 1}.ply",f"./view_prediction_transfer/origin/redandblack_vox10_{data_number[4] + i + 1}.ply")
            compress_ply(f"./view_prediction_transfer/origin/redandblack_vox10_{data_number[4] + i + 1}.ply","./drc_server/source/3/","./drc_server/source/2/","./drc_server/source/1/")
        else:
            print("没有满足条件的点云数据")
    return logs


