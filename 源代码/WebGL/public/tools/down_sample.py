import open3d as o3d
import numpy as np
import os

# 设置输入和输出文件夹路径
input_dir = "./ply"
output_dir = "./ply_low"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取输入文件夹中的所有 .ply 文件
ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]

# 设置体素大小进行下采样
voxel_size = 1.8  # 可根据需要调整体素大小

# 遍历文件夹中的所有 .ply 文件进行处理
for ply_file in ply_files:
    input_file = os.path.join(input_dir, ply_file)
    
    # 加载点云文件
    point_cloud = o3d.io.read_point_cloud(input_file)
    print(f"处理文件: {ply_file}, 原始点云包含 {len(point_cloud.points)} 个点")

    # 进行体素下采样
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    print(f"简化后点云包含 {len(downsampled_point_cloud.points)} 个点")

    # 获取点云坐标并四舍五入为整数
    downsampled_points = np.asarray(downsampled_point_cloud.points)

    # 四舍五入并转为整数
    downsampled_points = np.round(downsampled_points)

    # 将四舍五入后的点云坐标赋回点云对象
    downsampled_point_cloud.points = o3d.utility.Vector3dVector(downsampled_points)

    # 保存简化后的点云文件，指定为 ASCII 格式
    output_file = os.path.join(output_dir, ply_file)
    o3d.io.write_point_cloud(output_file, downsampled_point_cloud, write_ascii=True)

    print(f"简化并保存后的点云已保存为 {output_file}")

    # 后处理：修改 PLY 文件头部，将 'float' 转为 'int'，并将数据转换为整数
    def convert_ply_to_int(input_ply, output_ply):
        with open(input_ply, 'r') as f:
            lines = f.readlines()

        # 查找数据开始位置（即 header 后的位置）
        header_end_index = next(i for i, line in enumerate(lines) if line.startswith("end_header"))

        # 修改 header，将 float 转为 int
        for i in range(header_end_index):
            if "property double x" in lines[i]:
                lines[i] = lines[i].replace("double", "float")
            if "property double y" in lines[i]:
                lines[i] = lines[i].replace("double", "float")
            if "property double z" in lines[i]:
                lines[i] = lines[i].replace("double", "float")

        # 将点坐标从 float 转换为 int
        # for i in range(header_end_index + 1, len(lines)):
        #     line = lines[i]
        #     if line.strip():  # 忽略空行
        #         coords = line.split()
        #         coords[0] = str(int(float(coords[0])))  # x 转为 int
        #         coords[1] = str(int(float(coords[1])))  # y 转为 int
        #         coords[2] = str(int(float(coords[2])))  # z 转为 int
        #         lines[i] = " ".join(coords) + "\n"

        # 保存修改后的文件
        with open(output_ply, 'w') as f:
            f.writelines(lines)
        print(f"PLY 文件已保存为 {output_ply}")

    # 调用函数，转换 PLY 文件
    output_file_int = os.path.join(output_dir, ply_file)
    convert_ply_to_int(output_file, output_file_int)

print("所有文件已处理完毕！")
