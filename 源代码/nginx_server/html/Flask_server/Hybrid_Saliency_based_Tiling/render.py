import open3d as o3d
import time

point_cloud = o3d.io.read_point_cloud("./source/redandblack_vox10_1450.ply")

pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(point_cloud).cuda()

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  
vis.add_geometry(point_cloud)  

start_time = time.time()
for _ in range(100):
    vis.update_geometry(point_cloud)  
    vis.poll_events()  
    vis.update_renderer() 
end_time = time.time()

print(f"渲染 100 次耗时: {end_time - start_time:.4f} 秒")

vis.destroy_window()