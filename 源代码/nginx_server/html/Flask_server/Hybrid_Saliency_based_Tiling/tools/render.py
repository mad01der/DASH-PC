import open3d as o3d

file_path = "../source/redandblack_vox10_1450_sample.ply"
point_cloud = o3d.io.read_point_cloud(file_path)

if not point_cloud.is_empty():
    print("Success")
    o3d.visualization.draw_geometries([point_cloud], 
                                      window_name="result",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False)
else:
    print("Error rendering")
