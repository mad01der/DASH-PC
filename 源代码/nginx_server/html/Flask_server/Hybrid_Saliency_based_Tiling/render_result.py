import numpy as np
import open3d as o3d  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Loss as loss
import S_Inner as si
import S_external as se
import S_move as sm
import time as time

def assign_saliency_to_points(blocks, optimized_s):
    points_with_saliency = []
    for i, key in enumerate(blocks.keys()):
        block_points = blocks[key]['points']
        saliency = np.full((block_points.shape[0], 1), optimized_s[i])  
        points_with_saliency.append(np.hstack((block_points, saliency)))
    return np.vstack(points_with_saliency)

def visualize_saliency_with_colorbar(s,points_with_saliency):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_with_saliency[:, :3])
    saliency = points_with_saliency[:, 3]
    saliency_normalized = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency_mapped = np.power(saliency_normalized, 1)
    colormap = cm.get_cmap('coolwarm')
    colors = colormap(saliency_mapped)[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) 
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"./tile_result_pictures/render_result_{s}.png") 
    vis.destroy_window() 
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter([], [], c=[], cmap=colormap, vmin=saliency.min(), vmax=saliency.max())
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Saliency', rotation=270, labelpad=15)
    plt.savefig("./tile_result_pictures/colorbar.png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig) 
    
def get_optimized_s(s,N,blocks,blocks_later):
   print("waiting.......")
   inner_dispersion = []
   inter_frame_variability = []
   exterior_rarity = []
   # the value later is used for quick render and the value above is used for the whole process testing
#    inner_dispersion = [1.87, 8.41, 0.0, 9.45, 16.34, 15.47, 0.0, 8.49, 10.04, 7.31, 14.97, 11.72, 23.9, 16.96, 19.74, 0.0, 14.14, 15.23, 0.0, 18.69, 10.09, 3.28, 15.12, 18.92, 0.0, 0.0, 10.1]
#    exterior_rarity = [0.0389, 0.0537, 0.0629, 0.0973, 0.0606, 0.0492, 0.058, 0.0461, 0.0331, 0.0896, 0.0975, 0.1609, 0.3626, 0.2763, 0.1073, 0.1306, 0.0922, 0.0873, 0.157, 0.2178, 0.1614, 0.0987, 0.1367, 0.1389, 0.0621, 0.0167, 0.0332]
#    inter_frame_variability = [60.87, 11.85, 10, 20.21, 4.41, 6.11, 0.0, 6.82, 28.51, 54.98, 11.71, 26.87, 6.86, 5.17, 6.8, 0.0, 5.17, 13.28, 10, 18.32, 16.24, 39.16, 8.74, 7.87, 0.0, 10, 26.85]  
   start_time = time.time()
   for i in range(N):
       inner_dispersion.append(round(si.S_Inner(s,blocks,i),2))
       inter_frame_variability.append(round(sm.S_move(blocks,blocks_later,i),4))
   if(len(inner_dispersion) == N):
       for i in range(N):
           exterior_rarity.append(round(se.S_external(s,blocks,i),2))
   end_time = time.time()
   time_long = end_time - start_time
   print("time_long is",time_long)  
   optimized_s = loss.optimize_s_with_adam(N,blocks,inner_dispersion, exterior_rarity, inter_frame_variability, sigma=0.02, lr=0.01, epochs=50)
   return optimized_s

   








