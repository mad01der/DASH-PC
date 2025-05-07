import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H2_nav.csv')
df = df.rename(columns={
    'HMDPX': 'x',
    'HMDPY': 'y', 
    'HMDPZ': 'z',
    'HMDRX': 'rx',
    'HMDRY': 'ry',
    'HMDRZ': 'rz'
})
participant_df = df[df['Participant'] == 'P03_V1'].copy()
def calculate_direction(rx, ry, rz):
    rx = np.radians(rz)
    ry = np.radians(ry)
    rz = np.radians(rx)
    R_x = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = R_z @ R_y @ R_x
    direction = np.array([0, 0, 1])
    return np.dot(R, direction)
participant_df['direction'] = participant_df.apply(
    lambda row: calculate_direction(row['rx'], row['ry'], row['rz']), 
    axis=1
)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
center_point = (0.01, 1.7, -0.04)
ax.plot(participant_df['x'], participant_df['y'], participant_df['z'],
        color='green', linewidth=1, alpha=0.7)
ax.scatter(participant_df['x'], participant_df['y'], participant_df['z'],
           color='blue', s=10, alpha=0.4, label='Position Samples')
# for idx in range(0, len(participant_df), 20):
#     pos = participant_df.iloc[idx]
#     direction = pos['direction']
#     ax.quiver(pos['x'], pos['y'], pos['z'],
#               direction[0], direction[1], direction[2],
#               length=0.1, color='red', 
#               arrow_length_ratio=0.1,
#               linewidth=0.3,
#               alpha=0.6,
#               label='Gaze Direction' if idx == 0 else "")

ax.scatter(*center_point, color='#32CD32', s=150, 
           marker='D', edgecolor='black', 
           label='Reference Center')

ax.set_title('6DoF for Participant P03_V1 Trajectory', 
             fontsize=14, pad=20)
ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.25)

ax.set_box_aspect([1,1,1])  

ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()