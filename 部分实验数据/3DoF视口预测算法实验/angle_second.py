import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_angle(v1, v2):
    """计算两个向量之间的夹角（角度）"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(dot/norm, -1.0, 1.0)))

df = pd.read_csv('../../dataset/dataset_6DoF/interpolated/User1.csv')  
positions = df[['x', 'y', 'z']].values

angles = [compute_angle(positions[i]-positions[i-1], positions[i+1]-positions[i]) 
          for i in range(1, len(positions)-1)]
sorted_angles = np.sort(angles)
cdf = np.arange(1, len(sorted_angles)+1) / len(sorted_angles)
plt.xlim(0, 180) 
plt.figure(figsize=(8,5))
plt.plot(sorted_angles, cdf, linewidth=2)
plt.xlabel('Turning Angle (degrees)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('CDF of Turning Angles', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()