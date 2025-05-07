import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Dict

def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def load_vr_data(file_path: str) -> pd.DataFrame:
    pattern = re.compile(
        r"position: \{ x: (?P<x>-?\d+\.\d+), y: (?P<y>-?\d+\.\d+), z: (?P<z>-?\d+\.\d+) \},"
        r"\s+rotation: \{ x: (?P<rx>-?\d+\.\d+), y: (?P<ry>-?\d+\.\d+), z: (?P<rz>-?\d+\.\d+) \}\s+\}"
    )
    
    data = []
    with open(file_path, "r") as file:
        for match in pattern.finditer(file.read()):
            data.append({
                "position": {
                    "x": float(match.group("x")),
                    "y": float(match.group("y")),
                    "z": float(match.group("z"))
                },
                "rotation": {
                    "x": float(match.group("rx")),
                    "y": float(match.group("ry")),
                    "z": float(match.group("rz"))
                }
            })
    
    return pd.DataFrame([{
        'x': d['position']['x'],
        'y': d['position']['y'],  
        'z': d['position']['z'],
        'rx': d['rotation']['x'], 
        'ry': d['rotation']['y'],
        'rz': d['rotation']['z']
    } for d in data])

def plot_turning_angle_cdf(angles: List[float]):
    """绘制转向角度CDF图"""
    sorted_angles = np.sort(angles)
    cdf = np.arange(1, len(sorted_angles)+1) / len(sorted_angles)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(sorted_angles, cdf, 
             color='#1f77b4',
             linewidth=2.5,
             marker='o',
             markersize=4,
             markerfacecolor='white',
             markeredgewidth=1,
             alpha=0.8)
    
    # 添加关键百分位标记
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        idx = int(len(sorted_angles) * p / 100)
        plt.scatter(sorted_angles[idx], cdf[idx], 
                   color='red', 
                   s=60,
                   zorder=5)
        plt.text(sorted_angles[idx], cdf[idx]+0.03, 
                f'{p}%: {sorted_angles[idx]:.1f}°',
                ha='center',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.xlabel('Turning Angle (degrees)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('CDF of VR Headset Turning Angles', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 180)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 181, 30))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 加载数据
    file_path = "../location_record/record_1.txt"
    df = load_vr_data(file_path)
    
    # 将位置数据转换为numpy数组
    positions = df[['x', 'y', 'z']].to_numpy()
    
    # 计算转向角度（使用滑动窗口）
    angles = []
    for i in range(1, len(positions)-1):
        v1 = positions[i] - positions[i-1]  # 前一向量
        v2 = positions[i+1] - positions[i]  # 后一向量
        angles.append(compute_angle(v1, v2))
    
    # 可视化
    plot_turning_angle_cdf(angles)