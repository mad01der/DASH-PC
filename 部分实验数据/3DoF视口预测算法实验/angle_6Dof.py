import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def compute_angle(v1, v2):
    """计算两个向量之间的夹角（度）"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_turning_angles(df):
    """计算轨迹转向角度"""
    positions = df[['x', 'y', 'z']].values
    angles = []
    for i in range(1, len(positions)-1):
        v1 = positions[i] - positions[i-1]
        v2 = positions[i+1] - positions[i]
        angles.append(compute_angle(v1, v2))
    return angles

def plot_angle_cdf(angles, participant="P01_V1"):
    """绘制单个参与者的角度CDF图"""
    plt.figure(figsize=(10, 6))
    
    angle_counts = Counter(np.round(angles, 2))
    sorted_angles = sorted(angle_counts.keys())
    counts = [angle_counts[a] for a in sorted_angles]
    cdf = np.cumsum(counts) / np.sum(counts)
    
    x_smooth = np.linspace(min(sorted_angles), max(sorted_angles), 300)
    y_smooth = np.interp(x_smooth, sorted_angles, cdf)

    plt.plot(x_smooth, y_smooth, 
            color='#3498db', 
            linewidth=2,
            alpha=1
            )
    
    percentiles = [25, 50, 75, 90]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for p, color in zip(percentiles, colors):
        idx = np.searchsorted(cdf, p/100)
        if idx < len(sorted_angles):
            angle = sorted_angles[idx]
            plt.scatter(angle, p/100, 
                       color=color,
                       s=80,
                       edgecolor='white',
                       linewidth=1,
                       zorder=5,
                       label=f'{p}%: {angle:.1f}°')
            plt.text(sorted_angles[idx] + 11, cdf[idx] - 0.05,  # Reduced offset
                f'{p}%: {sorted_angles[idx]:.2f}',
                ha='center',
                va='bottom',
                fontsize=15,
                color=color,
                bbox=dict(facecolor='white', alpha=0.7, 
                         edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100, 15))
    plt.xlabel('orientation shift', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    # plt.legend(loc='lower right', fontsize=10)
    
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    print(f"\nParticipant {participant}:")
    print(f"  Number of angles: {len(angles)}")
    print(f"  Mean angle: {np.mean(angles):.2f}°")
    print(f"  Median angle: {np.median(angles):.2f}°")
    print(f"  Max angle: {np.max(angles):.2f}°")
    print(f"  Min angle: {np.min(angles):.2f}°")
    
    plt.show()

if __name__ == "__main__":
    # 加载H3_nav数据集
    df = pd.read_csv('../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv')
    
    # 重命名列
    df = df.rename(columns={
        'HMDPX': 'x',
        'HMDPY': 'y',
        'HMDPZ': 'z',
        'HMDRX': 'rx',
        'HMDRY': 'ry',
        'HMDRZ': 'rz'
    })
    
    # 选择P01_V1的数据
    participant_df = df[df['Participant'] == 'P01_V1']
    
    # 计算转向角度
    angles = calculate_turning_angles(participant_df)
    
    # 绘制CDF图
    plot_angle_cdf(angles, "P01_V1")