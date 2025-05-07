import numpy as np
import pandas as pd

def compute_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def process_csv_data(file_path):
    df = pd.read_csv(file_path)
    participants = df['Participant'].unique()
    data = {}
    for pid in participants:
        positions = df[df['Participant'] == pid][['HMDPX', 'HMDPY', 'HMDPZ']].values
        data[pid] = [np.array([x, y, z]) for x, y, z in positions]
    return data

def count_angle_occurrences(positions, thresholds=[5, 10, 20, 30], window_size=50):
    results = {threshold: [] for threshold in thresholds}
    total_angles = len(positions) - 2  # 每个夹角由三个连续点组成
    for i in range(total_angles):
        start_j = i
        end_j = min(i + window_size, total_angles)  # 确保不越界
        angles = []
        for j in range(start_j, end_j):
            v1 = positions[j + 1] - positions[j]
            v2 = positions[j + 2] - positions[j + 1]
            angle = compute_angle(v1, v2)
            angles.append(angle)
        for threshold in thresholds:
            count = sum(1 for angle in angles if angle <= threshold)
            results[threshold].append(count)
    return results

def compute_average(results):
    return {threshold: np.mean(counts) for threshold, counts in results.items()}

def print_results(pid, results, averages):
    print(f"\nParticipant: {pid}")
    for threshold, counts in results.items():
        print(f"Threshold {threshold}°:")
        print(f"  Average count: {averages[threshold]:.2f}")

# 示例文件路径，请根据实际情况调整
csv_file_path = '../../dataset/6DoF-HMD-UserNavigationData/NavigationData/H3_nav.csv'
participant_data = process_csv_data(csv_file_path)

for pid, positions in participant_data.items():
    if len(positions) < 3:
        print(f"Participant {pid} has insufficient data points.")
        continue
    results = count_angle_occurrences(positions)
    averages = compute_average(results)
    print_results(pid, results, averages)