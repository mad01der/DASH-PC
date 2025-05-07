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

def process_vr_data(csv_path):
    df = pd.read_csv(csv_path)
    positions = df[['x', 'y', 'z']].values
    return [np.array(pos) for pos in positions]

def count_continuous_angles(positions, thresholds=[5, 10, 20, 30], max_continuous=50):
    results = {threshold: [] for threshold in thresholds}
    
    for i in range(len(positions) - 2):
        v1 = positions[i]  
        count_dict = {threshold: 0 for threshold in thresholds}
        
        for threshold in thresholds:
            count = 0
            for j in range(i + 1, min(i + 1 + max_continuous, len(positions) - 1)):
                v2 = positions[j]
                angle = compute_angle(v1, v2)
                if angle <= threshold:
                    count += 1
                else:
                    break
            count_dict[threshold] = count
        
        for threshold in thresholds:
            results[threshold].append(count_dict[threshold])
    
    return results

def compute_average(results):
    return {threshold: np.mean(counts) for threshold, counts in results.items()}

def print_results(results, averages):
    for threshold, counts in results.items():
        print(f"Threshold {threshold}°:")
        print(f"  Average continuous vectors: {averages[threshold]:.2f}\n")

# 使用示例
csv_path = '../dataset/dataset_6DoF/interpolated/User1.csv'  
positions = process_vr_data(csv_path)
results = count_continuous_angles(positions)
averages = compute_average(results)
print_results(results, averages)