import numpy as np
import re

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

def process_vr_data(file_path):
    positions = []  
    pattern = re.compile(r"position: \{ x: (?P<x>-?\d+\.\d+), y: (?P<y>-?\d+\.\d+), z: (?P<z>-?\d+\.\d+) \}")
    with open(file_path, 'r') as file:
        content = file.read()
        matches = pattern.findall(content)
        for match in matches:
            position = {'x': float(match[0]), 'y': float(match[1]), 'z': float(match[2])}
            positions.append(np.array([position['x'], position['y'], position['z']]))
    return positions

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
    averages = {threshold: np.mean(counts) for threshold, counts in results.items()}
    return averages

def print_results(results, averages):
    for threshold, counts in results.items():
        print(f"Threshold {threshold}°:")
        # for i, count in enumerate(counts):
        #     print(f"  Point {i + 1} 后，满足夹角不超过 {threshold}° 的连续向量数: {count}")
        print(f" 平均连续向量数: {averages[threshold]:.2f}\n")

file_path = './location_record/record_1.txt'
positions = process_vr_data(file_path)
results = count_continuous_angles(positions)
averages = compute_average(results)
print_results(results, averages)
