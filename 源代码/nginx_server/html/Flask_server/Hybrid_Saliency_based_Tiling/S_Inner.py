import tools.FPFH_cal as fpfh
import numpy as np
import math
import os
import random

b = 11
R = 5

def compute_ftfh(s,points):
    ftfh_result = np.zeros(11)
    if(len(points) == 0):
        return np.zeros(11)
    for i in range(len(points)):
        for j in range(10): 
            ftfh_result[j] += s[i][j] 
    for i in range(10):
        ftfh_result[i] /= len(points)
    return ftfh_result

def S_Inner(s,blocks,index):
    keys = list(blocks.keys())
    block_any = blocks[keys[index]]  
    points = np.array(block_any['points'])
    if(len(points) != 0):
         sample_size = max(1, int(len(points) * 0.1))  
         sampled_indices = random.sample(range(len(points)), sample_size)
         points = points[sampled_indices]
    FPFH_result = []
    for i in range(len(points)):
         FPFH_result_temp = fpfh.compute_fpfh(i, points)
         FPFH_result.append(FPFH_result_temp)
    FTFH_result = compute_ftfh(FPFH_result,points)
    file_path = f'./records/record_{s}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        updated = False
        with open(file_path, 'w') as file:
            for line in lines:
                if line.startswith(f"index: {index},"):
                    file.write(f"index: {index}, value: [{', '.join(map(lambda x: f'{x:.2f}', FTFH_result))}]\n")
                    updated = True
                else:
                    file.write(line.strip() + '\n')
            if not updated:
                file.write(f"index: {index}, value: [{', '.join(map(lambda x: f'{x:.2f}', FTFH_result))}]\n")
    else:
        with open(file_path, 'w') as file:
            file.write(f"index: {index}, value: [{', '.join(map(lambda x: f'{x:.2f}', FTFH_result))}]\n")
    sum_all_points = 0  
    for i in range(len(points)):  
        sum_diff = 0 
        for n in range(b):  
            diff = FPFH_result[i][n] - FTFH_result[n]
            sum_diff += diff  
        sum_all_points += sum_diff 
    denominator = R - 1
    try:
        if denominator == 0:
            raise ValueError("R cannot be 1, as it would result in division by zero.")
        value = sum_all_points / denominator
        if value < 0:
            raise ValueError(f"Negative value encountered in sqrt calculation: {value}")
        S_i_I = math.sqrt(value)
    except ValueError as e:
        print(f"❌ Error: {e}")
        S_i_I = 0  
        print(f"✅ S_i_I set to 0 due to error.")
    print(index,"S-Inner-------------",S_i_I)
    return S_i_I