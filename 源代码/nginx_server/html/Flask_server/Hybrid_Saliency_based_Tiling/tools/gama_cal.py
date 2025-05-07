import numpy as np
import tools.FTFH_cal as ftfh
import os
def read_record_from_file(record_file, index):
    if not os.path.exists(record_file):
        return None
    with open(record_file, 'r') as file:
        for line in file:
            parts = line.strip().split(", value: ")
            if int(parts[0].split(":")[1].strip()) == index:
                value = np.fromstring(parts[1].strip("[]"), sep=" ")
                return value
    return None

def kafan_cal(s,blocks, index_1, index_2):
    keys = list(blocks.keys())
    record_file=f'./records/record_{s}.txt'
    result_1 = read_record_from_file(record_file, index_1)
    result_2 = read_record_from_file(record_file, index_2)
    if result_1 is None or result_2 is None:
        points_1 = np.array(blocks[keys[index_1]]['points'])
        points_2 = np.array(blocks[keys[index_2]]['points'])
        result_1 = ftfh.compute_ftfh(points_1)
        result_2 = ftfh.compute_ftfh(points_2)
    if len(result_1) != len(result_2):
        print(result_1)
        print(result_2)
        raise ValueError("Result arrays must have the same length")
    
    if np.all(result_1 == 0) and np.all(result_2 == 0):
        return 0

    b = 11
    result_1 = np.array(result_1[:b])
    result_2 = np.array(result_2[:b])
    numerator = (result_1 - result_2) ** 2
    denominator = result_1 + result_2
    chi_square_distance = np.sum(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0))
    return chi_square_distance 


def Yuma_cal(colors_1, colors_2):
    colors_1 = np.array(colors_1)
    colors_2 = np.array(colors_2)
    Yuma_single_1 = np.dot(colors_1, [0.299, 0.587, 0.114]) 
    Yuma_single_2 = np.dot(colors_2, [0.299, 0.587, 0.114])  
    Yuma_sum_1 = np.sum(Yuma_single_1) / 3
    Yuma_sum_2 = np.sum(Yuma_single_2) / 3
    return abs(Yuma_sum_1 - Yuma_sum_2)


def gama_cal(s,blocks,index_1, index_2):
    keys = keys = list(blocks.keys())
    colors_1 = np.array(blocks[keys[index_1]]['colors'])
    colors_2 = np.array(blocks[keys[index_2]]['colors'])
    kafan_result = kafan_cal(s,blocks,index_1, index_2)   
    Yuma_result = Yuma_cal(colors_1, colors_2)
    return kafan_result + 0.35 * Yuma_result

