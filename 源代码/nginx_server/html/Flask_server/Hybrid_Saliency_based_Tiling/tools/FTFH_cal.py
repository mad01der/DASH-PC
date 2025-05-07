import numpy as np 
import tools.FPFH_cal as fpfh

def compute_ftfh(points):
    ftfh_result = np.zeros(11)
    if(len(points) == 0):
        return np.zeros(11)
    for i in range(len(points)):
        for j in range(10): 
            ftfh_result[j] += fpfh.compute_fpfh(i, points)[j]
    for i in range(10):
        ftfh_result[i] /= len(points)
    return ftfh_result
