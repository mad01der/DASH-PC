
import numpy as np
import tools.neighbours as ng
import tools.gama_cal as ga
import math

R = 5

def cal_average_points(points):
    points = np.array(points)
    if len(points) == 0:
        return np.zeros(3)
    return np.mean(points, axis=0)

def dis(array):
    return np.linalg.norm(array)

def S_external(s,blocks,index):
   array = ng.get_neighbours(index,blocks)
   keys = keys = list(blocks.keys())
   index_1 = array[0]
   index_own = array[1]
   index_2 = array[2]
   blocks_1 = blocks[keys[array[0]]]
   blocks_own = blocks[keys[array[1]]]
   blocks_2 = blocks[keys[array[2]]]
   quantity_1 = ga.gama_cal(s,blocks,index_1,index_own) / (1 + dis(cal_average_points(blocks_1['points']) - cal_average_points(blocks_own['points'])))
   quantity_2 = ga.gama_cal(s,blocks,index_2,index_own) / (1 + dis(cal_average_points(blocks_2['points']) - cal_average_points(blocks_own['points'])))
   e_index = (quantity_1 + quantity_2 ) * (-1 / (R-3))
   result = 1 - math.exp(e_index)
   print(index,"S-external-----------",result)
   return result



