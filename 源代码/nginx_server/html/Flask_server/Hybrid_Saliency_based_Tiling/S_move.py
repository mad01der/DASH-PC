import S_external as se
import  numpy as np

def S_move(blocks_form,blocks_later,index):
    keys_form = list(blocks_form.keys())
    keys_later = list(blocks_later.keys())
    points_form = blocks_form[keys_form[index]]['points']
    if(len(points_form) == 0):
        S_D_i = 10   
    else:
        points_later = blocks_later[keys_later[index]]['points']
        v_i = se.cal_average_points(points_later) - se.cal_average_points(points_form)
        S_D_i = np.linalg.norm(v_i)
    print(index,"S-move-------------",S_D_i)
    return S_D_i


