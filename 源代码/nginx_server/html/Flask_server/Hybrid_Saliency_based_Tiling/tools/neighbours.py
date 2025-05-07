
import re

def get_neighbours(index_input,blocks):
    keys = list(blocks.keys())
    if index_input < 0 or index_input >= len(keys):
        return "Invalid dic. number"
    result = []
    if index_input == 0:
        for i in range(1, 3):
            result.append(i)
        result.insert(1,index_input)
    elif index_input == len(keys) - 1:
        for i in range(len(keys) - 3,len(keys) - 1):
            result.append(i)
        result.insert(1,index_input)   
    else:
        for i in range(max(0, index_input - 1), min(len(keys), index_input + 2)):
            key = keys[i]
            result.append(i)
    return  result
