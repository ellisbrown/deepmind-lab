import random
import numpy as np
from copy import deepcopy
import os


def get_base_map(n):
    base_map = [["*" for _ in range(n)]]
    for i in range(1, n-1):
        subres = ["*"]
        zeros = [" " for _ in range(n-2)]
        base_map.append(subres + zeros + subres)
    base_map.append(["*" for _ in range(n)])
    return base_map


def main():
    """
    setup: 
    n is length of square
    thinning is prob of object at a point
    pchars is possible chars
    """
    n = 17
    obj_loc = [n//2, n//2]
    objects = [chr(c) for c in range(ord("A"), ord("Z")+1) if chr(c) != 'P'][:20]
    print(objects)
    agent = "P"
    distances = [2]
    base_dir = './vlr_maps_oracle'

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # Create a base map
    base_map = get_base_map(n)

    # Agent spawn locs
    x, y = obj_loc
    agent_locs = []
    for dist in distances:
        agent_locs.append([x - dist, y])
        agent_locs.append([x + dist, y])
        agent_locs.append([x, y + dist])
        agent_locs.append([x, y - dist])
        agent_locs.append([x - dist, y - dist])
        agent_locs.append([x + dist, y - dist])

    map_set = {}
    # Generate map set
    for object in objects:
        map_set[object] = []
        for agent_loc in agent_locs:
            curr_map = deepcopy(base_map)
            x_obj, y_obj = obj_loc
            x_agent, y_agent = agent_loc
            curr_map[x_obj][y_obj] = object
            curr_map[x_agent][y_agent] = agent
            final_map = ""
            for r in curr_map:
                final_map += "".join(r)
                final_map += "\n"
            final_map = final_map[:-1]
            map_set[object].append(final_map)

    for key, map_list in map_set.items():     
        idx = 0   
        for map in map_list:
            file = os.path.join(base_dir, f"vlr_maps_{key}_{idx}.txt")
            text_file = open(file, "w") 
            text_file.write(map)
            text_file.close()
            idx += 1

# 
main()