from heapq import heappush, heappop  # Recommended.
import numpy as np
import operator
import time

from collections     import defaultdict
from flightsim.world import World

import sys
sys.path.append('/home/josh/Desktop/meam620/project3/proj3/code/')

from .occupancy_map import OccupancyMap # Recommended.

# def graph_search(world, resolution, margin, start, goal, astar):
#     """
#     Parameters:
#         world,      World object representing the environment obstacles
#         resolution, xyz resolution in meters for an occupancy map, shape=(3,)
#         margin,     minimum allowed distance in meters from path to obstacles.
#         start,      xyz position in meters, shape=(3,)
#         goal,       xyz position in meters, shape=(3,)
#         astar,      if True use A*, else use Dijkstra
#     Output:
#         return a tuple (path, nodes_expanded)
#         path,       xyz position coordinates along the path in meters with
#                     shape=(N,3). These are typically the centers of visited
#                     voxels of an occupancy map. The first point must be the
#                     start and the last point must be the goal. If no path
#                     exists, return None.
#         nodes_expanded, the number of nodes that have been expanded
#     """

#     # While not required, we have provided an occupancy map you may use or modify.
#     occ_map = OccupancyMap(world, resolution, margin)
#     # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
#     start_index = tuple(occ_map.metric_to_index(start))
#     goal_index = tuple(occ_map.metric_to_index(goal))

#     # Return a tuple (path, nodes_expanded)
#     return None, 0


def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_neighbours(node, occ_map, visited_nodes):
    neighbours =  np.array([[-1, -1, -1],[-1, -1,  0],[-1, -1,  1],
                            [-1,  0, -1],[-1,  0,  0],[-1,  0,  1],
                            [-1,  1, -1],[-1,  1,  0],[-1,  1,  1],
                            [ 0, -1, -1],[ 0, -1,  0],[ 0, -1,  1],
                            [ 0,  0, -1],             [ 0,  0,  1],
                            [ 0,  1, -1],[ 0,  1,  0],[ 0,  1,  1],
                            [ 1, -1, -1],[ 1, -1,  0],[ 1, -1,  1],
                            [ 1,  0, -1],[ 1,  0,  0],[ 1,  0,  1],
                            [ 1,  1, -1],[ 1,  1,  0],[ 1,  1,  1]])

    true_neighbours = node + neighbours

    #Valid neighbours that do not lie inside any obstacle or outside the graph              #Lol slower
    # time3 = time.time()
    # occupied   = list(map(occ_map.is_occupied_index, true_neighbours))
    # unoccupied = list(map(operator.not_, occupied))
    # valid_true_neighbours = true_neighbours[np.where(unoccupied)]
    # valid_true_neighbours = tuple(map(tuple, valid_true_neighbours))
    # time4 = time.time()
    # print(time4-time3)


    # time5 = time.time()
    valid_true_neighbours = true_neighbours[np.all(np.logical_and(true_neighbours >= 0, true_neighbours < occ_map.map.shape), axis=1), :]
    valid_true_neighbours = valid_true_neighbours[occ_map.map[valid_true_neighbours[:, 0], valid_true_neighbours[:, 1], valid_true_neighbours[:, 2]] == False]
    # valid_true_neighbours = [tuple(i) for i in valid_true_neighbours if set(list(i)) not in visited_nodes]
    # time6 = time.time()

    # print(time6-time5)

    return valid_true_neighbours

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index  = tuple(occ_map.metric_to_index(goal))

    from_start              = defaultdict(lambda: float("inf"))
    from_start[start_index] = get_distance(start_index, goal_index)
    nodes_expanded = 0
    parents_dict   = {}
    visited_nodes  = set()
    Q_list         = []
    heappush(Q_list, (0, start_index))      #Current Cost, Current Node

    while Q_list:
        cost, node = heappop(Q_list)
        visited_nodes.add(node)
        nodes_expanded += 1

        if node == goal_index:
            #We found our path!!
            start_to_end_path = [goal]
            back_prop         = goal_index
            
            while back_prop is not None:
                #if parent is the starting point then path found
                if parents_dict[back_prop] == start_index:
                    start_to_end_path.append(start)
                    # print(start_index)
                    # print(start)
                    break
                
                start_to_end_path.append(occ_map.index_to_metric_center(parents_dict[back_prop]))
                back_prop = parents_dict[back_prop]
            
            start_to_end_path.reverse()
            # print(np.stack(start_to_end_path[:]))
            return np.stack(start_to_end_path[:]), nodes_expanded

        for neighbour in get_neighbours(node, occ_map, visited_nodes):
            
            #If a neighbour has already been visited then skip
            neighbour = tuple(neighbour)
            if neighbour not in visited_nodes:
                # continue

                distance_between_points = get_distance(node, neighbour)
                distance_from_start     = from_start[node] + distance_between_points
                
                if astar == True:
                    #The difference between Dijkstra and Astar is just the heuristic cost
                    distance_between_pt_goal = get_distance(goal_index, neighbour)
                    heuristic                = distance_from_start + distance_between_pt_goal

                    if distance_from_start < from_start[neighbour]:
                        from_start[neighbour]   = distance_from_start
                        parents_dict[neighbour] = node
                        heappush(Q_list, (heuristic, neighbour))    #Cost, Node
                    
                elif astar == False:
                    if distance_from_start < from_start[neighbour]:
                        from_start[neighbour]   = distance_from_start
                        parents_dict[neighbour] = node
                        heappush(Q_list, (distance_from_start, neighbour)) #Cost, Node
    
    # Return a tuple (path, nodes_expanded)
    return None, nodes_expanded