import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List


def test():
    print("test")


# ---------------------------------------------


def random_search(
    game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    # initialize open and close list
    open_list = []
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = 0

    open_list.append((start, starting_state_g))
    support_list[start] = starting_state_g
    parent = {start: None}

    while open_list:
        # get the node with lowest f
        (
            current,
            current_cost,
        ) = open_list.pop()  # il primo parametro è la priorità che ora non ci interessa
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current):
            # check if neighbor in close list, if so continue
            if neighbor in close_list:
                continue  # if the condition is satisfied go back to for
            # compute neighbor g, h and f values
            neighbor_g = 1 + current_cost
            parent[neighbor] = current
            neighbor_entry = (neighbor, neighbor_g)
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                if neighbor_g >= support_list[neighbor]:
                    continue

            # add neighbor to open list and update support_list
            open_list.append(neighbor_entry)
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return None


def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:  # notice that start: None
        print("target", target)
        path.append(target)
        target = parent[target]
    path.reverse()
    return path


# ---------------------------------------------


def build_path_rand(
    parent: List[Tuple[int, int]], target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path = []
    for i in range(len(parent)):
        path.append(parent[i][0])
    return path


# ---------------------------------------------


def true_random_search(
    game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    parent = [(start, None)]

    current = start

    while True:
        if current == target:
            # print("Target found!")
            # print("Path length: ", len(parent))
            # print("Path: ", parent)
            path = build_path_rand(parent, target)
            return path

        neighbors = get_valid_moves(game_map, current)
        # print("neighbors", neighbors)
        neighbor = neighbors[np.random.randint(0, len(neighbors))]
        # print("Neighbor: ", neighbor)

        parent.append((neighbor, current))
        current = neighbor


def true_random_nsteps(
    game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    parent = [(start, None)]

    current = start

    for i in range(100):
        if current == target:
            # print("Target found!")
            # print("Path length: ", len(parent))
            # print("Path: ", parent)
            path = build_path_rand(parent, target)
            return path
        neighbors = get_valid_moves(game_map, current)
        # print("neighbors", neighbors)
        neighbor = neighbors[np.random.randint(0, len(neighbors))]
        # print("Neighbor: ", neighbor)

        parent.append((neighbor, current))
        current = neighbor
    path = build_path_rand(parent, target)
    return path


# ---------------------------------------------
