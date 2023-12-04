import numpy as np
import math
import random

from typing import Tuple, List

def get_player_location(game_map: np.ndarray, symbol : str = "@") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0])

# defined by us
def get_player_location1(game_map: np.ndarray, symbol : str = "@") -> 'ciao':
    x, y = np.where(game_map == ord(symbol))
    return (x, y)
#

def get_target_location(game_map: np.ndarray, symbol : str = ">") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0])

def is_wall(position_element: int) -> bool:
    obstacles = "|- "
    return chr(position_element) in obstacles

def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int]) -> List[Tuple[int, int]]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position    
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y-1]):
        valid.append((x, y-1)) 
    # East
    if x + 1 < x_limit and not is_wall(game_map[x+1, y]):
        valid.append((x+1, y)) 
    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y+1]):
        valid.append((x, y+1)) 
    # West
    if x - 1 > 0 and not is_wall(game_map[x-1, y]):
        valid.append((x-1, y))

    return valid

# ---------------------------------------------
# defined by us

def get_valid_actions(game_map: np.ndarray, current_position: Tuple[int, int]) -> List[int]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position    
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y-1]):
        valid.append(0) 
    # East
    if x + 1 < x_limit and not is_wall(game_map[x+1, y]):
        valid.append(1) 
    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y+1]):
        valid.append(2) 
    # West
    if x - 1 > 0 and not is_wall(game_map[x-1, y]):
        valid.append(3)

    return valid

# ---------------------------------------------

"""def direction_from_actions(action: List[Tuple[int, int]]) -> list(int):
    directions = []
    for i in range(len(action)):
        if action[i][0] == 0 and action[i][1] == -1:
            directions.append(0)
        elif action[i][0] == 1 and action[i][1] == 0:
            directions.append(1)
        elif action[i][0] == 0 and action[i][1] == 1:
            directions.append(2)
        elif action[i][0] == -1 and action[i][1] == 0:
            directions.append(3)
    return directions"""
   


# ---------------------------------------------


def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> list[int]:
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3
    }
    actions = []
    x_s, y_s = start
    for (x, y) in path:
        if x_s == x:
            if y_s > y: # we recall that we are in a matrix, therefore going West the column decreases
                actions.append(action_map["W"])
            else: actions.append(action_map["E"])
        elif y_s == y:
            if x_s > x: # we recall that we are in a matrix, therefore going North the row decreases
                actions.append(action_map["N"])
            else: actions.append(action_map["S"])
        else:
            raise Exception("x and y can't change at the same time. oblique moves not allowed!")
        x_s = x
        y_s = y
    
    return actions

# ---------------------------------------------
# defined by us
# NOTE: this function, even if the action is not valid, appends the same position in the list of positions
# this list of position will be the path
# this is why the agent will stay still for a certain number of steps, even if the action changes


def path_from_actions(game_map: np.ndarray, start: Tuple[int, int], actions: List[int]) -> List[Tuple[int, int]]:
    wrong = 0
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3
    }
    path = []
    x, y = start
    for action in actions:
        if action == action_map["N"]:
            if is_wall(game_map[x-1, y]):
                wrong += 1
            else:
                x -= 1
        elif action == action_map["E"]:
            if is_wall(game_map[x, y+1]):
                wrong += 1
            else:
                y += 1
        elif action == action_map["S"]:
            if is_wall(game_map[x+1, y]):
                wrong += 1
            else:
                x += 1
        elif action == action_map["W"]:
            if is_wall(game_map[x, y-1]):
                wrong += 1
            else:
                y -= 1
        else:
            raise Exception("Invalid action!")
        path.append((x, y))
    return path, wrong

# ---------------------------------------------
# path len return the position of the first occurence of the target in the path
# if the target is not in the path, it returns -1
# this way we get the length of the path from the start to the target

def pathlen(path, target):
    # Give the first occurernce of the target in the path
    for idx, pos in enumerate(path):
        if pos == target:
            return idx+1
    return -1


# ---------------------------------------------


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


# ---------------------------------------------
# functions defined by us
# to generate a random path, we need to generate a random sequence of actions
# NOTE: we generate a random sequence of VALID actions, so that the agent will never go through a wall
# we could implement the control in PROLOG

def build_path_rand(parent: List[Tuple[int, int]], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    for i in range(len(parent)):
        path.append(parent[i][0])
    return path


def random_nsteps(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], steps = 100) -> List[Tuple[int, int]]:

    parent = [(start, None)]
    current = start

    for i in range(steps):
        if current == target:
                path = build_path_rand(parent, target)
                return path
        neighbors = get_valid_moves(game_map, current)
        neighbor = neighbors[np.random.randint(0, len(neighbors))]

        parent.append((neighbor, current))
        current = neighbor
    path = build_path_rand(parent, target)
    return path

def random_nactions(actions = 100):
    return random.choices([0, 1, 2, 3], k = actions)