import numpy as np
import random
import math


def crossover(actions1, actions2):
    """Crossover two paths"""
    # randomly select a crossover point
    i = np.random.randint(1, min(len(actions1), len(actions2)))
    # return the two paths joined at the crossover point
    return actions1[:i] + actions2[i:]

def mutate(actions, mutation_rate=0.05):
    """Mutate a path"""
    # randomly select n postions to mutate
    idxs = random.sample(list(range(len(actions))), k = math.floor(len(actions)/10))
    # randomly select new actions for each position and replace
    for idx in idxs:
        actions[idx] = random.choice([0, 1, 2, 3])
    return actions

#fitness_function = lambda path: abs(path[-1][0] - target[0]) + abs(path[-1][1] - target[1])

def fitness_function(path, wrong_steps, map):
    # check if path contains the target in any position
    if map.target in path:
        # if so, return the position of the first occurence of the target
        return 0
    return abs(path[-1][0] - map.target[0]) + abs(path[-1][1] - map.target[1]) + wrong_steps

