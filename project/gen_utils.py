import numpy as np
import random
import math
from utils import count_loops, count_dead_ends, wrong_actions
from classes import Path, Individual, Map


def crossover(actions1, actions2):
    """Crossover two paths"""
    # randomly select a crossover point
    i = np.random.randint(1, min(len(actions1), len(actions2)))
    # return the two paths joined at the crossover point
    return actions1[:i] + actions2[i:]


def crossover_uniform(actions1, actions2):
    actions = []
    for i in range(len(actions1)):
        if random.random() < 0.5:
            actions.append(actions1[i])
        else:
            actions.append(actions2[i])
    return actions


def mutate(actions, bitmap, mutation_rate=0.15):
    """
    # randomly select n postions to mutate
    idxs = random.sample(list(range(len(actions))), k = math.floor(len(actions)/5))
    # randomly select new actions for each position and replace
    for idx in idxs:
        actions[idx] = random.choice([0, 1, 2, 3])"""
    for i in range(len(actions)):
        if random.random() < mutation_rate + (1 - bitmap[i]) * mutation_rate * 3:
            actions[i] = random.choice([0, 1, 2, 3])
    return actions


# fitness_function = lambda path: abs(path[-1][0] - target[0]) + abs(path[-1][1] - target[1])


def fitness_function(individual: Individual, map: Map):
    bonus = 0
    path: Path = individual.path.copy()

    # check if path contains the target in any position
    if map.target in individual.path.path:
        # if so, return the position of the first occurence of the target
        bonus = 1000  # very high value to make sure that this path is selected
        path.path = path.path[
            : path.path.index(map.target) + 1
        ]  # we are not interested in the moves after the target is reached

    # TODO: (exponential) decay for generation
    if individual.generation < 25:
        dead_ends = -50 * path.dead_ends
        loops = -20 * path.loops
        wrong = -50 * path.wrong_actions
        distance = -2 * (
            abs(path[-1][0] - map.target[0]) + abs(path[-1][1] - map.target[1])
        )
    else:
        dead_ends = -20* path.dead_ends
        loops = -10 * path.loops
        wrong = -20 * path.wrong_actions
        distance = -50 * (
            abs(path[-1][0] - map.target[0]) + abs(path[-1][1] - map.target[1])
        )

    return distance + wrong + loops + dead_ends + bonus
