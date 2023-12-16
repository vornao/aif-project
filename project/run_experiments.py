import gym
import minihack
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import IPython.display as display
import os
os.chdir("/Users/vornao/Developer/aif-project/project")

from classes import *
from tqdm import tqdm
from utils import *
import json

from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--max_generations', type=int, default=1000,)
parser.add_argument('--max_individuals', type=int, default=16,)
parser.add_argument('--experiments', type=int, default=100,)
parser.add_argument('--map', type=str, default="irene.des",)

args = parser.parse_args()
# create first generation
MAX_GENERATIONS = args.max_generations
MAX_INDIVIDUALS = args.max_individuals
EXPERIMENTS = args.experiments
MAP_NAME = args.map



# fix the seed for reproducibility (not fixing the seed for the whole program since we have imports!)
random.seed(6667)
np.random.seed(6667)


best_individuals = []

def run_experiment(winners):
    # take seed from dev urandom
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(
        "MiniHack-Navigation-Custom-v0",
        observation_keys=("chars", "pixel"),
        des_file=f"./maps/{MAP_NAME}",
        max_episode_steps=10000,
    )

    state = env.reset()
    game_map = state["chars"]  # type: ignore
    start = get_player_location(game_map)
    target = get_target_location(game_map)
    game_map = Map(game_map, start, target)

    individuals = [
        Individual(random_nactions(300), 1, game_map) for _ in range(MAX_INDIVIDUALS)
    ]
    individuals.sort(key=lambda x: x.fitness, reverse=True)

    for generation in range(MAX_GENERATIONS):
        best_fitness = individuals[0].fitness
        best_individuals.append(individuals[0])
        if generation % 25 == -1:
            print(
                format_loop.format(
                    generation,
                    best_fitness,
                    individuals[0].wrong_actions,
                    individuals[0].loops,
                    individuals[0].dead_ends,
                    individuals[0].distance,
                )
            )

        # take 2 best individuals -> maybe can be replaced with probability distribution based on fitness
        # also roulette wheel selection.
        p1, p2 = individuals[0], individuals[1]
        errors = p1.error_vector + p2.error_vector

        offspring = [
            softmax_mutate(
                crossover(p1.actions, p2.actions), errors, generation=generation
            )
            for _ in range(MAX_INDIVIDUALS)
        ]

        individuals[2:] = [
            Individual(offspring[i], generation + 1, game_map)
            for i in range(MAX_INDIVIDUALS - 2)
        ]

        individuals.sort(key=lambda x: x.fitness, reverse=True)

        if individuals[0].fitness == 0:
            break
    
    best_fitness = individuals[0].fitness
    best_individuals.append(individuals[0])
    winner = {"best_fitness": best_fitness, "generation": generation} # type: ignore
    print(winner)
    winners.append(winner)


if __name__ == '__main__':
    winners_list = []
    with Pool(8) as p:
        # run experiments passing winners_list as argument
        p.map(run_experiment, [winners_list for _ in range(EXPERIMENTS)])

    # export winners as json
    with open(f'winners.json_{MAX_INDIVIDUALS}_{MAP_NAME}', 'w') as f:
        json.dump(winners_list, f)
        
