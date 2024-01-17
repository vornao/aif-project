import gym
import minihack
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import IPython.display as display
import os

#os.chdir("/Users/vornao/Developer/aif-project/project")
os.chdir('/Users/vornao/Developer/aif-project/project')

from classes import *
from tqdm import tqdm
from utils import *
import json

from multiprocessing import Pool, Manager, Lock
from functools import partial
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Run experiments")
parser.add_argument(
    "--max_generations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--individuals",
    type=int,
    default=16,
)
parser.add_argument(
    "--experiments",
    type=int,
    default=100,
)
parser.add_argument(
    "--map",
    type=str,
    default="real_maze.des",
)
parser.add_argument(
    "--workers", 
    type=int, 
    default=6
)
parser.add_argument(
    "--fitness", 
    type=int, 
    default=0
)

args = parser.parse_args()
# create first generation
MAX_GENERATIONS = args.max_generations
MAX_INDIVIDUALS = args.individuals
EXPERIMENTS = args.experiments
MAP_NAME = args.map
WORKERS = args.workers
FITNESS = args.fitness

fitness_name = ''
if FITNESS == 0:
    fitness_name = '0'
elif FITNESS == 1:
    fitness_name = '1'
elif FITNESS == 2:
    fitness_name = '2'

# make directory into results/max_individuals
if not os.path.exists(f"results_{fitness_name}/run_{MAX_INDIVIDUALS}_map_{MAP_NAME.replace('.des', '')}"):
    os.mkdir(f"results_{fitness_name}/run_{MAX_INDIVIDUALS}_map_{MAP_NAME.replace('.des', '')}")

best_individuals = []


def run_experiment(winners, lock):
    # take seed from dev urandom
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)
    fitnesses_list = []
    has_won = False
    first_winning_generation = None

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
        Individual(random_nactions(300), 1, game_map, fitness=FITNESS) for _ in range(MAX_INDIVIDUALS)
    ]
    individuals.sort(key=lambda x: x.fitness, reverse=True)

    for generation in range(MAX_GENERATIONS):
        best_fitness = individuals[0].fitness
        best_individuals.append(individuals[0])

        p1, p2 = individuals[0], individuals[1]
        errors = p1.error_vector + p2.error_vector

        offspring = [
            softmax_mutate(
                crossover(p1.actions, p2.actions), errors, generation=generation
            )
            for _ in range(MAX_INDIVIDUALS)
        ]

        individuals[2:] = [
            Individual(offspring[i], generation + 1, game_map, fitness=FITNESS)
            for i in range(MAX_INDIVIDUALS - 2)
        ]

        if individuals[0].won and not has_won:
            has_won = True
            first_winning_generation = generation

        individuals.sort(key=lambda x: x.fitness, reverse=True)
        fitnesses_list.append(int(individuals[0].fitness))

    best_fitness = individuals[0].fitness

    winner = {
        "best_fitness": best_fitness,
        "generation": int(individuals[0].generation),  # type: ignore
        "wrong_actions": int(individuals[0].wrong_actions),
        "loops": int(individuals[0].loops),
        "dead_ends": int(individuals[0].dead_ends),
        "distance": int(individuals[0].distance),
        "fitnesses": fitnesses_list,
        "first_win": first_winning_generation,
    }

    for k, v in winner.items():
        try:
            winner[k] = int(v)
        except:
            pass


    winners.append(winner)


if __name__ == "__main__":
    with Manager() as manager:
        with Pool(WORKERS) as p:
            winners_list = manager.list()
            partial_run_experiment = partial(run_experiment, winners_list)
            with tqdm(total=EXPERIMENTS, colour='#9244c9') as pbar:
                for _ in p.imap_unordered(partial_run_experiment, range(EXPERIMENTS)):
                    pbar.update()

        # export winners as json
        winners_list = list(winners_list)
        with open(
            f'results_{fitness_name}/run_{MAX_INDIVIDUALS}_map_{MAP_NAME.replace(".des", "")}/stats.csv', "w"
        ) as f:
            # write csv header
            f.write("best_fitness,generation,wrong_actions,loops,dead_ends,distance,first_winner\n")
            for winner in winners_list:
                f.write(
                    f"{winner['best_fitness']},{winner['generation']},{winner['wrong_actions']},{winner['loops']},{winner['dead_ends']},{winner['distance']},{winner['first_win']}\n"
                )

        with open(f'results_{fitness_name}/run_{MAX_INDIVIDUALS}_map_{MAP_NAME.replace(".des", "")}/fitnesses.json', "w") as f:
            # for each winner in winners_list, write fitnesses lists to json
            json.dump([winner["fitnesses"] for winner in winners_list], f)
