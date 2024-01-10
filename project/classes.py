from utils import (
    random_nsteps,
    actions_from_path,
    wrong_actions,
    path_from_actions,
    is_wall,
    valid_actions_bitmap,
    loops_bitmap,
    dead_ends_bitmap,
    sum_bimaps,
    count_loops,
    count_dead_ends,
    softmax,
)
import numpy as np
import random

from typing import List, Tuple

"""
- utilizzare kb per generare solo mosse valide durante la mutazione e/o generazione iniziale
- sistemare fitness
- valutare se fare mutazioni in base anche a loop e _cul de sac_ e vedere implementazione che non penso sia banalissima
"""


def exponential_decay(generation, max_generations):
    return np.exp(-generation / max_generations)


class Map:
    def __init__(
        self, game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]
    ):
        self.layout: np.ndarray = game_map
        self.start: Tuple[int, int] = start
        self.target: Tuple[int, int] = target
        self.map_matrix = [
            [1 if is_wall(cell) else 0 for cell in row] for row in self.layout
        ]

    def __str__(self):
        return f"Map: {self.layout}\nStart: {self.start}\nTarget: {self.target}"

    def copy(self):
        return Map(self.layout.copy(), self.start, self.target)


class Individual:
    def __init__(self, actions: List[int], generation: int, game_map: Map):
        self.generation: int = generation
        self.actions: List[int] = actions
        self.game_map: Map = game_map

        # create the path from the actions
        self.path: List[Tuple[int, int]] = path_from_actions(
            game_map.layout, game_map.start, actions
        )

        self.won = self.game_map.target in self.path
        self.__check_init_params__()

        self.target_index = self.__get_target_index__()
        self.distance = self.__get_target_distance__()
        self.error_vector = self.__get_error_vector__()
        self.loops = count_loops(self.path)
        self.dead_ends = count_dead_ends(
            self.game_map.layout,
            self.path,
        )
        self.wrong_actions = wrong_actions(self.path)
        self.fitness: int = fitness_function(self, self.game_map)

    def __get_target_index__(self) -> int:
        if self.won:
            return self.path.index(self.game_map.target)
        return -1  # if target is not in the path return -1

    def __get_target_distance__(self) -> int:
        if self.won:
            return 0
        return abs(self.path[-1][0] - self.game_map.target[0]) + abs(
            self.path[-1][1] - self.game_map.target[1]
        )

    def __get_error_vector__(self):
        self.lb = loops_bitmap(self.path)
        self.va = valid_actions_bitmap(self.game_map.start, self.path)
        self.de = dead_ends_bitmap(self.game_map.layout, self.path)
        return np.array(sum_bimaps(self.lb, self.va, self.de))

    def __check_init_params__(self):
        if self.actions is None:
            raise ValueError("actions cannot be None")
        if self.path is None:
            raise ValueError("path cannot be None")
        if self.generation < 0 or self.generation is None:
            raise ValueError("generation index not valid")

    def __str__(self):
        return f"> Path: {self.path}\n> Fitness: {self.fitness}\n> Generation: {self.generation}\n> Wrong actions: {self.wrong_actions}"

    def __repr__(self):
        return self.__str__()

def crossover(actions1, actions2):
    """Crossover two paths"""
    # randomly select a crossover point
    i = np.random.randint(1, min(len(actions1), len(actions2)))
    actions = actions1[:i] + actions2[i:]
    # dictionary = {'actions': actions, 'index': i}
    # return the two paths joined at the crossover point
    return actions

def softmax_mutate(actions, error_vector: np.ndarray, mutation_rate=0.8, generation=0) -> List[int]:
    length = len(actions)
    error_vector = np.copy(error_vector)
    num_mutations = np.random.binomial(length, mutation_rate)
    num_mutations = exponential_decay(generation, 100) * num_mutations

    for _ in range(int(num_mutations)):
        i = np.random.choice(length, p=softmax(error_vector))
        actions[i] = np.random.choice([0, 1, 2, 3])
        error_vector[i] = 0

    return actions


def _softmax_mutate(actions, error_vector: np.ndarray, wrong_action_bitmap, mutation_rate=0.8, generation=0):
    length = len(actions)
    error_vector = np.copy(error_vector)
    wrong_actions_to_mutate = np.zeros(length)
    num_mutations = np.random.binomial(length, mutation_rate)
    #num_mutations = exponential_decay(generation, 100) * num_mutations

    for _ in range(int(num_mutations)):
        i = np.random.choice(length, p=softmax(error_vector))
        if delete_wrong_actions(actions, i, wrong_action_bitmap):
            wrong_actions_to_mutate[i] = 1
        else:
            actions[i] = np.random.choice([0, 1, 2, 3])
        error_vector[i] = 0 # cannot be mutated again

    number_of_wrong_actions = len(wrong_actions_to_mutate.nonzero()[0])
    for i in np.flip(wrong_actions_to_mutate.nonzero()[0]): # This is mindblowing!!!!!!
        actions = np.delete(actions, i)
    for _ in range(number_of_wrong_actions):
        actions = np.append(actions, np.random.choice([0, 1, 2, 3]))
    return list(actions)


def _mutate(actions, bitmap, mutation_rate=0.5):
    """
    # randomly select n postions to mutate
    idxs = random.sample(list(range(len(actions))), k = math.floor(len(actions)/5))
    # randomly select new actions for each position and replace
    for idx in idxs:
        actions[idx] = random.choice([0, 1, 2, 3])"""
    length = len(actions)
    for i in range(len(actions)):
        if random.random() < mutation_rate + (1 - bitmap[i]) * mutation_rate * 3:
            if i <= len(actions) - 2:
                actions = delete_loops(actions, i)
            if len(actions) < length:  # si è tolto un loop
                actions += random.choices(
                    [0, 1, 2, 3], k=2
                )  # aggiungo due azioni random (vedi Attenzione sotto)
            else:
                actions[i] = random.choice(
                    [0, 1, 2, 3]
                )  # se non si è tolto un loop mutazione "normale"
    return actions


# Per il momento leviamo solo i loops "banali"
# Attenzione: se elimina un loop restituisce actions lungo 2 in meno
# TODO: fare in modo che questa funzione si usa se a generazione n ci sono questi loops "banali", allora
# a generazione n+1 non ci sono più
def delete_loops(actions, index):
    # check if there is a situation like north, south, north or east, west, east (and viceversa)
    if (
        actions[index] == actions[index + 2]
        and actions[index + 1] == (2 + actions[index]) % 4
    ):
        actions = actions[:index] + actions[index + 2 :]
    return actions


def delete_wrong_actions(actions, index, bitmap):
    """NB. small loops are in this category too"""
    if bitmap[index]: 
        actions = actions[:index] + actions[index + 1 :]
        return True
    return False


def fitness_function(individual: Individual, game_map: Map) -> int:
    path: List[Tuple] = individual.path
    length = len(path)
    loops = individual.loops / length
    dead_ends = individual.dead_ends / length
    wrong_actions = individual.wrong_actions / length
    distance = -individual.distance
    if game_map.target in path:
        return distance - int(10 * loops) - int(10 * dead_ends) - int(10 * wrong_actions) + 100 # sum a bonus

    return distance - int(10 * loops) - int(10 * dead_ends) - int(10 * wrong_actions)
