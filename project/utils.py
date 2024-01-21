import numpy as np
import math
import random
import matplotlib.pyplot as plt
import IPython.display as display

from tqdm import tqdm
from typing import Tuple, List


# Setting for the print
format_title = "Generation {}, fitness: {}, position: {}, action: {}, wrong actions: {}, loops: {}, dead ends: {}, step: {}/{}"
format_loop = "best_individual in generation {}: fitness: {}, \
wrong actions: {}, \
loops: {}, \
dead_ends: {}, \
distance: {}"


STATIC_MANHATTAN = "static_manhattan"
INFORMED_MANHATTAN = "informed_manhattan"
DYNAMIC_FITNESS = "dynamic_fitness"

match = {
    STATIC_MANHATTAN: 0,
    INFORMED_MANHATTAN: 1,
    DYNAMIC_FITNESS: 2,
}


actions_set = {0, 1, 2, 3}


######  MATH UTILS  #################################################

def exponential_decay(generation, max_generations):
    return np.exp(-generation / max_generations)


def linear_decay(generation, max_generations):
    return (1 - (generation / max_generations)) + 0.1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


def crossover(actions1, actions2):
    """Crossover two paths"""
    # randomly select a crossover point
    i = np.random.randint(1, min(len(actions1), len(actions2)))
    actions = actions1[:i] + actions2[i:]
    # dictionary = {'actions': actions, 'index': i}
    # return the two paths joined at the crossover point
    return actions


def softmax_mutate(
    actions,
    error_vector: np.ndarray,
    mutation_rate=0.8,
    generation=0,
    max_generations=1000,
    decay=True,
) -> List[int]:
    length = len(actions)
    error_vector = np.copy(error_vector)
    num_mutations = np.random.binomial(length, mutation_rate)

    if decay and mutation_rate > 0.1:
        num_mutations = linear_decay(generation, max_generations) * num_mutations

    for _ in range(int(num_mutations)):
        i = np.random.choice(length, p=softmax(error_vector))
        wrong = {actions[i]}
        actions[i] = np.random.choice(list(actions_set - wrong))
        error_vector[i] = 0

    return actions


def random_mutate(
    actions,
    error_vector: np.ndarray,
    mutation_rate=0.8,
    generation=0,
    max_generations=1000,
    decay=True,
):
    length = len(actions)
    num_mutations = np.random.binomial(length, mutation_rate)

    for _ in range(int(num_mutations)):
        i = np.random.randint(0, length)
        actions[i] = np.random.choice(list(actions_set))

    return actions


######  CLASSES  ####################################################

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
    def __init__(
        self, actions: List[int], generation: int, game_map: Map, fitness: int = 0
    ):
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

        if fitness == 0:
            self.fitness: int = fitness_manhattan(self, self.game_map)
        elif fitness == 1:
            self.fitness: int = fitness_function(self, self.game_map)
        elif fitness == 2:
            self.fitness: int = fitness_function_dynamic(self, self.game_map)
        else:
            raise ValueError("fitness function not valid")

    def __get_target_index__(self) -> int:
        if self.won:
            return self.path.index(self.game_map.target)
        return 300  # if target is not in the path return 300, i.e., the maximum number of steps

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
        self.loops = np.sum(self.lb)
        self.dead_ends = np.sum(self.de)
        self.wrong_actions = np.sum(self.va)

        bsum = np.array(sum_bimaps(self.lb, self.va, self.de))

        if self.won:
            bsum[self.target_index:] = 0

        return bsum
    
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


######  GENETIC ALGORITHM  ##########################################
    
def run_genetic(
    env,
    n_individuals: int,
    n_generations: int,
    fitness_type: str,
    mutation_rate=0.8,
    seed=42, # ૮꒰ ˶• ༝ •˶꒱ა ♡
) -> List[Individual]:
    """
    Run the genetic algorithm for the given number of generations, with the given number of individuals.

    :param env: gym environment
    :param n_individuals: number of individuals in the population
    :param n_generations: number of generations
    :param mutation_rate: probability of a mutation
    :param fitness_type: type of fitness function to use [static_manhattan, informed_manhattan, dynamic_manhattan]
    :return: list of best individuals for each generation
    """

    random.seed(seed)
    np.random.seed(seed)

    fit = match[fitness_type]
    state = env.reset()
    game_map = state["chars"]  # type: ignore
    start = get_player_location(game_map)  # type: ignore
    target = get_target_location(game_map)  # type: ignore
    game_map = Map(game_map, start, target)  # type: ignore

    # create first generation
    MAX_GENERATIONS = n_generations
    MAX_INDIVIDUALS = n_individuals

    best_individuals = []

    print("> Creating initial population...")
    """individuals = [
        Individual(starting_actions, 1, game_map) for _ in range(MAX_INDIVIDUALS)
    ]
    """
    individuals = [
        Individual(random_nactions(300), 1, game_map, fitness=fit)
        for _ in range(MAX_INDIVIDUALS)
    ]
    individuals.sort(key=lambda x: x.fitness, reverse=True)
    print("> Evolving...")

    best_fitness = individuals[0].fitness
    best_individuals.append(individuals[0])
    
    with tqdm(total=MAX_GENERATIONS, colour="#9244c9", ncols=150) as pbar:
        for generation in range(MAX_GENERATIONS):
            
            # take 2 best individuals -> maybe can be replaced with probability distribution based on fitness
            # also roulette wheel selection.
            p1, p2 = individuals[0], individuals[1]
            errors = p1.error_vector + p2.error_vector
            offspring = [
                softmax_mutate(
                    crossover(p1.actions, p2.actions),  ##
                    errors,
                    generation=generation,
                    mutation_rate=mutation_rate,
                    max_generations=MAX_GENERATIONS,
                    decay=not (p1.won),
                )
                for _ in range(MAX_INDIVIDUALS)
            ]

            individuals[2:] = [
                Individual(offspring[i], generation + 1, game_map, fitness=fit)
                for i in range(MAX_INDIVIDUALS - 2)
            ]

            individuals.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness = individuals[0].fitness
            best_individuals.append(individuals[0])

            if individuals[0].fitness == 0 and fit == 0:
                print(
                    f"> best individual in generation {individuals[0].generation}: {individuals[0].fitness}, wrong actions: {individuals[0].wrong_actions}"
                )
                break

            pbar.set_postfix(
                best_fitness=best_fitness,
                distance=individuals[0].distance,
                dead_ends=individuals[0].dead_ends,
                loops=individuals[0].loops,
                wrong_actions=individuals[0].wrong_actions,
                refresh=False,
            )

            pbar.update(1)

    return best_individuals


def plot_winner_path(env, game, game_map, best_individuals):
    """
    Plot the path of the best individual.
    :param env: gym environment
    :param game: game map
    :param game_map: game map
    :param best_individuals: list of best individuals for each generation, from run_genetic function
    """

    env.reset()
    plt.rcParams["figure.figsize"] = [15, 7]
    individual = best_individuals[-1]

    image = plt.imshow(game[:, 250:1000], aspect="auto")

    # for generation, path in enumerate(best_paths):
    # plt.title(f"Generation {generation}, fitness: {best_scores[generation]:.2f}, last move: {path[-1]}")
    # start = best_paths[0]
    # path = best_paths[-1]
    actions = []
    actions = best_individuals[-1].actions
    wrong = 0

    for i, action in enumerate(actions):
        try:
            s, _, _, _ = env.step(action)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.title(
                format_title.format(
                    individual.generation,
                    individual.fitness,
                    individual.path[i],
                    action,
                    wrong,
                    individual.loops,
                    individual.dead_ends,
                    i + 1,
                    len(actions),
                )
            )
            image.set_data(s["pixel"][:, 300:950])
            # time.sleep(0.1)
            if individual.path[i] == game_map.target:
                print("YOU WON!")
                break
            if individual.path[i] == best_individuals[-1].path[i - 1]:
                wrong += 1
        except RuntimeError:
            print("YOU WON!")


######  FITNESS FUNCTIONS  ##########################################

def fitness_manhattan(individual: Individual, game_map: Map) -> int:
    return 0 - individual.distance


def fitness_function(individual: Individual, game_map: Map) -> int:
    path: List[Tuple] = individual.path
    length = len(path)
    loops = individual.loops / length
    dead_ends = individual.dead_ends / length
    wrong_actions = individual.wrong_actions / length
    distance = -individual.distance

    return distance - int(10 * loops) - int(10 * dead_ends) - int(10 * wrong_actions)


def fitness_function_dynamic(individual: Individual, game_map: Map) -> int:
    path: List[Tuple] = individual.path
    length = len(path)
    loops = individual.loops / length
    dead_ends = individual.dead_ends / length
    wrong_actions = individual.wrong_actions / length
    distance = -individual.distance
    if game_map.target in path:
        # retrurn the number of steps to reach the target
        return -individual.target_index

    return distance - 300  # we penalize the distance if the target is not reached


######  MiniHack MAP UTILS  #########################################

def get_player_location(
        game_map: np.ndarray, 
        symbol: str = "@"
    ) -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0])


def get_target_location(game_map: np.ndarray, symbol: str = ">") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0])


def is_wall(position_element: int) -> bool:
    obstacles = "|- "
    return chr(position_element) in obstacles


# We can also define 'is_wall' using the KB
def is_wall_kb(position: tuple[int, int], KB) -> bool:
    result = list(
        KB.query(
            f"maze(M), nth1({position[0]+1}, M, Row), nth1({position[1]+1}, Row, Cell)"
            )
        )  # type: ignore
    if result:
        cell_value = result[0]["Cell"]  # type: ignore
        # print(f"Cell value: {cell_value}")
    else:
        raise ("Query result is empty.")  # type: ignore

    return cell_value


def get_valid_moves(
    game_map: np.ndarray, current_position: Tuple[int, int]
) -> List[Tuple[int, int]]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y - 1]):
        valid.append((x, y - 1))
    # East
    if x + 1 < x_limit and not is_wall(game_map[x + 1, y]):
        valid.append((x + 1, y))
    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y + 1]):
        valid.append((x, y + 1))
    # West
    if x - 1 > 0 and not is_wall(game_map[x - 1, y]):
        valid.append((x - 1, y))

    return valid


def get_valid_actions(
    game_map: np.ndarray, current_position: Tuple[int, int]
) -> List[int]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position
    # North
    if y - 1 > 0 and not is_wall(game_map[x - 1, y]):
        valid.append(0)
    # East
    if x + 1 < x_limit and not is_wall(game_map[x, y + 1]):
        valid.append(1)
    # South
    if y + 1 < y_limit and not is_wall(game_map[x + 1, y]):
        valid.append(2)
    # West
    if x - 1 > 0 and not is_wall(game_map[x, y - 1]):
        valid.append(3)

    return valid


def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> list[int]:
    action_map = {"N": 0, "E": 1, "S": 2, "W": 3}
    actions = []
    x_s, y_s = start
    for x, y in path:
        if x_s == x:
            if (
                y_s > y
            ):  # we recall that we are in a matrix, therefore going West the column decreases
                actions.append(action_map["W"])
            else:
                actions.append(action_map["E"])
        elif y_s == y:
            if (
                x_s > x
            ):  # we recall that we are in a matrix, therefore going North the row decreases
                actions.append(action_map["N"])
            else:
                actions.append(action_map["S"])
        else:
            raise Exception(
                "x and y can't change at the same time. oblique moves not allowed!"
            )
        x_s = x
        y_s = y

    return actions


# ---------------------------------------------
# NOTE: this function, even if the action is not valid, appends the same position in the list of positions
# this list of position will be the path
# this is why the agent will stay still for a certain number of steps, even if the action changes


def path_from_actions(
    game_map: np.ndarray, start: Tuple[int, int], actions: List[int]
) -> List[Tuple[int, int]]:
    action_map = {"N": 0, "E": 1, "S": 2, "W": 3}
    path = []
    x, y = start
    for action in actions:
        if action == action_map["N"]:
            if x != 0 and not is_wall(game_map[x - 1, y]):
                x -= 1
        elif action == action_map["E"]:
            if y < game_map.shape[1] and not is_wall(game_map[x, y + 1]):
                y += 1
        elif action == action_map["S"]:
            if x < game_map.shape[0] and not is_wall(game_map[x + 1, y]):
                x += 1
        elif action == action_map["W"]:
            if y != 0 and not is_wall(game_map[x, y - 1]):
                y -= 1
        else:
            raise Exception("Invalid action!")
        path.append((x, y))
    return path


# We can also define 'path_from_actions' using the KB
def path_from_actions_kb(
    game_map: np.ndarray, start: Tuple[int, int], actions: List[int], KB
) -> List[Tuple[int, int]]:
    action_map = {"N": 0, "E": 1, "S": 2, "W": 3}
    path = []
    x, y = start
    for action in actions:
        if action == action_map["N"]:
            if x != 0 and not is_wall_kb((x - 1, y), KB):
                x -= 1
        elif action == action_map["E"]:
            if y < game_map.shape[1] and not is_wall_kb((x, y + 1), KB):
                y += 1
        elif action == action_map["S"]:
            if x < game_map.shape[0] and not is_wall_kb((x + 1, y), KB):
                x += 1
        elif action == action_map["W"]:
            if y != 0 and not is_wall_kb((x, y - 1), KB):
                y -= 1
        else:
            raise Exception("Invalid action!")
        path.append((x, y))
    return path


def wrong_actions(path: List[Tuple[int, int]]) -> int:
    wrong = 0
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            wrong += 1
    return wrong


# ---------------------------------------------
# path len returns the position of the first occurence of the target in the path
# if the target is not in the path, it returns -1
# this way we get the length of the path from the start to the target


def pathlen(path, target):
    # Give the first occurernce of the target in the path
    for idx, pos in enumerate(path):
        if pos == target:
            return idx + 1
    return -1


# ---------------------------------------------
# to generate a random path, we need to generate a random sequence of actions
# NOTE: we generate a random sequence of VALID actions, so that the agent will never go through a wall
# we could implement the control in PROLOG


def build_path_rand(
    parent: List[Tuple[int, int]], target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path = []
    for i in range(len(parent)):
        path.append(parent[i][0])
    return path


def random_nvalid_actions_kb(
    start: Tuple[int, int], target: Tuple[int, int], KB, steps: int = 100
):
    actions = []
    possible_actions = [0, 1, 2, 3]
    current = start
    for i in range(steps):
        query_string = f"findall(Action, is_valid_action({current[0]+1}, {current[1]+1}, Action), Actions), intersection(Actions, {possible_actions}, ValidActions)"
        results = list(KB.query(query_string))
        valid_actions = results[0]["ValidActions"]  # type: ignore
        action = valid_actions[np.random.randint(0, len(valid_actions))]  # type: ignore
        actions.append(action)
        current = apply_action[action](current[0], current[1])
    return actions


apply_action = {
    0: lambda x, y: (x - 1, y),
    1: lambda x, y: (x, y + 1),
    2: lambda x, y: (x + 1, y),
    3: lambda x, y: (x, y - 1),
}


def random_nsteps(
    game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], steps=100
) -> List[Tuple[int, int]]:
    parent = [(start, None)]
    current = start

    for i in range(steps):
        if current == target:
            path = build_path_rand(parent, target)  # type: ignore
            return path
        neighbors = get_valid_moves(game_map, current)
        neighbor = neighbors[np.random.randint(0, len(neighbors))]
        parent.append((neighbor, current))  # type: ignore
        current = neighbor
    path = build_path_rand(parent, target)  # type: ignore
    return path[1:]


# TODO: write 'random_nsteps' using the KB


def random_nactions(actions=100):
    return random.choices([0, 1, 2, 3], k=actions)


# ---------------------------------------------


def count_loops(path: List[Tuple[int, int]]):
    loops = 0
    for i in range(1, len(path) - 1):
        window = path[i - 1 : i + 2]
        loops += window[0] == window[2]
    return loops


def is_loop(path: List[Tuple[int, int]], index: int):
    window = path[index - 1 : index + 2]
    return window[0] == window[2]


# Here we check if the element in index is in the previous locations:
# NB. we notice that the situation in which you go through a wall is a generic_loop
def is_generic_loop(path, index):
    current_location = path[index]
    previous_locations = path[: index - 1]
    return current_location in previous_locations

def is_k_loop(path, index, k):
    current_location = path[index]
    previous_locations = path[index - k: index]
    return current_location in previous_locations


def count_dead_ends(game_map: np.ndarray, path: List[Tuple[int, int]]):
    dead_ends = 0
    for i in range(1, len(path) - 1):
        if is_dead_end(game_map, path[i]):
            dead_ends += 1
    return dead_ends


def is_dead_end(game_map: np.ndarray, position: Tuple[int, int]):
    # check if the only valid action is path[index - 1]
    if len(get_valid_actions(game_map, position)) == 1:
        return True
    else:
        return False
    

def delete_loops(actions, index):
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


######  BITMAPS TO COMPUTE ERRORS  ##################################

def valid_actions_bitmap(start, path):
    prev = path[0]
    bitmap = [0 if prev == start else 1]

    for i in path[1:]:
        if prev == i:  
            bitmap.append(1)
        else:
            bitmap.append(0)
        prev = i
    return bitmap


def loops_bitmap(path):
    bitmap = [0] * len(path)
    k = 3
    for i in range(1, len(path) - 1):
        if is_k_loop(path, i, k):
            bitmap[i] = 1
    return bitmap


def dead_ends_bitmap(game_map, path):
    bitmap = [0] * len(path)
    for i in range(1, len(path) - 1):
        if is_dead_end(game_map, path[i]):
            bitmap[i] = 1
    return bitmap


def sum_bimaps(*args):
    """Return the bitwise sum of the bitmaps"""
    return [sum(x) for x in zip(*args)]