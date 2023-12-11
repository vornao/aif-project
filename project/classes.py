from utils import (
    random_nsteps,
    actions_from_path,
    wrong_actions,
    path_from_actions,
    is_wall,
    valid_actions_bitmap,
    count_loops,
    count_dead_ends,
)
import numpy as np
import random

"""
- sistemare classe path e utilizzarla al posto di assegnare tutte le cose del path all'individuo
- utilizzare kb per generare solo mosse valide durante la mutazione e/o generazione iniziale
- sistemare fitness
- valutare se fare mutazioni in base anche a loop e _cul de sac_ e vedere implementazione che non penso sia banalissima
"""


class Map:
    def __init__(self, map, start, target):
        self.map = map
        self.start = start
        self.target = target
        # generate a matrix with the same size of the map where each cell is 1 if it is a wall and 0 otherwise
        self.map_matrix = [
            [1 if is_wall(cell) else 0 for cell in row] for row in self.map
        ]

    def __str__(self):
        return f"Map: {self.map}\nStart: {self.start}\nTarget: {self.target}"


class Path:
    def __init__(self, path, game_map, start, target):
        if path is None:
            self.path = random_nsteps(game_map, start, target)

        self.path = path
        self.game_map = game_map
        self.start = start
        self.target = target
        self.actions = actions_from_path(
            self.start, self.path
        )  # cosa succede se il path ha due posizioni uguali di fila?
        self.loops = count_loops(self.path)
        self.valid_actions_bitmap = valid_actions_bitmap(self.start, self.path)
        self.wrong_actions = wrong_actions(self.path)
        self.dead_ends = count_dead_ends(self.path)

    def __str__(self):
        return f"Path: {self.path}\nActions: {self.actions}"
    
    def __getitem__(self, index):
        return self.path[index]
    
    def copy(self):
        return Path(self.path.copy(), self.game_map, self.start, self.target)
    


class Individual:
    def __init__(self, actions, generation: int, game_map: Map):
        if actions is None:
            raise ValueError("actions cannot be None")
        self.actions = actions
        self.generation = generation
        self.path = Path(
            path_from_actions(game_map.map, game_map.start, self.actions),
            game_map.map,
            game_map.start,
            game_map.target,
        )
        self.fitness = fitness_function(self, game_map)
        self.wrong_actions = self.path.wrong_actions

    def __str__(self):
        return f"{self.path}\nFitness: {self.fitness}\nGeneration: {self.generation}\nWrong actions: {self.path.wrong_actions}"

    def __repr__(self):
        return self.__str__()
    

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
    bonus = 1
    path: Path = individual.path.copy()

    # check if path contains the target in any position
    if map.target in individual.path.path:
        # if so, return the position of the first occurence of the target
        bonus=3# we are not interested in the moves after the target is reached

    # TODO: (exponential) decay for generation
    """
    if individual.generation < 25:
        distance = -1 * (
            abs(path.path[-1][0] - map.target[0]) + abs(path.path[-1][1] - map.target[1])
        )
        dead_ends = -1 * path.dead_ends if distance > -50 else 0
        loops = -1 * path.loops if distance > -50 else 0
        wrong = -1 * path.wrong_actions if distance > -50 else 0
    """
    #else:
    distance = -1 * (
        abs(path.path[-1][0] - map.target[0]) + abs(path.path[-1][1] - map.target[1])
    )
    dead_ends = -1 * path.dead_ends if distance < -50 else 0
    loops = -1 * path.loops if distance < -50 else 0
    wrong = -1 * path.wrong_actions if distance < -50 else 0

    return (distance + wrong + loops + dead_ends)/bonus
