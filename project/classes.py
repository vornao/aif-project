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
    def __init__(self, game_map, start, target):
        self.game_map = game_map
        self.start = start
        self.target = target
        # generate a matrix with the same size of the map where each cell is 1 if it is a wall and 0 otherwise
        self.map_matrix = [
            [1 if is_wall(cell) else 0 for cell in row] for row in self.game_map
        ]

    def __str__(self):
        return f"Map: {self.game_map}\nStart: {self.start}\nTarget: {self.target}"
    
    def copy(self):
        return Map(self.game_map.copy(), self.start.copy(), self.target.copy())


class Path:
    def __init__(self, path, game_map: Map):
        if path is None:
            self.path = random_nsteps(game_map.game_map, game_map.start, game_map.target)

        self.path = path
        self.game_map = game_map
        self.actions = actions_from_path(
            self.game_map.start, 
            self.path
        )  # cosa succede se il path ha due posizioni uguali di fila?
        self.loops = count_loops(self.path)
        self.valid_actions_bitmap = valid_actions_bitmap(self.game_map.start, self.path)
        self.wrong_actions = wrong_actions(self.path)
        self.dead_ends = count_dead_ends(self.path)

    def __str__(self):
        return f"Path: {self.path}\nActions: {self.actions}"
    
    def __getitem__(self, index):
        return self.path[index]
    
    def copy(self):
        return Path(self.path.copy(), self.game_map.copy())
    


class Individual:
    def __init__(self, actions, generation: int, game_map: Map):
        if actions is None:
            raise ValueError("actions cannot be None")
        self.actions = actions
        self.generation = generation
        self.game_map = game_map
        self.path = Path(
            path_from_actions(self.game_map.game_map, self.game_map.start, self.actions),
            self.game_map.game_map
        )
        self.fitness = fitness_function(self, self.game_map)
        self.wrong_actions = self.path.wrong_actions

        self.won = self.game_map.target in self.path.path
        self.last_position = self.get_target_index()
        self.distance = self.get_target_distance()

    def get_target_index(self):
        if self.won:
            return self.path.path.index(self.path.target)
        return self.path.path[-1]
    
    def get_target_distance(self):
        if self.won:
            return 0
        return abs(self.last_position[0] - self.path.target[0]) + abs(self.last_position[1] - self.path.target[1])
    
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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(individual: Individual, game_map: Map):
    bonus = 1
    path: Path = individual.path.copy()
    target_index = -301

    # check if path contains the target in any position
    if game_map.target in individual.path.path:
        bonus = 3# we are not interested in the moves after the target is reached
        path.path = path.path[: path.path.index(game_map.target) + 1]
        target_index = -path.path.index(game_map.target)

    # TODO: (exponential) decay for generation
    """
    if individual.generation < 25:
        distance = -1 * (
            abs(path.path[-1][0] - game_map.target[0]) + abs(path.path[-1][1] - game_map.target[1])
        )
        dead_ends = -1 * path.dead_ends if distance > -50 else 0
        loops = -1 * path.loops if distance > -50 else 0
        wrong = -1 * path.wrong_actions if distance > -50 else 0
    """
    #else:
    distance = -5 * (
        abs(path.path[-1][0] - game_map.target[0]) + abs(path.path[-1][1] - game_map.target[1])
    )
    #dead_ends = -1 * path.dead_ends if distance < -10 else 0
    loops = -1 * path.loops #if distance < -10 else 0
    wrong = -1 * path.wrong_actions #if distance < -10 else 0

    return (distance + wrong + loops + target_index)


"""distance = -1 * (
        abs(path.path[-1][0] - game_map.target[0]) + abs(path.path[-1][1] - game_map.target[1])
    )
    dead_ends = -1 * path.dead_ends if distance < -10 else 0
    loops = -1 * path.loops if distance < -10 else 0
    wrong = -1 * path.wrong_actions if distance < -10 else 0

    return (distance + wrong + loops + dead_ends)/bonus"""