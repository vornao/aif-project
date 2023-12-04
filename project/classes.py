from utils import random_nsteps, actions_from_path, path_from_actions
from gen_utils import fitness_function

class Map:
    def __init__(self, map, start, target):
        self.map = map
        self.start = start
        self.target = target

    def __str__(self):
        return f'Map: {self.map}\nStart: {self.start}\nTarget: {self.target}'

class Path:
    def __init__(self, path, game_map, start, target):
        if path is None:
            self.path = random_nsteps(game_map, start, target)
        self.path = path
        self.game_map = game_map
        self.start = start
        self.target = target
        self.actions = actions_from_path(self.start, self.path)
    
    def __str__(self):
        return f'Path: {self.path}\nActions: {self.actions}'

class Individual:
    def __init__(self, actions, generation: int, map: Map):
        self.actions = actions
        self.path, self.wrong_actions = path_from_actions(map.map, map.start, self.actions)
        self.fitness = fitness_function(self.path, self.wrong_actions, map)
        self.generation = generation

    def __str__(self):
        return f'{self.path}\nFitness: {self.fitness}\nGeneration: {self.generation}\nWrong actions: {self.wrong_actions}'

    def __repr__(self):
        return self.__str__()

