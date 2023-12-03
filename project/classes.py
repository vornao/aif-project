from utils import true_random_nsteps, actions_from_path

class Path:
    def __init__(self, path, game_map, start, target):
        if path is None:
            self.path = true_random_nsteps(game_map, start, target)
        self.path = path
        self.game_map = game_map
        self.start = start
        self.target = target
        self.actions = actions_from_path(self.start, self.path)
    
    def __str__(self):
        return f'Path: {self.path}\nActions: {self.actions}'

class Individual:
    def __init__(self, path: Path, generation: int):
        self.path = path
        self.fitness = 666
        self.generation = generation

    def __str__(self):
        return f'{self.path}\nFitness: {self.fitness}\nGeneration: {self.generation}'

    def __repr__(self):
        return self.__str__()

