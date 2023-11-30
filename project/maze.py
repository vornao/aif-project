import random

# create labyrinth with matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Maze:
    def __init__(self, size):
        self.size = size
        self.maze = np.zeros((size, size), dtype=int)
        self.maze[0, 0] = 1
        self.maze[size - 1, size - 1] = 2

    def set_cell(self, x, y, value):
        self.maze[x, y] = value

    def is_valid_location(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.maze[x, y] != 0

    def distance_to_exit(self, x, y):
        exit_x, exit_y = np.where(self.maze == 2)
        return abs(exit_x - x) + abs(exit_y - y)

    def __str__(self):
        return str(self.maze)

def print_maze(maze):
    plt.imshow(maze.maze, cmap='binary')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Define constants
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100

# Define possible actions (genes)
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def fitness(individual):
    x, y = 0, 0  # Initial position
    for action in individual:
        if action == 'UP' and x > 0 and maze[x - 1][y] == 0:
            x -= 1
        elif action == 'DOWN' and x < len(maze) - 1 and maze[x + 1][y] == 0:
            x += 1
        elif action == 'LEFT' and y > 0 and maze[x][y - 1] == 0:
            y -= 1
        elif action == 'RIGHT' and y < len(maze[0]) - 1 and maze[x][y + 1] == 0:
            y += 1
            
    distance_to_exit = abs(x - len(maze) + 1) + abs(y - len(maze[0]) + 1)
    return 1 / (distance_to_exit + 1)

def create_individual():
    return [random.choice(ACTIONS) for _ in range(5 * 5)]  # Size of the maze

def crossover(parent1, parent2):
    """Perform crossover to create a new individual."""
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual):
    """Perform mutation on an individual."""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.choice(ACTIONS)
    return individual

# Print the initial maze
print("Initial Maze:")
print_maze(maze)

population = [create_individual() for _ in range(POPULATION_SIZE)]

for generation in range(MAX_GENERATIONS):
    fitness_scores = [fitness(individual) for individual in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    parents = random.choices(population, weights=probabilities, k=2)
    offspring = [crossover(parents[0], parents[1]) for _ in range(POPULATION_SIZE - 2)]
    offspring = [mutate(individual) for individual in offspring]
    population = parents + offspring

    best_individual = population[fitness_scores.index(max(fitness_scores))]
    print(f"Generation {generation + 1}, Best Fitness: {fitness(best_individual)}")

# Print the best individual in the final generation
best_individual = population[fitness_scores.index(max(fitness_scores))]
print("Best Individual:", best_individual)

# Print the final maze using the best individual's path
print("\nFinal Maze:")
x, y = 0, 0
for action in best_individual:
    if action == 'UP':
        x -= 1
    elif action == 'DOWN':
        x += 1
    elif action == 'LEFT':
        y -= 1
    elif action == 'RIGHT':
        y += 1
    maze.set_cell(x, y, 'X')  # Mark the path with 'X'
print_maze(maze)
