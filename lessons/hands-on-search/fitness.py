def count_dead_ends(path):
    dead_ends = 0
    for i in range(1, len(path) - 1):
        if is_dead_end(path, i):
            dead_ends += 1
    return dead_ends

def is_dead_end(path, index):
    current_location = path[index]
    neighbors = [path[index - 1], path[index + 1]]
    return neighbors.count(current_location) == 2


def count_loops(path):
    loops = 0
    for i in range(2, len(path) - 1):
        if is_loop(path, i):
            loops += 1
    return loops

def is_loop(path, index):
    current_location = path[index]
    previous_locations = path[:index - 1]
    return current_location in previous_locations


def calculate_checkpoint_bonus(path, checkpoints):
    # Calculate a bonus based on how many checkpoints are visited
    visited_checkpoints = set(path) & set(checkpoints)
    return len(visited_checkpoints) * 50  # Adjust the weight based on the importance of checkpoints

def calculate_diversity_penalty(path, previous_paths):
    # Penalize paths that are too similar to previous paths
    similarity_threshold = 0.8  # Adjust the threshold based on desired diversity
    similarity_scores = [calculate_similarity(path, prev_path) for prev_path in previous_paths]
    if any(score > similarity_threshold for score in similarity_scores):
        return -50  # Adjust the penalty based on the importance of diversity
    else:
        return 0

def calculate_similarity(path1, path2):
    # Calculate a similarity score between two paths
    common_locations = set(path1) & set(path2)
    similarity_score = len(common_locations) / max(len(path1), len(path2))
    return similarity_score

def path_reaches_goal(path):
    return path[-1] == target

def count_collisions(path):
    collisions = 0
    for x, y in path:
        if is_wall(game_map[x, y]):
            collisions += 1
    return collisions



# Try for different checkpoints: random, placed near the goal, placed near dead-ends, placed in complex areas
#TODO: salvare tutte le generazioni in una lista previous_paths

def fitness_function(path, checkpoints, generation):
    #length_penalty = -len(path)
    goal_bonus = 10 if path_reaches_goal(path) else 0
    collision_penalty = -count_collisions(path)
    revisit_penalty = -len(set(path)) # Penalize paths that visit the same location multiple times
    dead_end_penalty = -count_dead_ends(path)
    loop_penalty = -count_loops(path)

    # Checkpoint factor
    checkpoint_bonus = calculate_checkpoint_bonus(path, checkpoints)

    # Diversity factor applied only in the first few generations: this choice (combined with exponential decay
    # to gradually reduce the mutation rate as the algorithm progresses) encourages diversity in the first steps
    # of the optimization process
    diversity_penalty = 0
    if generation < 5:
        diversity_penalty = calculate_diversity_penalty(path, previous_paths[:generation])

    # Combine factors to compute overall fitness
    fitness = goal_bonus + collision_penalty + revisit_penalty + dead_end_penalty + loop_penalty + checkpoint_bonus + diversity_penalty

    return fitness