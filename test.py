# Self-Organizing Urban Growth with Cellular Automata and Genetic Algorithm

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import scipy.ndimage as ndimage


# Define cell types
CELL_TYPES = ['E', 'R', 'I', 'C', 'G', 'T']  # Empty, Residential, Industrial, Commercial, Green, Transport
# COLOR_MAP = {'E': 'white', 'R': 'blue', 'I': 'gray', 'C': 'orange', 'G': 'green', 'T': 'black'}
COLOR_MAP = {'E': (1.0, 1.0, 1.0), 'R': (0.0, 0.0, 1.0), 'I': (0.5, 0.5, 0.5), 'C': (1.0, 0.65, 0.0), 'G': (0.0, 1.0, 0.0), 'T': (0.0, 0.0, 0.0)}

# Grid size
GRID_SIZE = 30

def init_grid():
    return np.full((GRID_SIZE, GRID_SIZE), 'E')

def plot_grid(grid):
    fig, ax = plt.subplots()
    color_array = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            color_array[x, y] = COLOR_MAP.get(grid[x, y], (1.0, 1.0, 1.0))
    ax.imshow(color_array, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Get neighborhood cell counts
def get_neighborhood_counts(grid, x, y):
    counts = {k: 0 for k in CELL_TYPES}
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                counts[grid[nx, ny]] += 1
    return counts

# Apply rules to generate next grid state
def apply_rules(grid, rules):
    new_grid = grid.copy()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for rule in rules:
                if grid[x, y] == rule['from']:
                    counts = get_neighborhood_counts(grid, x, y)
                    match = all(counts.get(k, 0) >= v for k, v in rule['condition'].items())
                    if match and random.random() < rule['prob']:
                        new_grid[x, y] = rule['to']
                        break
    return new_grid

# Evaluate fitness of a city grid
def evaluate_fitness(grid):
    total = GRID_SIZE * GRID_SIZE
    score = 0

    # Target city proportions (in % of total grid cells)
    target_ratios = {
        'R': 0.30,
        'I': 0.10,
        'C': 0.15,
        'G': 0.20,
        'T': 0.15,
    }

    cell_counts = {k: np.sum(grid == k) for k in CELL_TYPES}

    # Penalize large deviation from target ratios
    proportion_penalty = 0
    for k, target in target_ratios.items():
        actual = cell_counts.get(k, 0) / total
        proportion_penalty += abs(actual - target)

    # Score based on proximity benefits
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell = grid[x, y]
            counts = get_neighborhood_counts(grid, x, y)
            if cell == 'R':
                score += 2 * counts['G']
                score -= counts['I']
            elif cell == 'I':
                score += counts['T']
            elif cell == 'C':
                score += counts['R']

    # Connectivity bonus for transport
    transport_grid = (grid == 'T').astype(int)
    labeled_array, num_features = ndimage.label(transport_grid)
    largest_component = 0
    if num_features > 0:
        component_sizes = ndimage.sum(transport_grid, labeled_array, range(1, num_features + 1))
        largest_component = max(component_sizes)
    connectivity_bonus = (largest_component / (total * 0.1))

    fitness = (score / total) + connectivity_bonus
    fitness -= 5 * proportion_penalty  # Apply strong penalty to skewed proportions
    return fitness


# Generate random ruleset
def random_rule():
    return {
        'from': 'E',
        'to': random.choice(['R', 'I', 'C', 'G', 'T']),
        'condition': {random.choice(CELL_TYPES): random.randint(2, 4)},
        'prob': round(random.uniform(0.3, 1.0), 2)
    }

def generate_random_ruleset(n=10):
    return [random_rule() for _ in range(n)]

# Genetic Algorithm
def evolve_population(population, generations=20):
    for gen in range(generations):
        scored = []
        for rules in population:
            grid = init_grid()
            for _ in range(10):
                grid = apply_rules(grid, rules)
            fitness = evaluate_fitness(grid)
            scored.append((fitness, rules))

        scored.sort(reverse=True, key=lambda x: x[0])
        best = scored[:10]
        print(f"Generation {gen}, Best Fitness: {best[0][0]:.2f}")

        # Elitism + crossover + mutation
        new_population = [copy.deepcopy(rules) for _, rules in best]
        while len(new_population) < len(population):
            parent1 = random.choice(best)[1]
            parent2 = random.choice(best)[1]
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        population = new_population

    return population[0]  # Return best ruleset

def crossover(rules1, rules2):
    idx = random.randint(1, len(rules1)-1)
    return rules1[:idx] + rules2[idx:]

def mutate(rules):
    for rule in rules:
        if random.random() < 0.1:
            rule['prob'] = round(min(1.0, max(0.1, rule['prob'] + random.uniform(-0.1, 0.1))), 2)

# Main simulation
if __name__ == '__main__':
    pop_size = 30
    population = [generate_random_ruleset() for _ in range(pop_size)]
    best_rules = evolve_population(population)

    # Show result
    final_grid = init_grid()
    for _ in range(20):
        final_grid = apply_rules(final_grid, best_rules)
    print(final_grid)
    plot_grid(final_grid)
    print("Best Ruleset:", best_rules)
