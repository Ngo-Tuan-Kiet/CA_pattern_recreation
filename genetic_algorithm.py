# ca_ga_pattern_match.py
import numpy as np
import cellpylib as cpl
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Parameters
GRID_SIZE = (20, 20)
POP_SIZE = 40
NUM_GENERATIONS = 80
NUM_STEPS = 50
DEFAULT_MUTATION_RATE = 0.015
TOURNAMENT_SIZE = 4
ELITE_COUNT = 5
STAGNATION_LIMIT = 4
MUTATION_INCREASE = 0.005
MAX_MUTATION_RATE = 0.1

INITIAL_GRID = np.random.randint(0, 2, size=GRID_SIZE, dtype=int)

# Generate target pattern (user-defined or load from file)
target_pattern = np.zeros(GRID_SIZE, dtype=int)

# target_pattern[5:15, 5:15] = 1  # simple square for testing

# Smiley face pattern
# Eyes (thicker)
target_pattern[5:8, 6:9] = 1
target_pattern[5:8, 11:14] = 1
# Mouth corners (thicker)
target_pattern[12:14, 6:9] = 1
target_pattern[12:14, 11:14] = 1
# Mouth curve (thicker)
target_pattern[14:16, 8:12] = 1

def random_rule():
    return np.random.randint(0, 2, size=(2 ** 9,), dtype=int)

def apply_rule(grid, rule, n_last=10):
    def rule_fn(neighborhood, position=None, t=None):
        idx = int(''.join(map(str, neighborhood.flatten())), 2)
        return rule[idx]
    
    # Evolve using cellpylib
    grid_history = cpl.evolve2d(
        np.array([grid]),
        timesteps=NUM_STEPS,
        neighbourhood="Moore",
        r=1,
        apply_rule=rule_fn
    )
    # return n_last grids
    return grid_history [-n_last:]

def fitness(rule):
    final_grids = apply_rule(INITIAL_GRID, rule)
    # negative Hamming distances
    fitness_score = 0
    target_ones = np.sum(target_pattern == 1)
    target_zeros = np.sum(target_pattern == 0)

    # Avoid division by zero
    reward_1 = 4000 / target_ones if target_ones > 0 else 0
    reward_0 = 4000 / target_zeros if target_zeros > 0 else 0

    for grid in final_grids:
        matches_1 = (target_pattern == 1) & (grid == 1)
        matches_0 = (target_pattern == 0) & (grid == 0)
        false_negatives = (target_pattern == 1) & (grid == 0)
        false_positives = (target_pattern == 0) & (grid == 1)

        fitness_score += reward_1 * np.sum(matches_1)
        fitness_score += reward_0 * np.sum(matches_0)
        fitness_score -= reward_1 / 4 * np.sum(false_negatives)
        fitness_score -= reward_1 / 4 * np.sum(false_positives)


        # TODO: calculate Langtons Lambda Parameter
        # TODO: Reward Diversity
        # TODO: Reward equal number of black cells
    
    return fitness_score

def tournament_selection(pop, fitnesses):
    selected = random.sample(list(zip(pop, fitnesses)), TOURNAMENT_SIZE)
    return max(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child


# def mutate(rule, mutation_rate):
#     for i in range(len(rule)):
#         if random.random() < mutation_rate:
#             rule[i] = 1 - rule[i]
#     return rule

def mutate(rule, mutation_rate):
    mutation_mask = np.random.rand(len(rule)) < mutation_rate
    rule[mutation_mask] ^= 1
    return rule


def visualize_evolution(grid_history, seconds=2):
    fig, ax = plt.subplots()
    ax.set_title("Evolution of Cellular Automaton")
    ax.axis('off')

    def update(frame):
        ax.clear()
        ax.imshow(grid_history[frame], cmap='binary')
        ax.set_title(f"Generation {frame+1}")
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=2000/len(grid_history))
    
    plt.show(block=False)
    plt.pause(seconds)
    plt.close(fig)


def evolve():
    population = [random_rule() for _ in range(POP_SIZE)]
    fitness_log = []
    best_fitness = -np.inf
    stagnation_counter = 0
    mutation_rate = DEFAULT_MUTATION_RATE

    for gen in range(NUM_GENERATIONS):
        fitnesses = [fitness(rule) for rule in population]

        best_idx = np.argmax(fitnesses)
        current_best = max(fitnesses)

        fitness_log.append(fitnesses[best_idx])
        # print(f"Generation {gen}: Best fitness = {fitnesses[best_idx]}")

        if current_best <= best_fitness:
            stagnation_counter += 1
            if stagnation_counter >= STAGNATION_LIMIT:
                mutation_rate = min(mutation_rate + MUTATION_INCREASE, MAX_MUTATION_RATE)
                print(f"No improvement for {STAGNATION_LIMIT} gens. Increasing mutation rate to {mutation_rate:.4f}")
        else:
            best_fitness = current_best
            stagnation_counter = 0
            mutation_rate = DEFAULT_MUTATION_RATE

        elites = [population[i] for i in np.argsort(fitnesses)[-ELITE_COUNT:]]
        new_population = elites.copy()
        print(f"Generation {gen}: Elite fitnesses = {[fitness(rule) for rule in elites]}")
        visualize_evolution(apply_rule(INITIAL_GRID, elites[-1]))

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            # Ensure child is unique
            # while any(np.array_equal(child, existing) for existing in new_population):
            #     child = mutate(crossover(parent1, parent2), mutation_rate)

            new_population.append(child)

        
        population = new_population

    # Final best rule
    final_fitnesses = [fitness(rule) for rule in population]
    best_rule = population[np.argmax(final_fitnesses)]
    return best_rule, fitness_log

if __name__ == "__main__":
    print("Target:\n", target_pattern.astype(int))
    print("Initial grid:\n", INITIAL_GRID.astype(int))
    print("Evolving...")
    best_rule, fitness_log = evolve()
    result = apply_rule(INITIAL_GRID, best_rule)
    visualize_evolution(result)

    print("Target pattern vs. Evolved result:")
    print("Target:\n", target_pattern.astype(int))
    print("Result:\n", result[-1].astype(int))

    # Show animation   of best rule

