# ca_ga_pattern_match.py
import numpy as np
import cellpylib as cpl
import random
from PIL import Image
import pygame
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Parameters
GRID_SIZE = (100, 100)
POP_SIZE = 200
NUM_GENERATIONS = 3000
NUM_STEPS = 200
DEFAULT_MUTATION_RATE = 0.005
TOURNAMENT_SIZE = 10
ELITE_COUNT = 5
STAGNATION_LIMIT = 5
NUM_GENERATIONS_BREAKING = 200
MUTATION_INCREASE = 0.0025
MAX_MUTATION_RATE = 0.075

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
    return grid_history

def apply_rule_async(grid, rule, n_last=10):
    height, width = grid.shape
    grid_history = np.zeros((NUM_STEPS + 1, height, width), dtype=np.int8)
    grid_history[0] = grid
    
    # Cache for neighborhood rule lookups
    neighborhood_to_result = {}
    
    for t in range(NUM_STEPS):
        current = grid_history[t]
        next_grid = np.zeros_like(current)
        
        for i in range(height):
            for j in range(width):
                # Construct the 3x3 neighborhood with periodic boundary conditions
                neighborhood = np.zeros((3, 3), dtype=np.int8)
                
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = (i + di) % height  # Wrap around for boundary conditions
                        nj = (j + dj) % width
                        neighborhood[di+1, dj+1] = current[ni, nj]
                
                # Convert neighborhood to a tuple for dictionary lookup
                neighborhood_tuple = tuple(map(tuple, neighborhood))
                
                # Look up or calculate the result
                if neighborhood_tuple not in neighborhood_to_result:
                    # Convert flattened neighborhood to binary string, then to integer
                    idx = int(''.join(map(str, neighborhood.flatten())), 2)
                    neighborhood_to_result[neighborhood_tuple] = rule[idx]
                
                next_grid[i, j] = neighborhood_to_result[neighborhood_tuple]
        
        grid_history[t+1] = next_grid
    
    # Return the last n_last grids
    return grid_history


def apply_rule_sync(grid, rule, num_steps=20, n_last=10):
    # Convert rule into NumPy array if it's not already
    rule = np.asarray(rule, dtype=np.uint8)

    # Define 3x3 neighborhood weights for indexing
    powers_of_two = np.array([[256, 128,  64],
                              [32,   16,   8],
                              [4,     2,   1]], dtype=np.uint16)

    history = [grid.copy()]

    for _ in range(num_steps - 1):
        # Compute 3x3 neighborhood code for each cell
        neighborhood = convolve2d(history[-1], np.ones((3, 3), dtype=int), mode='same', boundary='wrap')
        
        # Use bit shifting to create index
        idx_grid = convolve2d(history[-1], powers_of_two, mode='same', boundary='wrap')

        # Apply rule: each cell becomes rule[idx]
        new_grid = rule[idx_grid]
        history.append(new_grid)

    # Return only the last n grids
    return history


def fitness(rule, n_last=10):
    # print (f"Evaluating rule: {rule}")
    final_grids = apply_rule_sync(INITIAL_GRID, rule)
    # negative Hamming distances
    fitness_score = 0
    target_ones = np.sum(target_pattern == 1)
    target_zeros = np.sum(target_pattern == 0)

    # Avoid division by zero
    reward_1 = 400 / target_ones if target_ones > 0 else 0
    reward_0 = 400 / target_zeros if target_zeros > 0 else 0

    for grid in final_grids[-n_last:]:
        matches_1 = (target_pattern == 1) & (grid == 1)
        matches_0 = (target_pattern == 0) & (grid == 0)
        false_negatives = (target_pattern == 1) & (grid == 0)
        false_positives = (target_pattern == 0) & (grid == 1)


        fitness_score += reward_1 * np.sum(matches_1)
        fitness_score += reward_0 * np.sum(matches_0)
        fitness_score -= reward_1 * np.sum(false_negatives)
        fitness_score -= reward_0 * np.sum(false_positives)


        # TODO: calculate Langtons Lambda Parameter
        # TODO: Reward Diversity
        # TODO: Reward equal number of black cells

    
    # calculate Langtons lambda parameter fot hte rule


    
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
    mutated_rule = rule.copy()
    mutated_rule[mutation_mask] ^= 1
    return mutated_rule


def visualize_evolution(grid_history, seconds=2, save_path=None):
    fig, ax = plt.subplots()
    ax.set_title("Evolution of Cellular Automaton")
    ax.axis('off')

    def update(frame):
        ax.clear()
        ax.imshow(grid_history[frame], cmap='binary')
        ax.set_title(f"CA step: {frame+1}")
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=2000/len(grid_history))

    if save_path:
        ani.save(save_path, writer='imagemagick', fps=5)
    else:
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
        fitnesses = [fitness(rule) for rule in tqdm(population, desc="Evaluating")]

        best_idx = np.argmax(fitnesses)
        current_best = max(fitnesses)

        fitness_log.append(fitnesses[best_idx])
        # print(f"Generation {gen}: Best fitness = {fitnesses[best_idx]}")

        if current_best <= best_fitness:
            stagnation_counter += 1
            if stagnation_counter >= STAGNATION_LIMIT:
                mutation_rate = min(mutation_rate + MUTATION_INCREASE, MAX_MUTATION_RATE)
                print(f"No improvement for {stagnation_counter} gens. Increasing mutation rate to {mutation_rate:.4f}")
            if stagnation_counter >= NUM_GENERATIONS_BREAKING:
                print(f"Stagnation for {stagnation_counter} generations. Breaking...")
                break
        else:
            best_fitness = current_best
            stagnation_counter = 0
            mutation_rate = DEFAULT_MUTATION_RATE

        elites = [population[i] for i in np.argsort(fitnesses)[-ELITE_COUNT:]]
        # copy best 2 to new population
        new_population = elites[-2:].copy()
        print(f"Generation {gen}: Elite fitnesses = {[fitness(rule) for rule in elites]}")
        
        # Mutate elites
        for i in range(ELITE_COUNT):
            mutated_elite = mutate(new_population[i], mutation_rate)
            new_population.append(mutated_elite)

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


def png_to_grid(path, threshold=128):
    image = Image.open(path).convert('L')  # Convert to grayscale
    image = image.point(lambda p: p < threshold and 1)  # Binarize
    grid = np.array(image, dtype=np.uint8)
    return grid


def draw_and_save_grid(grid_size=200, cell_size=7):
    pygame.init()

    window_size = grid_size * cell_size
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Draw on Grid (ESC or close window to finish)")

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    clock = pygame.time.Clock()
    drawing = False

    running = True
    while running:
        clock.tick(120)  # Limit to 120 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

        if drawing:
            x, y = pygame.mouse.get_pos()
            col = x // cell_size
            row = y // cell_size
            if 0 <= col < grid_size and 0 <= row < grid_size:
                grid[row, col] = 1

        # Draw everything
        screen.fill((255, 255, 255))  # White background
        for y in range(grid_size):
            for x in range(grid_size):
                if grid[y, x] == 1:
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, (0, 0, 0), rect)

        pygame.display.flip()

    pygame.quit()
    return grid


if __name__ == "__main__":

    # Set random seed for reproducibility
    # random.seed(42)d

    base_save_path = "evol6"
    target_pattern = png_to_grid("stay4_initial.png")
    
    # INITIAL_GRID = target_pattern.copy()

    initial_grid_save_path = f"{base_save_path}_initial.png"
    plt.imsave(initial_grid_save_path, INITIAL_GRID, cmap='binary')

    print("Target:\n", target_pattern.astype(int))
    print("Initial grid:\n", INITIAL_GRID.astype(int))
    print("Evolving...")
    best_rule, fitness_log = evolve()
    result = apply_rule(INITIAL_GRID, best_rule)

    visualize_evolution(result, seconds=2, save_path=f"{base_save_path}.gif")

    # Save the best rule
    np.save(f"{base_save_path}_best_rule.npy", best_rule)
    # Save the fitness log
    np.save(f"{base_save_path}_fitness_log.npy", fitness_log)
    # Save the final grid
    plt.imsave(f"{base_save_path}_final.png", result[-1], cmap='binary')

    print("Target pattern vs. Evolved result:")
    print("Target:\n", target_pattern.astype(int))
    print("Result:\n", result[-1].astype(int))

    # Show animation   of best rule

    exit()

    grid_of_1s = np.ones(GRID_SIZE, dtype=int)
    grid_of_0s = np.zeros(GRID_SIZE, dtype=int)

    test_grids = [target_pattern, grid_of_1s, grid_of_0s]
    target_ones = np.sum(target_pattern == 1)
    target_zeros = np.sum(target_pattern == 0)

    # Avoid division by zero
    reward_1 = 4000 / target_ones if target_ones > 0 else 0
    reward_0 = 4000 / target_zeros if target_zeros > 0 else 0

    for grid in test_grids:
        fitness_score = 0
        matches_1 = (target_pattern == 1) & (grid == 1)
        matches_0 = (target_pattern == 0) & (grid == 0)
        false_negatives = (target_pattern == 1) & (grid == 0)
        false_positives = (target_pattern == 0) & (grid == 1)


        fitness_score += reward_1 * np.sum(matches_1)
        fitness_score += reward_0 * np.sum(matches_0)
        fitness_score -= (reward_1 / 1) * np.sum(false_negatives)
        fitness_score -= (reward_1 / 1) * np.sum(false_positives)

        print(f"Fitness score for grid:\n{grid.astype(int)}\nScore: {fitness_score}")

    exit()

    grid_of_1s = np.ones(GRID_SIZE, dtype=int)
    grid_of_0s = np.zeros(GRID_SIZE, dtype=int)
    grid_with_1_1s = np.zeros(GRID_SIZE, dtype=int)
    grid_with_1_1s[6, 6] = 1  # simple square for testing
    grid_with_1_0s = np.ones(GRID_SIZE, dtype=int)
    grid_with_1_0s[5, 5] = 0  # simple square for testing

    target_pattern = grid_with_1_1s

    test_grids = [target_pattern, grid_of_1s, grid_of_0s]
    target_ones = np.sum(target_pattern == 1)
    target_zeros = np.sum(target_pattern == 0)

    # Avoid division by zero
    reward_1 = 4000 / target_ones if target_ones > 0 else 0
    reward_0 = 4000 / target_zeros if target_zeros > 0 else 0

    for grid in test_grids:
        fitness_score = 0
        matches_1 = (target_pattern == 1) & (grid == 1)
        matches_0 = (target_pattern == 0) & (grid == 0)
        false_negatives = (target_pattern == 1) & (grid == 0)
        false_positives = (target_pattern == 0) & (grid == 1)


        fitness_score += reward_1 * np.sum(matches_1)
        fitness_score += reward_0 * np.sum(matches_0)
        fitness_score -= (reward_1 / 1) * np.sum(false_negatives)
        fitness_score -= (reward_0 / 1) * np.sum(false_positives)

        print(f"Fitness score for grid:\n{grid.astype(int)}\nScore: {fitness_score}")

    exit()

