import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 20              # Grid size
T = 10              # Max phase (mod T)
steps = 100         # Number of simulation steps
neighborhood_size = 1  # Moore neighborhood

# Initialize grid with random phases
grid = np.random.randint(0, T, size=(N, N))

def get_neighbors(grid, x, y):
    """Returns the phase values of Moore neighbors around (x, y)."""
    neighbors = []
    for dx in range(-neighborhood_size, neighborhood_size + 1):
        for dy in range(-neighborhood_size, neighborhood_size + 1):
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % N
            ny = (y + dy) % N
            neighbors.append(grid[nx, ny])
    return neighbors

def ca_rule(current, neighbors, genome):
    """Update rule defined by an evolved genome (lookup table)."""
    # Use average neighbor phase (mod T)
    avg = sum(neighbors) / len(neighbors)
    diff = (avg - current) % T
    
    # Genome is a vector of T values mapping diff to increment
    index = int(diff) % T
    return (current + genome[index]) % T

# Example genome: fixed increments to shift toward neighbor average
genome = np.random.randint(0, 3, size=T)  # Replace with GA-evolved genome

def update_grid(grid, genome):
    new_grid = np.zeros_like(grid)
    for i in range(N):
        for j in range(N):
            neighbors = get_neighbors(grid, i, j)
            new_grid[i, j] = ca_rule(grid[i, j], neighbors, genome)
    return new_grid

# Run simulation
history = []
for t in range(steps):
    history.append(grid.copy())
    grid = update_grid(grid, genome)

# Visualization
def plot_phases(history):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        idx = int(i * steps / 5)
        ax.imshow(history[idx], cmap="hsv", vmin=0, vmax=T)
        ax.set_title(f"Step {idx}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_phases(history)
