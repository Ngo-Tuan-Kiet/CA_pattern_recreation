import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import seaborn as sns

plt.rcParams.update({'font.size': 14}) 

def plot_fitness_convergence(fitness_log, noise_level, save_path=None):
    generations = [log['generation'] for log in fitness_log]
    best_fitnesses = [log['best_fitness'] for log in fitness_log]
    avg_fitnesses = [log['average_fitness'] for log in fitness_log]

    plt.figure()
    # color palette
    sns.set_palette("CMRmap", 3)

    plt.plot(generations, best_fitnesses, label='Best Fitness')
    plt.plot(generations, avg_fitnesses, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.title(f'Fitness Convergence using AAM on smiley pattern (Noise: {noise_level*100:.0f}%)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    


def plot_average_final_best_fitness(results_folders, noise_levels, baseline=None, save_plot=False):
    noise_to_best_fitnesses = {nl: [] for nl in noise_levels}

    for result_folder in results_folders:
        for noise_level in noise_levels:
            level_str = f"smiley_noise_{int(noise_level * 100):02d}"
            fitness_log_path = os.path.join(result_folder, level_str, "fitness.npy")

            if os.path.exists(fitness_log_path):
                fitness_log = np.load(fitness_log_path, allow_pickle=True)
                print(fitness_log)

                if isinstance(fitness_log, np.ndarray) and len(fitness_log) > 0:
                    last_entry = fitness_log[-1]
                    if isinstance(last_entry, dict) and 'best_fitness' in last_entry:
                        noise_to_best_fitnesses[noise_level].append(last_entry['best_fitness'])
                    else:
                        print(f"Unexpected format in: {fitness_log_path}")
                else:
                    print(f"Empty or invalid log in: {fitness_log_path}")
            else:
                print(f"Missing fitness log: {fitness_log_path}")

    # Compute average best fitness for each noise level
    avg_best_fitnesses = []
    for nl in noise_levels:
        values = noise_to_best_fitnesses[nl]
        if values:
            avg = np.mean(values)
        else:
            avg = np.nan
        avg_best_fitnesses.append(avg)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot([int(n * 100) for n in noise_levels], avg_best_fitnesses, marker='o', label='Avg Final Best Fitness')

    if baseline is not None:
        plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline')

    plt.title("Average Final Best Fitness Across Runs")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Avg Final Best Fitness")
    plt.grid(True)
    plt.legend()

    if save_plot:
        plt.savefig("avg_final_best_fitness_plot.png")

    plt.show()

def plot_avg_best_fitness_by_noise(root_dir, baseline=0.0, noise_level_labels=None):
    """
    Traverses the given directory recursively, extracts the last entry's best_fitness
    from each fitness.npy file, groups them by noise level (assumed to be the
    immediate parent directory name), averages them, and plots the result with percent labels.

    Parameters:
        root_dir (str): Path to the root directory containing experiment runs.
        baseline (float): Minimum y-axis value for the plot.
        noise_level_labels (dict): Optional mapping from noise folder names to x-axis labels (e.g., "noise_10" -> "10%").
    """
    noise_fitness = defaultdict(list)

    for root, dirs, files in os.walk(root_dir):
        if 'fitness.npy' in files:
            path_parts = os.path.normpath(root).split(os.sep)
            noise_level = path_parts[-1]  # Adjust if needed

            try:
                fitness_data = np.load(os.path.join(root, 'fitness.npy'), allow_pickle=True)
                if isinstance(fitness_data, np.ndarray) and fitness_data.size > 0:
                    last_entry = fitness_data[-1]
                    best_fitness = float(last_entry['best_fitness'])
                    noise_fitness[noise_level].append(best_fitness)
            except Exception as e:
                print(f"Skipping {root} due to error: {e}")

    if not noise_fitness:
        print("No valid fitness.npy files found.")
        return

    # Use provided labels or default to folder names
    label_map = noise_level_labels if noise_level_labels else {k: k for k in noise_fitness}

    # Sort noise levels by label (assuming keys like 'noise_10', 'noise_20'...)
    sorted_keys = sorted(noise_fitness.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))

    avg_values = [np.mean(noise_fitness[k]) for k in sorted_keys]
    x_labels = [label_map.get(k, k) for k in sorted_keys]

    plt.figure(figsize=(10, 5))
    sns.set_palette("husl", len(avg_values))
    plt.bar(x_labels, avg_values, color=sns.color_palette("blend:#7AB,#EDA", 7))  
    # show values on top of bars
    for i, v in enumerate(avg_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline})')
    plt.ylim(bottom=0.60)
    plt.ylabel('Average Best Fitness (Last Generation)')
    plt.xlabel('Noise Level')
    plt.title('Average Final Best Fitness per Noise Level')
    plt.legend()
    plt.tight_layout()
    plt.show()

results_folders = ["smiley_results_accelerated/smiley_results_2"]
noise_levels = [0.025, 0.05, 0.10, 0.20]

# smiley: 0.798117244330338
# circle_full: 0.6222222222222222
# ./circle_hollow_results: 0.7484823625922887
baseline = 0.6222


fitness_log = np.load("./smiley_results_accelerated/smiley_results_3/smiley_noise_20/fitness.npy", allow_pickle=True)
print(fitness_log)
plot_fitness_convergence(fitness_log, noise_level=0.2)

exit()

plot_avg_best_fitness_by_noise("./circle_full_results", baseline=baseline, noise_level_labels={
    "smiley_noise_02": "2.5%",
    "smiley_noise_05": "5%",
    "smiley_noise_10": "10%",
    "smiley_noise_20": "20%",
    "smiley_noise_30": "30%",
    "smiley_noise_40": "40%",
    "smiley_noise_50": "50%",
})

