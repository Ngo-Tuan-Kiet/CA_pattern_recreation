import numpy as np
import matplotlib.pyplot as plt
import os



def plot_fitness_convergence(fitness_log, noise_level, save_path=None):
    generations = [log['generation'] for log in fitness_log]
    best_fitnesses = [log['best_fitness'] for log in fitness_log]
    avg_fitnesses = [log['average_fitness'] for log in fitness_log]

    plt.figure()
    plt.plot(generations, best_fitnesses, label='Best Fitness')
    plt.plot(generations, avg_fitnesses, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.title(f'Fitness Convergence (Noise: {noise_level*100:.0f}%)')
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

results_folders = ["denoising_results", "denoising_results_norm"]
noise_levels = [0.05, 0.10, 0.20, 0.30]
baseline = 0.8

# plot_average_final_best_fitness(results_folders, noise_levels, baseline=baseline, save_plot=True)

fitness_log = np.load("./smiley_results_11/smiley_noise_05/fitness.npy", allow_pickle=True)
print(fitness_log)
plot_fitness_convergence(fitness_log, noise_level=0.2)