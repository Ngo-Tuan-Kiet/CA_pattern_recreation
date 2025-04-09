import numpy as np
import genetic_algorithm as ga

rule = np.load('./data/stay3_best_rule.npy')
grids = ga.apply_rule_async(ga.target_pattern, rule)

ga.visualize_evolution(grids)