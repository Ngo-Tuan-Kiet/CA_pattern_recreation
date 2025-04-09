- had to weigh score more if black cells match then wgite cells because, often times the background colour has more cells, but are more irrelevant (pattern is small -> converge to blank grid and pattern big -> converge to full grid)
    target_ones = np.sum(target_pattern == 1)
    target_zeros = np.sum(target_pattern == 0)

    reward_1 = 4000 / target_ones if target_ones > 0 else 0
    reward_0 = 4000 / target_zeros if target_zeros > 0 else 0

    for grid in final_grids[-n_last:]:
        matches_1 = (target_pattern == 1) & (grid == 1)
        matches_0 = (target_pattern == 0) & (grid == 0)
        false_negatives = (target_pattern == 1) & (grid == 0)
        false_positives = (target_pattern == 0) & (grid == 1)

        fitness_score += reward_1 * np.sum(matches_1)
        fitness_score += reward_0 * np.sum(matches_0)

- still chaotic -> had to punish mismatches 
        fitness_score -= reward_1 / 4 * np.sum(false_negatives)
        fitness_score -= reward_0 / 4 * np.sum(false_positives)


- converges now to boring grids where only some stable 2-4 blocks survive or chaotic 

- often times no improvement for generations -> crossover often times too unstable for improvement -> just mutate elites first 
