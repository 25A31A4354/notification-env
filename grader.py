def grade(total_reward, max_possible_reward):
    if max_possible_reward == 0:
        return 0.0

    score = total_reward / max_possible_reward

    return max(0.0, min(1.0, score))