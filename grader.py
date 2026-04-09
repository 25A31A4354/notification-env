def grade(total_reward, max_possible_reward):
    if max_possible_reward == 0:
        return 0.01

    score = total_reward / max_possible_reward

    return max(0.01, min(0.99, score))