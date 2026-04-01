def grade(total_reward, max_possible_reward):
    score = total_reward / max_possible_reward

    # keep score between 0.0 and 1.0
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0

    return score