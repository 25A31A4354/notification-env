from env import NotificationEnv
from grader import grade
from tasks import TASKS
import random

ACTIONS = ["show_now", "delay", "mute"]

def run_task(task):
    env = NotificationEnv()
    state = env.reset()

    total_reward = 0
    max_possible = task["steps"] * 10  # max reward per step

    for _ in range(task["steps"]):
        action = random.choice(ACTIONS)  # random agent

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    score = grade(total_reward, max_possible)

    return total_reward, score

def main():
    print("Running Inference...\n")

    for task in TASKS:
        total_reward, score = run_task(task)

        print(f"Task: {task['name']}")
        print(f"Total Reward: {total_reward}")
        print(f"Score: {round(score, 2)}")
        print("-" * 30)

if __name__ == "__main__":
    main()