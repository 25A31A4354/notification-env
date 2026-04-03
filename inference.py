import time
from env import NotificationEnv
from grader import grade
from tasks import TASKS

ACTIONS = ["show_now", "delay", "mute"]

def simple_agent(state):
    if state.notification_type == "urgent":
        return "show_now"
    if state.user_state == "studying":
        return "mute"
    return "delay"

def run_all():
    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        print(f"[START] Task: {task['name']}")

        for step in range(task["steps"]):
            action = simple_agent(state)

            print(f"[STEP] {step} | State: {state} | Action: {action}")

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        score = grade(total_reward, max_possible)

        print(f"[END] Score: {score}\n")

def main():
    while True:
        run_all()
        time.sleep(10)  # keeps app alive

if __name__ == "__main__":
    main()