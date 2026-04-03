import os
from env import NotificationEnv
from grader import grade
from tasks import TASKS
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

VALID_ACTIONS = ["show_now", "delay", "mute"]

def get_action_from_llm(state):
    prompt = f"""
User is {state.user_state}.
Notification is {state.notification_type}.

Choose ONE action:
show_now, delay, mute

Only return one word.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in VALID_ACTIONS:
            return "delay"

        return action

    except:
        return "delay"


def run_task(task):
    env = NotificationEnv()
    state = env.reset()

    total_reward = 0
    max_possible = task["steps"] * 10

    print(f"[START] Task: {task['name']}")

    for step in range(task["steps"]):
        action = get_action_from_llm(state)

        print(f"[STEP] {step} | State: {state} | Action: {action}")

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    score = grade(total_reward, max_possible)

    print(f"[END] Score: {score}\n")

    return total_reward, score


def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()