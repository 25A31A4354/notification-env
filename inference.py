import os
import requests
import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    try:
        url = os.getenv("API_BASE_URL") + "/chat/completions"

        headers = {
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.getenv("MODEL_NAME"),
            "messages": [
                {
                    "role": "user",
                    "content": f"You are an AI managing phone notifications.\n\nUser state: {state.user_state}\nNotification: {state.notification_type}\nHistory: {state.history}\n\nChoose ONLY ONE action:\nshow_now\ndelay\nmute\n\nRespond with only one word."
                }
            ],
            "temperature": 0
        }

        response = requests.post(url, headers=headers, json=payload)

        print("STATUS:", response.status_code)
        print("RAW RESPONSE:", response.text)

        raw = response.json()["choices"][0]["message"]["content"].lower()

        if "show" in raw:
            return "show_now"
        elif "mute" in raw:
            return "mute"
        elif "delay" in raw:
            return "delay"
        else:
            return "delay"

    except Exception as e:
        print("LLM ERROR:", e)
        return "delay"

def run_env():
    output = ""

    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        output += f"\n[START] Task: {task['name']}\n"

        for step in range(task["steps"]):
            action = simple_agent(state)

            output += f"[STEP] {step} | State: {state} | Action: {action}\n"

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        score = grade(total_reward, max_possible)
        output += f"[END] Score: {score}\n"

    return output

demo = gr.Interface(
    fn=run_env,
    inputs=[],
    outputs="text",
    title="Smart Notification Manager AI",
    description="Run AI-based notification decision system"
)

demo.launch(server_name="0.0.0.0", server_port=7860)