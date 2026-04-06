import os
import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def decide_action(state):
    history = state.history if hasattr(state, "history") and state.history else []
    
    prompt = f"""
You are an intelligent notification manager.

User State: {state.user_state}
Notification Type: {state.notification_type}
Recent History: {history}

Rules:

* If user is studying or sleeping:

  * social → mute
  * work → delay
  * urgent → show_now

* If user is free_time:

  * social → show_now
  * urgent → show_now
  * work → show_now

Return ONLY one word:
show_now OR delay OR mute
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a strict decision-making AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    action = response.choices[0].message.content.strip().lower()

    if action not in ["show_now", "delay", "mute"]:
        return "delay"

    return action

def run_env():
    output = ""

    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        output += f"\n[START] Task: {task['name']}\n"

        for step in range(task["steps"]):
            try:
                action = decide_action(state)
            except Exception:
                action = "delay"

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