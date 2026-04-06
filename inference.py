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

    print("STATE:", state.user_state, state.notification_type)

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

    print("RAW RESPONSE:", response)

    content = response.choices[0].message.content

    if content is None:
        return "delay"

    action = content.strip().lower()

    if "show" in action:
        return "show_now"
    elif "mute" in action:
        return "mute"
    elif "delay" in action:
        return "delay"
    else:
        print("INVALID ACTION FROM MODEL:", action)
        return "delay"

def rule_based_action(state):
    recent = state.history if hasattr(state, "history") and state.history else []
    user = state.user_state
    notif = state.notification_type

    # Count recent action patterns
    recent_actions = [str(h).split(":")[-1] if ":" in str(h) else str(h) for h in recent]
    delay_streak = sum(1 for a in recent_actions[-3:] if a == "delay")
    show_streak = sum(1 for a in recent_actions[-3:] if a == "show_now")

    # ── PRIORITY 1: SLEEPING — always delay (protect sleep) ──
    if user == "sleeping":
        return "delay"

    # ── PRIORITY 2: STUDYING — protect focus ──
    if user == "studying":
        if notif == "social":
            return "mute"
        elif notif == "urgent":
            return "show_now"
        elif notif == "work":
            return "delay"
        return "delay"

    # ── PRIORITY 3: FREE TIME — show everything ──
    if user == "free_time":
        # Break show_now streaks to avoid spam penalty
        if show_streak >= 3:
            return "delay"
        return "show_now"

    # ── PRIORITY 4: URGENT fallback (unknown states) ──
    if notif == "urgent":
        return "show_now"

    # ── PRIORITY 5: HISTORY-BASED ADJUSTMENT ──
    # Break delay streaks to avoid stagnation penalty
    if delay_streak >= 3 and notif != "social":
        return "show_now"

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
            action = rule_based_action(state)

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