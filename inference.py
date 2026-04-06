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

    # Count patterns
    urgent_count = sum(1 for h in recent if "urgent:show_now" in str(h))
    social_count = sum(1 for h in recent if "social:show_now" in str(h))

    # PRIORITY 1: URGENT ALWAYS IMPORTANT
    if state.notification_type == "urgent":
        return "show_now"

    # PRIORITY 2: DURING STUDY/SLEEP → PROTECT FOCUS
    if state.user_state in ["studying", "sleeping"]:
        if state.notification_type == "social":
            return "mute"
        if state.notification_type == "work":
            return "delay"

    # PRIORITY 3: FREE TIME BEHAVIOR
    if state.user_state == "free_time":
        if state.notification_type == "social":
            # If too many socials already shown → avoid spam
            if social_count >= 2:
                return "delay"
            return "show_now"

        if state.notification_type == "work":
            return "show_now"

    # PRIORITY 4: SMART ADJUSTMENT USING HISTORY
    # If too many urgent already handled → allow delay sometimes
    if urgent_count >= 2 and state.notification_type == "work":
        return "delay"

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