import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    user = state.user_state
    notif = state.notification_type

    # ── 1. USER STATE RULES ──
    if user == "studying":
        if notif == "social":
            action = "mute"
        elif notif == "work":
            action = "delay"
        elif notif == "urgent":
            action = "show_now"
        else:
            action = "delay"
    elif user == "sleeping":
        action = "delay"
    elif user == "free_time":
        action = "show_now"
    else:
        action = "delay"

    # ── 2. SAFETY RULES ──
    if notif == "urgent" and action == "mute":
        action = "delay"
    if user == "studying" and notif == "social" and action == "show_now":
        action = "mute"

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