import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    user = state.user_state
    notif = state.notification_type
    history = state.history if hasattr(state, "history") and state.history else []

    # ── 1. CORE DECISION RULES (highest priority) ──
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
        if notif == "urgent":
            action = "delay"
        else:
            action = "delay"
    elif user == "free_time":
        action = "show_now"
    else:
        # fallback for any unknown user state
        if notif == "urgent":
            action = "show_now"
        else:
            action = "delay"

    # ── 2. HISTORY ADJUSTMENT (low priority) ──
    # Only apply if last 3 actions are identical.
    # NEVER override urgent decisions or studying rules.
    last_actions = history[-3:]
    if len(last_actions) == 3 and user != "studying" and notif != "urgent":
        if last_actions.count("delay") == 3:
            action = "show_now"
        elif last_actions.count("show_now") == 3:
            action = "delay"

    # ── 3. SAFETY RULES ──
    # Never mute urgent notifications
    if notif == "urgent" and action == "mute":
        action = "delay"
    # Never show social during studying
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