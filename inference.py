import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    user = state.user_state
    notif = state.notification_type
    history = state.history if hasattr(state, "history") and state.history else []

    # ── 1. SAFETY RULES (highest priority — checked first AND last) ──
    # These constraints are absolute and cannot be violated.

    # ── 2. USER STATE LOGIC ──
    if user == "free_time":
        action = "show_now"
    elif user == "studying":
        if notif == "social":
            action = "mute"
        elif notif == "work":
            action = "delay"
        elif notif == "urgent":
            action = "show_now"
        else:
            action = "delay"
    elif user == "sleeping":
        action = "delay"  # ALL notifications delayed while sleeping
    else:
        action = "delay"  # safe fallback

    # ── 3. URGENT OVERRIDE (smart) ──
    if notif == "urgent":
        if user == "sleeping":
            action = "delay"
        else:
            action = "show_now"

    # ── 4. HISTORY ADJUSTMENT (lowest priority) ──
    # Only for non-critical situations to avoid spam/stagnation penalties.
    last_actions = history[-2:]
    if len(last_actions) == 2 and user != "studying" and notif != "urgent":
        if last_actions.count("show_now") == 2:
            action = "delay"    # avoid spam penalty
        elif last_actions.count("delay") == 2:
            action = "show_now"  # avoid stagnation

    # ── SAFETY ENFORCEMENT (final guardrails) ──
    # Never mute urgent
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