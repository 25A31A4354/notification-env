import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    user = state.user_state
    notif = state.notification_type
    history = state.history if hasattr(state, "history") and state.history else []

    if notif == "urgent":
        return "show_now"

    recent_same = len(history) >= 2 and history[-1] == history[-2]
    last_action = history[-1] if history else None
    delay_count = history.count("delay") if history else 0
    many_delays = delay_count >= 3

    if user == "studying":
        if notif == "social":
            if recent_same and last_action == "mute":
                return "delay"
            return "mute"
        if notif == "work":
            if recent_same and last_action == "delay":
                return "show_now"
            return "delay"
        return "delay"

    if user == "sleeping":
        if recent_same and last_action == "delay":
            return "show_now"
        return "delay"

    if user == "free_time":
        return "show_now"

    if many_delays:
        return "show_now"

    if recent_same and last_action == "delay":
        return "show_now"
    if recent_same and last_action == "mute":
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