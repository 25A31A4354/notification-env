import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS


def agent(state):
    """
    Optimal reward-maximizing agent based on env.py reward table:

    SLEEPING:  mute=+5, delay=0, show_now=-10
    FREE_TIME: show_now=+5, delay=+2, mute=0
    STUDYING:
      social:  mute=+10, delay=+2, show_now=-10
      work:    delay=+6, show_now=+3, mute=-5
      urgent:  show_now=+10, else=-8
    """
    user = state.user_state
    notif = state.notification_type

    if user == "sleeping":
        return "mute"           # +5 (best possible for sleeping)

    if user == "studying":
        if notif == "social":
            return "mute"       # +10
        elif notif == "work":
            return "delay"      # +6
        elif notif == "urgent":
            return "show_now"   # +10
        return "delay"

    if user == "free_time":
        return "show_now"       # +5 (best possible for free_time)

    return "delay"              # safe fallback


def run_env():
    output = ""

    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        output += f"\n[START] Task: {task['name']}\n"

        for step in range(task["steps"]):
            action = agent(state)

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