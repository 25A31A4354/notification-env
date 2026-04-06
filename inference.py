"""
Smart Notification Manager AI
OpenEnv Hackathon — Production Agent

Reward-maximizing deterministic agent based on env.py reward table:

  SLEEPING:  mute=+5  | delay=0   | show_now=-10
  FREE_TIME: show_now=+5 | delay=+2 | mute=0
  STUDYING:
    social:  mute=+10  | delay=+2  | show_now=-10
    work:    delay=+6  | show_now=+3 | mute=-5
    urgent:  show_now=+10 | else=-8
"""

import logging
import gradio as gr
from env import NotificationEnv
from grader import grade
from tasks import TASKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_ACTIONS = {"show_now", "delay", "mute"}


def agent(state) -> str:
    """
    Deterministic optimal agent.

    Selects the action with the highest known reward for each
    (user_state, notification_type) pair, as defined in env.py.

    Args:
        state: Observation with user_state, notification_type, history.

    Returns:
        One of: "show_now", "delay", "mute"
    """
    user = state.user_state
    notif = state.notification_type

    # SLEEPING: mute=+5 is always best (show_now=-10 is catastrophic)
    if user == "sleeping":
        return "mute"

    # STUDYING: each notification type has a clear optimal action
    if user == "studying":
        if notif == "social":
            return "mute"       # +10
        elif notif == "work":
            return "delay"      # +6
        elif notif == "urgent":
            return "show_now"   # +10
        return "delay"          # safe fallback

    # FREE_TIME: show_now=+5 is always best
    if user == "free_time":
        return "show_now"

    # Unknown state fallback
    logger.warning("Unknown user_state: %s — defaulting to delay", user)
    return "delay"


def run_env() -> str:
    """
    Runs the agent across all tasks and returns the full evaluation log.
    """
    output = ""

    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        output += f"\n[START] Task: {task['name']}\n"
        logger.info("Starting task: %s (%d steps)", task["name"], task["steps"])

        for step in range(task["steps"]):
            try:
                action = agent(state)
                if action not in VALID_ACTIONS:
                    logger.error("Invalid action '%s' — falling back to delay", action)
                    action = "delay"
            except Exception as e:
                logger.error("Agent error at step %d: %s — falling back to delay", step, e)
                action = "delay"

            output += f"[STEP] {step} | State: {state} | Action: {action}\n"
            logger.info("Step %d | %s | %s | -> %s", step, state.user_state, state.notification_type, action)

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        score = grade(total_reward, max_possible)
        output += f"[END] Score: {score}\n"
        logger.info("Task '%s' complete — Score: %.4f", task["name"], score)

    return output


demo = gr.Interface(
    fn=run_env,
    inputs=[],
    outputs="text",
    title="Smart Notification Manager AI",
    description="Deterministic reward-maximizing notification agent — OpenEnv Hackathon"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)