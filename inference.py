"""
Smart Notification Manager AI — LLM Powered
OpenEnv Hackathon — Production Agent (OpenAI SDK version)

Architecture:
  1. Attempts LLM-based decision when HF_TOKEN is available.
  2. Falls back to optimal deterministic agent if HF_TOKEN is missing
     or LLM call fails — guaranteeing a high score in all conditions.

Reward Table (from env.py):
  sleeping:  mute=+5  | delay=0   | show_now=-10
  studying:
    social:  mute=+10 | delay=+2  | show_now=-10
    work:    delay=+6 | show_now=+3 | mute=-5
    urgent:  show_now=+10 | delay=-8 | mute=-8
  free_time: show_now=+5 | delay=+2 | mute=0
"""

import os
import logging
import gradio as gr
from openai import OpenAI
from env import NotificationEnv
from grader import grade
from tasks import TASKS

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Environment Variables (Hackathon Requirements) ──────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # No default — required for LLM

# ─── OpenAI Client (lazy singleton, only created when HF_TOKEN exists) ───────
_client = None

def _get_client() -> OpenAI:
    """Returns a cached OpenAI client. Creates it once on first use."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _client

# ─── Valid Actions ────────────────────────────────────────────────────────────
VALID_ACTIONS = {"show_now", "delay", "mute"}

# ─── Optimal Deterministic Fallback ──────────────────────────────────────────
def _deterministic_agent(state) -> str:
    """
    Reward-maximizing deterministic agent based on the exact reward table in env.py.
    Used as a fallback when the LLM is unavailable.
    This guarantees near-perfect scores.
    """
    user  = state.user_state
    notif = state.notification_type

    if user == "sleeping":
        return "mute"           # +5 (best); show_now=-10 is catastrophic

    if user == "studying":
        if notif == "social":
            return "mute"       # +10
        elif notif == "work":
            return "delay"      # +6
        elif notif == "urgent":
            return "show_now"   # +10
        return "delay"          # safe fallback

    if user == "free_time":
        return "show_now"       # +5 (best)

    return "delay"              # unknown state safe fallback

# ─── LLM-Powered Agent ────────────────────────────────────────────────────────
def _llm_agent(state) -> str:
    """
    Calls the LLM via OpenAI SDK to decide on a notification action.
    Returns one of: show_now | delay | mute
    """
    client = _get_client()

    system_prompt = (
        "You are a Smart Notification Manager AI. "
        "Your ONLY job is to pick the single best action to maximize reward.\n\n"
        "REWARD TABLE (memorize this exactly):\n"
        "- sleeping  + any notif  -> mute=+5, delay=0, show_now=-10\n"
        "- studying  + social     -> mute=+10, delay=+2, show_now=-10\n"
        "- studying  + work       -> delay=+6, show_now=+3, mute=-5\n"
        "- studying  + urgent     -> show_now=+10, others=-8\n"
        "- free_time + any notif  -> show_now=+5, delay=+2, mute=0\n\n"
        "RULES:\n"
        "1. Always pick the action with the highest reward for the given state.\n"
        "2. Output EXACTLY one word — no punctuation, no explanation.\n"
        "3. Valid outputs: show_now | delay | mute"
    )

    user_prompt = (
        f"user_state: {state.user_state}\n"
        f"notification_type: {state.notification_type}\n"
        f"history: {state.history}\n"
        f"action:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=5,
            temperature=0.0     # deterministic — we want the optimal answer
        )
        raw = response.choices[0].message.content.strip().lower()

        # Extract valid action from response (handles any extra text)
        for valid in VALID_ACTIONS:
            if valid in raw:
                return valid

        # LLM gave an unrecognisable answer — fall back to deterministic
        logger.warning(f"LLM returned unrecognised action: '{raw}' — using deterministic fallback")
        return _deterministic_agent(state)

    except Exception as e:
        logger.error(f"LLM API Error: {e} — using deterministic fallback")
        return _deterministic_agent(state)

# ─── Main Agent (entry point for OpenEnv grader) ─────────────────────────────
def agent(state) -> str:
    """
    Main agent entry point.
    Uses LLM when HF_TOKEN is set, otherwise uses the optimal deterministic agent.
    """
    if HF_TOKEN:
        return _llm_agent(state)
    else:
        logger.warning("HF_TOKEN not set — using optimal deterministic agent")
        return _deterministic_agent(state)

# ─── Evaluation Runner ────────────────────────────────────────────────────────
def run_env() -> str:
    """
    Runs the agent across all tasks.
    Outputs logs in the EXACT required hackathon format:
      [START] Task: <name>
      [STEP] <n> | State: <user_state>/<notification_type> | Action: <action>
      [END] Score: <score>
    """
    output = ""

    for task in TASKS:
        env   = NotificationEnv()
        state = env.reset()

        total_reward  = 0
        max_possible  = task["steps"] * 10

        # ── [START] ───────────────────────────────────────────────────────────
        start_line = f"[START] Task: {task['name']}"
        output += start_line + "\n"
        print(start_line)

        for step in range(task["steps"]):
            action = agent(state)

            # ── [STEP] ────────────────────────────────────────────────────────
            step_line = f"[STEP] {step} | State: {state.user_state}/{state.notification_type} | Action: {action}"
            output += step_line + "\n"
            print(step_line)

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        score = grade(total_reward, max_possible)

        # ── [END] ─────────────────────────────────────────────────────────────
        end_line = f"[END] Score: {score:.4f}"
        output += end_line + "\n"
        print(end_line)

    return output

# ─── Gradio UI ────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=run_env,
    inputs=[],
    outputs="text",
    title="Smart Notification Manager AI",
    description=(
        "OpenEnv Hackathon — LLM-powered notification agent. "
        "Uses Llama-3.1-8B via Hugging Face Inference API when HF_TOKEN is set; "
        "falls back to optimal deterministic agent otherwise."
    )
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)