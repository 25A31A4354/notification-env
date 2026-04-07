"""
Smart Notification Manager AI — LLM Powered
OpenEnv Hackathon — Production Agent (OpenAI SDK version)
"""

import os
import logging
import gradio as gr
from openai import OpenAI
from env import NotificationEnv
from grader import grade
from tasks import TASKS

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 2. Hackathon Requirements: Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default value

# 3. Initialize OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

VALID_ACTIONS = {"show_now", "delay", "mute"}

def get_llm_action(state) -> str:
    """
    Calls the LLM to decide on a notification action.
    """
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set. Falling back to 'delay'.")
        return "delay"

    system_prompt = (
        "You are a Smart Notification Manager. Your goal is to maximize user satisfaction (reward) "
        "by deciding whether to show a notification now, delay it, or mute it.\n"
        "User States:\n"
        "- sleeping: Muting is best (+5), showing now is terrible (-10).\n"
        "- studying: Social (mute=+10, delay=+2, show_now=-10), Work (delay=+6, show_now=+3, mute=-5), "
        "Urgent (show_now=+10, else=-8).\n"
        "- free_time: Showing now is best (+5), delay is okay (+2).\n"
        "Output ONLY the word: 'show_now', 'delay', or 'mute'."
    )

    user_prompt = f"Current State: {state.user_state}\nNotification: {state.notification_type}\nHistory: {state.history}\nAction:"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        action = response.choices[0].message.content.strip().lower()
        
        # Clean up any potential conversational filler
        for valid in VALID_ACTIONS:
            if valid in action:
                return valid
        return "delay"
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return "delay"

def agent(state) -> str:
    """
    Agent wrapper for OpenEnv evaluation.
    """
    return get_llm_action(state)

def run_env() -> str:
    """
    Runs the agent across all tasks and returns the full evaluation log in hackathon format.
    """
    output = ""

    for task in TASKS:
        env = NotificationEnv()
        state = env.reset()

        total_reward = 0
        max_possible = task["steps"] * 10

        # Hackathon Requirement: [START]
        output += f"[START] Task: {task['name']}\n"

        for step in range(task["steps"]):
            action = agent(state)
            
            # Hackathon Requirement: [STEP]
            output += f"[STEP] {step} | State: {state.user_state}/{state.notification_type} | Action: {action}\n"

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        score = grade(total_reward, max_possible)
        # Hackathon Requirement: [END]
        output += f"[END] Score: {score}\n"

    return output

# Gradio Interface for testing
demo = gr.Interface(
    fn=run_env,
    inputs=[],
    outputs="text",
    title="Smart Notification Manager AI",
    description="LLM-powered agent using OpenAI SDK — Hackathon Evaluation Mode"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)