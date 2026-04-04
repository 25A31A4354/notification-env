import os
import gradio as gr
from openai import OpenAI
from env import NotificationEnv
from grader import grade
from tasks import TASKS

def simple_agent(state):
    client_kwargs = {}
    if os.environ.get("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
    if os.environ.get("API_BASE_URL"):
        client_kwargs["base_url"] = os.environ.get("API_BASE_URL")
        
    try:
        client = OpenAI(**client_kwargs)
        model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
        
        system_prompt = (
            "You are a smart notification manager. "
            "Your output must be exactly one of the following words and nothing else: "
            "show_now, delay, mute."
        )
        
        user_prompt = (
            "Decide the best action for the current notification based on these rules:\n"
            "- Avoid disturbing the user during studying or sleeping.\n"
            "- Prioritize urgent notifications.\n"
            "- Consider the user's history of notifications.\n\n"
            f"User State: {state.user_state}\n"
            f"Notification Type: {state.notification_type}\n"
            f"History: {state.history}\n\n"
            "Action (show_now/delay/mute):"
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        action = response.choices[0].message.content.strip().lower()
        
        if "show" in action:
            return "show_now"
        elif "mute" in action:
            return "mute"
        elif "delay" in action:
            return "delay"
        else:
            return "delay"
        
    except Exception as e:
        print(f"API Error: {e}")
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