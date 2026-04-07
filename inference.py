from fastapi import FastAPI, Request
from env import NotificationEnv

env = NotificationEnv()
app = FastAPI()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs

@app.post("/step")
async def step(request: Request):
    # Safe robust parsing for JSON bodies (like {"action": "mute"})
    try:
        data = await request.json()
    except Exception:
        data = {}
        
    action = data.get("action") if isinstance(data, dict) else None
    
    # Fallback to Query Parameters (like ?action=mute)
    if action is None:
        action = request.query_params.get("action")
        
    # Execute the environment step
    obs, reward, done, info = env.step(action)
    
    return {
        "observation": obs,
        "reward": reward,
        "done": done
    }