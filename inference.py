from fastapi import FastAPI
from env import NotificationEnv

env = NotificationEnv()
app = FastAPI()

@app.post("/reset")
def reset():
    obs = env.reset()
    # FastAPI automatically serializes Pydantic models (like state/obs) to JSON
    return obs

@app.post("/step")
def step(action: str):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done
    }