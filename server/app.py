from fastapi import FastAPI
from env import NotificationEnv

app = FastAPI()

env = NotificationEnv()

@app.get("/reset")
def reset():
    state = env.reset()
    return {"state": str(state)}

@app.post("/step")
def step(action: str):
    next_state, reward, done, info = env.step(action)
    return {
        "next_state": str(next_state),
        "reward": reward,
        "done": done,
        "info": info
    }
