"""
OpenEnv API Server — Smart Notification Manager AI

Exposes /reset, /step, and /state endpoints for OpenEnv validator compliance.
"""

from fastapi import FastAPI, HTTPException, Request
from env import NotificationEnv

app = FastAPI(
    title="Smart Notification Manager AI",
    description="OpenEnv-compliant API for the notification management environment.",
    version="1.0.0"
)

env = NotificationEnv()





VALID_ACTIONS = {"show_now", "delay", "mute"}


@app.post("/reset")
def reset():
    """Reset the environment and return the initial state."""
    state = env.reset()
    return {
        "user_state": state.user_state,
        "notification_type": state.notification_type,
        "history": state.history
    }


@app.post("/step")
async def step(request: Request):
    """Take a step — accepts action via JSON body or query param. Crash-safe."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    action = data.get("action") if isinstance(data, dict) else None

    if action is None:
        action = request.query_params.get("action")

    if action is None:
        return {"error": "action is required"}

    if action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action}'. Must be one of: {sorted(VALID_ACTIONS)}"
        )

    state, reward, done, _ = env.step(action)
    return {
        "user_state": state.user_state,
        "notification_type": state.notification_type,
        "history": state.history,
        "reward": reward,
        "done": done
    }


@app.get("/state")
def get_state():
    """Return the current environment state."""
    current = env.current_state
    if current is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return {
        "user_state": current.user_state,
        "notification_type": current.notification_type,
        "history": current.history
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
