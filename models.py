from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    user_state: str
    notification_type: str
    history: List[str]

class Action(BaseModel):
    action: str

class Reward(BaseModel):
    value: float