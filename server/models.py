from pydantic import BaseModel, Field
from typing import Dict, Any


class HackathonAction(BaseModel):
    category: str | None = None
    policy: str | None = None
    type: str | None = None
    response: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HackathonObservation(BaseModel):
    ticket_text: str = ""
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)