"""Pydantic command schemas for the WebSocket channel."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


ALLOWED_COMMANDS = {
    "step", "play", "pause", "reset", "regenerate",
    "set", "load_state", "request_frame", "request_node_boundary", "ping",
}


class Command(BaseModel):
    cmd: str
    n: Optional[int] = None
    key: Optional[str] = None
    value: Any = None
    state: Optional[dict[str, Any]] = None
    node_id: Optional[str] = None
    include_boundaries: bool = True
    fps: float = 30.0

    @field_validator("cmd")
    @classmethod
    def _check_cmd(cls, v: str) -> str:
        if v not in ALLOWED_COMMANDS:
            raise ValueError(
                f"unknown cmd {v!r}; allowed: {sorted(ALLOWED_COMMANDS)}"
            )
        return v

    @field_validator("fps")
    @classmethod
    def _clamp_fps(cls, v: float) -> float:
        return max(1.0, min(120.0, float(v)))

    @field_validator("n")
    @classmethod
    def _clamp_n(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return max(1, min(1000, int(v)))


class WelcomeMessage(BaseModel):
    type: str = "welcome"
    sessionId: str
    state: dict[str, Any]
    commands: list[str] = Field(default_factory=lambda: sorted(ALLOWED_COMMANDS))


class ErrorMessage(BaseModel):
    type: str = "error"
    message: str


# ---- custom dataset upload ----

class CustomPoint(BaseModel):
    x: float
    y: float
    label: float


class CustomDatasetUploadRequest(BaseModel):
    points: list[CustomPoint]
    name: Optional[str] = None