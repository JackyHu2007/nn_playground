"""In-memory session store with simple TTL eviction."""
from __future__ import annotations

import time
import uuid
from threading import Lock

from ..data.types import Example2D
from .session import TrainingSession
from .state import State


class SessionManager:
    def __init__(self, ttl_seconds: float = 3600.0):
        self.ttl = float(ttl_seconds)
        self._sessions: dict[str, tuple[TrainingSession, float]] = {}
        self._custom_datasets: dict[str, tuple[list[Example2D], float]] = {}
        self._lock = Lock()

    def create(self, state: State | None = None) -> tuple[str, TrainingSession]:
        if state is None:
            state = State()
        session = TrainingSession(state, manager=self)
        sid = str(uuid.uuid4())
        with self._lock:
            self._sessions[sid] = (session, time.monotonic())
        return sid, session

    def get(self, sid: str) -> TrainingSession | None:
        with self._lock:
            entry = self._sessions.get(sid)
            if entry is None:
                return None
            session, _ = entry
            self._sessions[sid] = (session, time.monotonic())
            return session

    def delete(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    # ---- custom datasets ----
    def add_custom(self, points: list[Example2D]) -> str:
        did = str(uuid.uuid4())
        with self._lock:
            self._custom_datasets[did] = (list(points), time.monotonic())
        return did

    def get_custom(self, did: str) -> list[Example2D] | None:
        with self._lock:
            entry = self._custom_datasets.get(did)
            if entry is None:
                return None
            points, _ = entry
            self._custom_datasets[did] = (points, time.monotonic())
            return list(points)

    def cleanup(self) -> int:
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired = [sid for sid, (_, ts) in self._sessions.items()
                       if now - ts > self.ttl]
            for sid in expired:
                del self._sessions[sid]
            removed += len(expired)
            expired_custom = [did for did, (_, ts) in self._custom_datasets.items()
                              if now - ts > self.ttl]
            for did in expired_custom:
                del self._custom_datasets[did]
            removed += len(expired_custom)
        return removed

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)