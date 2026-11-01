"""WebSocket endpoint driving a TrainingSession."""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from ..session.manager import SessionManager
from ..session.session import TrainingSession
from ..session.state import State
from .schemas import Command, ErrorMessage, WelcomeMessage

logger = logging.getLogger(__name__)
router = APIRouter()

# How often (in steps) the play loop refreshes decision boundaries.
# Decision-boundary computation is the dominant cost; 5 steps is visually
# indistinguishable at 30 fps while cutting per-step work significantly.
BOUNDARY_EVERY_N_STEPS = 5

# Sessions currently owned by an open WebSocket. A second connection to the
# same session would build its own Runner (hence its own lock) and mutate the
# same TrainingSession concurrently, so we refuse to rejoin an owned session.
_active_sessions: set[str] = set()


def _safe_put(queue: asyncio.Queue, item: dict[str, Any]) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass


class Runner:
    def __init__(self, session: TrainingSession):
        self.session = session
        self._lock = asyncio.Lock()

    async def call(self, fn, *args, **kwargs):
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(fn, *args, **kwargs)
            )

    async def frame_now(self, include_boundaries: bool = True) -> dict[str, Any]:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.session.get_frame, include_boundaries
            )


class Player:
    def __init__(self, runner: Runner):
        self.runner = runner
        self._task: Optional[asyncio.Task] = None
        self._playing = False
        self.fps: float = 30.0
        self.include_boundaries: bool = True
        self._step_count = 0

    def is_playing(self) -> bool:
        return self._playing and self._task is not None and not self._task.done()

    def start(self, queue: asyncio.Queue) -> None:
        if self.is_playing():
            return
        self._playing = True
        self._step_count = 0
        self._task = asyncio.create_task(self._loop(queue))

    async def _loop(self, queue: asyncio.Queue) -> None:
        try:
            while self._playing:
                start = asyncio.get_running_loop().time()
                update_b = (self._step_count % BOUNDARY_EVERY_N_STEPS == 0)
                await self.runner.call(self.runner.session.one_step, update_b)
                frame = await self.runner.frame_now(self.include_boundaries)
                _safe_put(queue, frame)
                self._step_count += 1

                elapsed = asyncio.get_running_loop().time() - start
                delay = max(0.0, 1.0 / self.fps - elapsed)
                if delay > 0:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("player crashed")
        finally:
            self._playing = False

    async def stop(self) -> None:
        self._playing = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _sender(ws: WebSocket, queue: asyncio.Queue) -> None:
    try:
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
    except asyncio.CancelledError:
        raise
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("sender crashed")


async def _handle_command(
        cmd: Command,
        runner: Runner,
        player: Player,
        queue: asyncio.Queue,
        ws: WebSocket,
) -> None:
    session = runner.session

    if cmd.cmd == "step":
        await player.stop()
        n = cmd.n or 1
        for k in range(n):
            await runner.call(session.one_step, k == n - 1)
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "play":
        player.fps = cmd.fps
        player.include_boundaries = cmd.include_boundaries
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))
        player.start(queue)

    elif cmd.cmd == "pause":
        await player.stop()

    elif cmd.cmd == "reset":
        await player.stop()
        await runner.call(session.reset)
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "regenerate":
        await player.stop()
        await runner.call(session.regenerate)
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "set":
        if cmd.key is None:
            raise ValueError("'set' requires a 'key'")
        await runner.call(session.set, cmd.key, cmd.value)
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "load_state":
        if not isinstance(cmd.state, dict):
            raise ValueError("'load_state' requires a 'state' dict")
        new_state = State.from_dict(cmd.state)
        await player.stop()
        await runner.call(session.load_state, new_state)
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "request_frame":
        # Reading the session while the play loop is mutating it races on
        # network/weights/boundaries; the play loop already pushes frames.
        if player.is_playing():
            return
        _safe_put(queue, await runner.frame_now(cmd.include_boundaries))

    elif cmd.cmd == "request_node_boundary":
        # Zoom request: only valid while paused. Sent directly (bypasses the
        # frame queue) so it cannot be evicted by a full queue.
        if player.is_playing():
            await ws.send_json(
                ErrorMessage(message="cannot zoom while playing; pause first").model_dump()
            )
        elif not cmd.node_id:
            await ws.send_json(
                ErrorMessage(message="'request_node_boundary' requires 'node_id'").model_dump()
            )
        else:
            boundary = await runner.call(session.get_node_boundary, cmd.node_id)
            if boundary is None:
                await ws.send_json(
                    ErrorMessage(message=f"unknown node {cmd.node_id!r}").model_dump()
                )
            else:
                await ws.send_json({
                    "type": "node_boundary",
                    "nodeId": cmd.node_id,
                    "boundary": boundary,
                })

    elif cmd.cmd == "ping":
        _safe_put(queue, {"type": "pong"})


def get_manager(websocket: WebSocket) -> SessionManager:
    return websocket.app.state.session_manager


@router.websocket("/ws")
async def websocket_endpoint(
        websocket: WebSocket,
        session: Optional[str] = Query(default=None),
        manager: SessionManager = Depends(get_manager),
) -> None:
    await websocket.accept()

    runner: Optional[Runner] = None
    player: Optional[Player] = None
    sender_task: Optional[asyncio.Task] = None
    session_id: Optional[str] = None

    try:
        if session:
            existing = manager.get(session)
            # Only rejoin when no other live connection owns this session;
            # otherwise two Runners would mutate one TrainingSession under
            # different locks.
            if existing is not None and session not in _active_sessions:
                session_id = session
                runner = Runner(existing)
        if runner is None:
            session_id, fresh = manager.create(State())
            runner = Runner(fresh)
        _active_sessions.add(session_id)

        # If the persisted state references a custom dataset that no longer
        # exists (e.g. server restarted), drop it so the session doesn't
        # break on the first frame.
        try:
            did = runner.session.state.custom_dataset_id
            if did and manager.get_custom(did) is None:
                runner.session.state.custom_dataset_id = None
                runner.session.set_custom_dataset_by_id(None)
        except Exception:
            logger.exception("custom dataset recovery failed")

        player = Player(runner)
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        await websocket.send_json(
            WelcomeMessage(
                sessionId=session_id,
                state=runner.session.state.to_dict(),
            ).model_dump()
        )
        _safe_put(queue, await runner.frame_now())

        sender_task = asyncio.create_task(_sender(websocket, queue))

        while True:
            raw = await websocket.receive_json()
            try:
                cmd = Command.model_validate(raw)
            except Exception as exc:
                _safe_put(queue, ErrorMessage(message=str(exc)).model_dump())
                continue
            try:
                await _handle_command(cmd, runner, player, queue, websocket)
            except Exception as exc:
                logger.exception("command failed: %s", raw)
                _safe_put(queue, ErrorMessage(message=str(exc)).model_dump())

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("websocket error")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if session_id:
            _active_sessions.discard(session_id)
        if player is not None:
            await player.stop()
        if sender_task is not None:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass