"""FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --ws-per-message-deflate
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import router as rest_router
from .api.websocket import router as ws_router
from .session.manager import SessionManager


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session_manager = SessionManager(ttl_seconds=3600.0)

    async def cleanup_loop() -> None:
        while True:
            try:
                await asyncio.sleep(60)
                removed = app.state.session_manager.cleanup()
                if removed:
                    logger.info("evicted %d expired sessions", removed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("cleanup loop error")

    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title="Rocky Playground", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    app.include_router(rest_router, prefix="/api")
    app.include_router(ws_router)

    static_dir = Path(__file__).resolve().parents[1] / "frontend"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()