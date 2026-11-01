"""REST endpoints (lightweight; the real traffic goes through the WS)."""
from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..core.seed import make_rng
from ..data.registry import DATASETS, MULTI_DATASETS, REG_DATASETS
from ..data.types import Example2D
from ..session.state import State
from .schemas import CustomDatasetUploadRequest


router = APIRouter()

_MAX_CUSTOM_POINTS = 50_000
_MAX_CLASSES = 12


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/state/default")
async def default_state() -> dict[str, Any]:
    return State().to_dict()


@router.get("/datasets")
async def datasets() -> dict[str, list[dict[str, Any]]]:
    """Return metadata + a 200-point preview for every dataset."""
    result: list[dict[str, Any]] = []
    for category, mapping in (
        ("classification", DATASETS),
        ("multiclass", MULTI_DATASETS),
        ("regression", REG_DATASETS),
    ):
        for name, gen in mapping.items():
            rng = make_rng("preview", 0)
            pts = gen.train_generator(200, 0.0, rng)
            result.append({
                "category": category,
                "name": name,
                "numClasses": int(gen.num_classes),
                "isProbabilistic": gen.is_probabilistic,
                "preview": [[p.x, p.y, p.label] for p in pts],
            })
    return {"datasets": result}


@router.get("/commands")
async def commands() -> dict[str, list[str]]:
    from .schemas import ALLOWED_COMMANDS
    return {"commands": sorted(ALLOWED_COMMANDS)}


@router.post("/datasets/custom")
async def upload_custom_dataset(
    body: CustomDatasetUploadRequest,
    request: Request,
) -> dict[str, Any]:
    manager = request.app.state.session_manager

    if not body.points:
        raise HTTPException(status_code=400, detail="no points supplied")
    if len(body.points) > _MAX_CUSTOM_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"too many points (max {_MAX_CUSTOM_POINTS})",
        )

    points: list[Example2D] = []
    for i, p in enumerate(body.points):
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.label)):
            raise HTTPException(
                status_code=400,
                detail=f"point #{i} has non-finite values",
            )
        points.append(Example2D(float(p.x), float(p.y), float(p.label)))

    unique = sorted({p.label for p in points})
    if len(unique) < 2:
        raise HTTPException(
            status_code=400,
            detail="need at least 2 distinct labels",
        )

    is_int = all(abs(u - round(u)) < 1e-9 for u in unique)
    if (is_int and len(unique) >= 3
            and unique[0] == 0.0
            and unique[-1] == float(len(unique) - 1)):
        problem = "multiclass"
        num_classes = len(unique)
    elif len(unique) == 2:
        problem = "classification"
        num_classes = 2
    else:
        problem = "regression"
        num_classes = 1

    if problem == "multiclass" and num_classes > _MAX_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"too many classes (max {_MAX_CLASSES})",
        )

    did = manager.add_custom(points)

    if len(points) <= 200:
        preview = [[p.x, p.y, p.label] for p in points]
    else:
        step = len(points) / 200.0
        preview = [
            [points[int(i * step)].x,
             points[int(i * step)].y,
             points[int(i * step)].label]
            for i in range(200)
        ]

    return {
        "datasetId": did,
        "numPoints": len(points),
        "problem": problem,
        "numClasses": num_classes,
        "preview": preview,
    }