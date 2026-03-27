from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="HDT Active Learning (FastAPI)")
logger = logging.getLogger("hdt.active_learning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingRequest(BaseModel):
    mode: str = Field(default="active-learning")
    epochs: int = Field(default=3, ge=1)
    learningRate: float = Field(default=0.0001, gt=0)
    batchSize: int = Field(default=8, ge=1)


class DeployRequest(BaseModel):
    action: Literal["deploy", "rollback"] = "deploy"
    modelVersion: Optional[str] = None


# In-memory simulation store (good enough for the dashboard & local dev).
queue_lock = asyncio.Lock()
feedback_queue: list[dict[str, Any]] = []

training_lock = asyncio.Lock()
_training_job: dict[str, Any] = {
    "jobId": None,
    "state": "idle",
    "startedAt": None,
    "completedAt": None,
}

deployment_lock = asyncio.Lock()
_current_model_version: str = "v0.0.0"
_model_history: list[str] = ["v0.0.0"]


def _seed_queue_if_empty() -> None:
    # Seed a few fake candidates so the UI has something to show immediately.
    # The dashboard will also display any uploads performed by the user.
    if feedback_queue:
        return

    labels = ["Review", "Correction", "Uncertain"]
    for idx in range(3):
        feedback_queue.append(
            {
                "id": str(uuid.uuid4()),
                "caseId": f"CASE-{1000 + idx}",
                "slideId": f"SLIDE-{77 + idx}",
                "text": f"Sample expert input text #{idx + 1}",
                "revisedAnnotation": f"Revised annotation #{idx + 1}",
                "confidence": round(random.random(), 3),
                "label": labels[idx % len(labels)],
                "suggestedAnnotation": f"AI suggestion for item #{idx + 1}",
                "ts": int(time.time()),
            }
        )


@app.on_event("startup")
def on_startup() -> None:
    _seed_queue_if_empty()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _queue_item_from_upload(
    *,
    case_id: str,
    slide_id: str,
    text: str,
    revised_annotation: str,
    image: Optional[UploadFile],
) -> dict[str, Any]:
    # Create a dashboard-friendly shape. The UI tolerates missing fields.
    case_id_value = case_id.strip() or f"CASE-{random.randint(10000, 99999)}"
    slide_id_value = slide_id.strip() or f"SLIDE-{random.randint(100, 999)}"
    text_value = text.strip() or "Uploaded expert input text"
    revised_value = revised_annotation.strip() or "Uploaded revised annotation"

    return {
        "id": str(uuid.uuid4()),
        "caseId": case_id_value,
        "slideId": slide_id_value,
        "text": text_value,
        "revisedAnnotation": revised_value,
        "confidence": round(random.random(), 3),
        "label": "Review",
        "suggestedAnnotation": "AI suggestion (simulated)",
        "imageFilename": image.filename if image is not None else None,
        "ts": int(time.time()),
    }


@app.get("/feedback")
async def get_feedback() -> list[dict[str, Any]]:
    async with queue_lock:
        # Return a shallow copy so callers can't mutate our internal list.
        return list(feedback_queue)


@app.post("/feedback")
async def upload_feedback(
    caseId: str = Form(default=""),
    slideId: str = Form(default=""),
    text: str = Form(default=""),
    revisedAnnotation: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
) -> dict[str, Any]:
    item = _queue_item_from_upload(
        case_id=caseId,
        slide_id=slideId,
        text=text,
        revised_annotation=revisedAnnotation,
        image=image,
    )

    async with queue_lock:
        feedback_queue.append(item)

    return {
        "status": "accepted",
        "id": item["id"],
        "message": "Feedback uploaded (simulated).",
    }


async def _simulate_training(req: TrainingRequest) -> dict[str, Any]:
    async with training_lock:
        job_id = str(uuid.uuid4())
        payload = req.model_dump()
        _training_job.update(
            {
                "jobId": job_id,
                "state": "logged",
                "startedAt": int(time.time()),
                "completedAt": int(time.time()),
                "params": payload,
            }
        )
    logger.info("Received train request (no-op): %s", payload)

    return {
        "status": "ok",
        "message": "Train request logged (no-op).",
        "jobId": job_id,
        "mode": req.mode,
        "epochs": req.epochs,
        "learningRate": req.learningRate,
        "batchSize": req.batchSize,
    }


async def _simulate_deploy(req: DeployRequest) -> dict[str, Any]:
    global _current_model_version
    async with deployment_lock:
        payload = req.model_dump()
        model_version_after = (req.modelVersion or _current_model_version or "v0.0.0").strip() or "v0.0.0"
        _current_model_version = model_version_after
    logger.info("Received deploy request (no-op): %s", payload)

    return {
        "status": "ok",
        "message": "Deploy request logged (no-op)." if req.action == "deploy" else "Rollback request logged (no-op).",
        "action": req.action,
        "modelVersion": model_version_after,
    }


@app.post("/models/annotation/train")
async def train_model(req: TrainingRequest) -> dict[str, Any]:
    return await _simulate_training(req)


@app.post("/mods/annotation/train")
async def train_model_alias(req: TrainingRequest) -> dict[str, Any]:
    # Alias to match the screenshot path (mods vs models).
    return await _simulate_training(req)


@app.post("/models/annotation/deploy")
async def deploy_model(req: DeployRequest) -> dict[str, Any]:
    return await _simulate_deploy(req)


@app.post("/mods/annotation/deploy")
async def deploy_model_alias(req: DeployRequest) -> dict[str, Any]:
    # Alias to match the screenshot path (mods vs models).
    return await _simulate_deploy(req)

