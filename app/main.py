from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="HDT Active Learning (FastAPI)")

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
        _training_job.update(
            {
                "jobId": job_id,
                "state": "running",
                "startedAt": int(time.time()),
                "completedAt": None,
                "params": req.model_dump(),
            }
        )

    # Simulate a short training window.
    await asyncio.sleep(1.5)

    async with training_lock:
        _training_job["state"] = "completed"
        _training_job["completedAt"] = int(time.time())

    return {
        "status": "ok",
        "message": "Fine-tuning started (simulated).",
        "jobId": job_id,
        "mode": req.mode,
        "epochs": req.epochs,
        "learningRate": req.learningRate,
        "batchSize": req.batchSize,
    }


async def _simulate_deploy(req: DeployRequest) -> dict[str, Any]:
    global _current_model_version
    async with deployment_lock:
        if req.action == "deploy":
            # Move forward: create a "new" model version.
            version_hint = (req.modelVersion or "").strip()
            if version_hint:
                new_version = version_hint
            else:
                major = len(_model_history)
                new_version = f"v0.{major}.{random.randint(0, 9)}"

            _current_model_version = new_version
            _model_history.append(new_version)
        else:
            # Roll back: pop to previous version if possible.
            if len(_model_history) > 1:
                _model_history.pop()
                _current_model_version = _model_history[-1]

        model_version_after = _current_model_version

    # Simulate a short deploy/rollback window.
    await asyncio.sleep(1.0)

    return {
        "status": "ok",
        "message": "Deploy requested (simulated)." if req.action == "deploy" else "Rollback requested (simulated).",
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

