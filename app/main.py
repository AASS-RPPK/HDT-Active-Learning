from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Kafka configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Topics consumed
TOPIC_FEEDBACK_REQUEST  = "feedback.request"
TOPIC_MODEL_TRAIN_DEPLOY = "model.train.deploy"

# Topics produced
TOPIC_AI_PREDICTION_RETRIEVE = "aiPrediction.retrieve"
TOPIC_FEEDBACK_RESPONSE      = "feedback.response"

logger = logging.getLogger("hdt.active_learning")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
queue_lock    = asyncio.Lock()
feedback_queue: list[dict[str, Any]] = []

training_lock = asyncio.Lock()
_training_job: dict[str, Any] = {
    "jobId":       None,
    "state":       "idle",
    "startedAt":   None,
    "completedAt": None,
}

deployment_lock = asyncio.Lock()
_current_model_version: str = "v0.0.0"
_model_history: list[str]   = ["v0.0.0"]

# ---------------------------------------------------------------------------
# Shared Kafka producer (initialized in lifespan)
# ---------------------------------------------------------------------------
_producer: AIOKafkaProducer | None = None


def _seed_queue_if_empty() -> None:
    if feedback_queue:
        return
    labels = ["Review", "Correction", "Uncertain"]
    for idx in range(3):
        feedback_queue.append({
            "id":                 str(uuid.uuid4()),
            "caseId":             f"CASE-{1000 + idx}",
            "slideId":            f"SLIDE-{77 + idx}",
            "text":               f"Sample expert input text #{idx + 1}",
            "revisedAnnotation":  f"Revised annotation #{idx + 1}",
            "confidence":         round(random.random(), 3),
            "label":              labels[idx % len(labels)],
            "suggestedAnnotation":f"AI suggestion for item #{idx + 1}",
            "ts":                 int(time.time()),
        })


def _queue_item_from_upload(
    *,
    case_id: str,
    slide_id: str,
    text: str,
    revised_annotation: str,
    image: Optional[UploadFile],
) -> dict[str, Any]:
    return {
        "id":                 str(uuid.uuid4()),
        "caseId":             case_id.strip()  or f"CASE-{random.randint(10000, 99999)}",
        "slideId":            slide_id.strip() or f"SLIDE-{random.randint(100, 999)}",
        "text":               text.strip()     or "Uploaded expert input text",
        "revisedAnnotation":  revised_annotation.strip() or "Uploaded revised annotation",
        "confidence":         round(random.random(), 3),
        "label":              "Review",
        "suggestedAnnotation":"AI suggestion (simulated)",
        "imageFilename":      image.filename if image is not None else None,
        "ts":                 int(time.time()),
    }


async def _simulate_training(mode: str, epochs: int, lr: float, batch: int) -> dict[str, Any]:
    async with training_lock:
        job_id = str(uuid.uuid4())
        _training_job.update({
            "jobId":       job_id,
            "state":       "logged",
            "startedAt":   int(time.time()),
            "completedAt": int(time.time()),
            "params":      {"mode": mode, "epochs": epochs, "learningRate": lr, "batchSize": batch},
        })
    logger.info("Train request logged (no-op): mode=%s epochs=%d", mode, epochs)
    return {
        "status":       "ok",
        "message":      "Train request logged (no-op).",
        "jobId":        job_id,
        "mode":         mode,
        "epochs":       epochs,
        "learningRate": lr,
        "batchSize":    batch,
    }


async def _simulate_deploy(action: str, model_version: Optional[str]) -> dict[str, Any]:
    global _current_model_version
    async with deployment_lock:
        version_after = (model_version or _current_model_version or "v0.0.0").strip() or "v0.0.0"
        _current_model_version = version_after
    logger.info("Deploy request logged (no-op): action=%s version=%s", action, version_after)
    return {
        "status":       "ok",
        "message":      "Deploy request logged (no-op)." if action == "deploy" else "Rollback request logged (no-op).",
        "action":       action,
        "modelVersion": version_after,
        "currentVersion": version_after,
    }


# ---------------------------------------------------------------------------
# Kafka consumers
# ---------------------------------------------------------------------------

async def _consume_feedback_request(producer: AIOKafkaProducer) -> None:
    """
    Consume feedback.request → record feedback in queue → produce aiPrediction.retrieve.
    This is step 1 of the feedback pipeline.
    """
    consumer = AIOKafkaConsumer(
        TOPIC_FEEDBACK_REQUEST,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="active-learning-feedback-request",
        auto_offset_reset="latest",
    )
    await consumer.start()
    logger.info("Consumer started: %s", TOPIC_FEEDBACK_REQUEST)
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode())
            task_id = data.get("taskId", "")
            item = _queue_item_from_upload(
                case_id=data.get("caseId", ""),
                slide_id=data.get("slideId", ""),
                text=data.get("feedbackText", ""),
                revised_annotation=data.get("revisedAnnotation", ""),
                image=None,
            )
            async with queue_lock:
                feedback_queue.append(item)
            logger.info("[kafka] feedback.request recorded taskId=%s feedbackId=%s", task_id, item["id"])
            # Forward to AI Prediction for annotation retrieval
            await producer.send_and_wait(
                TOPIC_AI_PREDICTION_RETRIEVE,
                json.dumps({
                    "taskId":     task_id,
                    "feedbackId": item["id"],
                    "caseId":     item["caseId"],
                    "slideId":    item["slideId"],
                    "log":        "Feedback recorded; triggering AI Prediction retrieval.",
                }).encode(),
            )
            logger.info("[kafka] aiPrediction.retrieve produced for taskId=%s", task_id)
    finally:
        await consumer.stop()


async def _consume_model_train_deploy(producer: AIOKafkaProducer) -> None:
    """
    Consume model.train.deploy → simulate train+deploy → produce feedback.response.
    This is the final step of the feedback pipeline.
    """
    consumer = AIOKafkaConsumer(
        TOPIC_MODEL_TRAIN_DEPLOY,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="active-learning-model-train-deploy",
        auto_offset_reset="latest",
    )
    await consumer.start()
    logger.info("Consumer started: %s", TOPIC_MODEL_TRAIN_DEPLOY)
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode())
            task_id = data.get("taskId", "")
            train_result  = await _simulate_training("active-learning", 3, 0.0001, 8)
            deploy_result = await _simulate_deploy("deploy", None)
            deployed_version = deploy_result["currentVersion"]
            logger.info("[kafka] model.train.deploy processed taskId=%s version=%s", task_id, deployed_version)
            # Signal pipeline completion back to the Workflow Producer
            await producer.send_and_wait(
                TOPIC_FEEDBACK_RESPONSE,
                json.dumps({
                    "taskId":               task_id,
                    "outcome":              "completed",
                    "deployedModelVersion": deployed_version,
                    "trainingJobId":        train_result["jobId"],
                    "log":                  "Model trained and deployed (simulated).",
                }).encode(),
            )
            logger.info("[kafka] feedback.response produced for taskId=%s", task_id)
    finally:
        await consumer.stop()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer

    _seed_queue_if_empty()

    # Connect Kafka producer
    logger.info("Connecting to Kafka at %s …", KAFKA_BOOTSTRAP)
    for attempt in range(1, 31):
        try:
            _producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
            await _producer.start()
            logger.info("Kafka producer connected (attempt %d).", attempt)
            break
        except Exception as exc:
            logger.warning("Kafka not ready (attempt %d/30): %s", attempt, exc)
            _producer = None
            await asyncio.sleep(5)
    else:
        logger.error("Could not connect to Kafka — Kafka consumers not started.")
        yield
        return

    # Start consumer background tasks
    kafka_tasks = [
        asyncio.create_task(_consume_feedback_request(_producer),   name="consumer-feedback-request"),
        asyncio.create_task(_consume_model_train_deploy(_producer),  name="consumer-model-train-deploy"),
    ]
    logger.info("Started %d Kafka consumer tasks.", len(kafka_tasks))

    yield

    for t in kafka_tasks:
        t.cancel()
    await asyncio.gather(*kafka_tasks, return_exceptions=True)
    await _producer.stop()
    logger.info("Kafka producer stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="HDT Active Learning (FastAPI)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    mode: str = Field(default="active-learning")
    epochs: int = Field(default=3, ge=1)
    learningRate: float = Field(default=0.0001, gt=0)
    batchSize: int = Field(default=8, ge=1)


class DeployRequest(BaseModel):
    action: Literal["deploy", "rollback"] = "deploy"
    modelVersion: Optional[str] = None


# ---------------------------------------------------------------------------
# REST endpoints (unchanged public API)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/feedback")
async def get_feedback() -> list[dict[str, Any]]:
    async with queue_lock:
        return list(feedback_queue)


@app.post("/feedback")
async def upload_feedback(
    caseId:             str          = Form(default=""),
    slideId:            str          = Form(default=""),
    text:               str          = Form(default=""),
    revisedAnnotation:  str          = Form(default=""),
    image:              Optional[UploadFile] = File(default=None),
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
    return {"status": "accepted", "id": item["id"], "message": "Feedback uploaded (simulated)."}


@app.post("/models/annotation/train")
async def train_model(req: TrainingRequest) -> dict[str, Any]:
    return await _simulate_training(req.mode, req.epochs, req.learningRate, req.batchSize)


@app.post("/mods/annotation/train")
async def train_model_alias(req: TrainingRequest) -> dict[str, Any]:
    return await _simulate_training(req.mode, req.epochs, req.learningRate, req.batchSize)


@app.post("/models/annotation/deploy")
async def deploy_model(req: DeployRequest) -> dict[str, Any]:
    return await _simulate_deploy(req.action, req.modelVersion)


@app.post("/mods/annotation/deploy")
async def deploy_model_alias(req: DeployRequest) -> dict[str, Any]:
    return await _simulate_deploy(req.action, req.modelVersion)
