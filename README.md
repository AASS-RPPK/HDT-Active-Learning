# HDT Active Learning (FastAPI)

This service is a simulated backend for the Active Learning UI. It implements the endpoints shown in your screenshot:

- `POST /feedback` (upload expert feedback)
- `GET /feedback` (retrieve queued feedback candidates)
- `POST /mods/annotation/train` (start fine-tuning simulation)
- `POST /mods/annotation/deploy` (deploy/rollback simulation)

It also provides `/models/annotation/*` aliases since the web frontend calls those paths.

## Run

1. Install dependencies
   - `pip install -r requirements.txt`
2. Start the server
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Default CORS is wide open (`allow_origins=["*"]`) for local/dev use.

Note: the queue and model state are stored in-memory and reset on server restart.

## Endpoints

### Health

`GET /health`

Response:
```json
{ "status": "ok" }
```

### Feedback queue

`GET /feedback`

Returns an array of queued items:
```json
[
  {
    "id": "uuid",
    "caseId": "CASE-1234",
    "slideId": "SLIDE-77",
    "text": "string",
    "revisedAnnotation": "string",
    "confidence": 0.42,
    "label": "Review",
    "suggestedAnnotation": "AI suggestion (simulated)",
    "imageFilename": null,
    "ts": 1710000000
  }
]
```

`POST /feedback`

Content-Type: `multipart/form-data`

Form fields:
- `caseId` (string, optional)
- `slideId` (string, optional)
- `text` (string, optional)
- `revisedAnnotation` (string, optional)
- `image` (file, optional)

Response:
```json
{
  "status": "accepted",
  "id": "uuid",
  "message": "Feedback uploaded (simulated)."
}
```

### Fine-tuning (simulated)

`POST /mods/annotation/train`
`POST /models/annotation/train` (alias)

Body (JSON):
```json
{
  "mode": "active-learning",
  "epochs": 3,
  "learningRate": 0.0001,
  "batchSize": 8
}
```

Behavior:
- Waits about `1.5s` to simulate a training job.

Response (example):
```json
{
  "status": "ok",
  "message": "Fine-tuning started (simulated).",
  "jobId": "uuid",
  "mode": "active-learning",
  "epochs": 3,
  "learningRate": 0.0001,
  "batchSize": 8
}
```

### Deploy / Rollback (simulated)

`POST /mods/annotation/deploy`
`POST /models/annotation/deploy` (alias)

Body (JSON):
```json
{
  "action": "deploy",
  "modelVersion": "optional-version-string"
}
```

Behavior:
- Waits about `1.0s` to simulate deploy/rollback.
- For `deploy`: adds a new model version to in-memory history.
- For `rollback`: steps back one version (if history allows).

Response (example):
```json
{
  "status": "ok",
  "message": "Deploy requested (simulated).",
  "action": "deploy",
  "modelVersion": "v0.1.3"
}
```

## Quick curl examples

Upload feedback:
```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "caseId=CASE-1024" \
  -F "slideId=SLIDE-77A" \
  -F "text=Expert input text" \
  -F "revisedAnnotation=Revised annotation"
```

Retrieve queue:
```bash
curl "http://localhost:8000/feedback"
```

Start fine-tuning:
```bash
curl -X POST "http://localhost:8000/models/annotation/train" \
  -H "Content-Type: application/json" \
  -d '{"mode":"active-learning","epochs":3,"learningRate":0.0001,"batchSize":8}'
```

Deploy/rollback:
```bash
curl -X POST "http://localhost:8000/models/annotation/deploy" \
  -H "Content-Type: application/json" \
  -d '{"action":"rollback"}'
```

