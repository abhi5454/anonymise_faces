"""FastAPI job service: POST /jobs, GET /jobs/{id}, GET /health."""

import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import config as config_mod
from common.hashing import sha256_file
from jobs.store import JobStore

UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR", "/Users/abhisheklal/Workspace/anonymise_faces/uploads")
DB_PATH = os.environ.get(
    "JOBS_DB", "/Users/abhisheklal/Workspace/anonymise_faces/jobs.sqlite3")
OUT_DIR = os.environ.get(
    "ARTIFACT_DIR", "/Users/abhisheklal/Workspace/anonymise_faces/artifacts")
CROPS_DIR = os.environ.get(
    "CROPS_DIR", "/Users/abhisheklal/Workspace/anonymise_faces/crops")


class JobRequest(BaseModel):
    upload_id: str
    content_sha256: str | None = None
    video_path: str | None = None
    config: dict | None = None


def create_app(store=None, worker=None):
    """Build the FastAPI app. Worker wiring stays outside for testability.

    With ``ANON_WORKER=1`` an in-process worker is started so
    ``uvicorn jobs.api:app`` is a complete service (stand-in for SQS/Celery).
    Tests keep passing ``worker=None`` and stay single-process and fast.
    """
    store = store or JobStore(DB_PATH)
    if worker is None and os.environ.get("ANON_WORKER") == "1":
        from jobs.worker import Worker

        worker = Worker(store, out_dir=OUT_DIR, crops_dir=CROPS_DIR)
        worker.start()
    app = FastAPI(title="anonymise-faces jobs")
    app.state.store = store
    app.state.worker = worker

    @app.get("/health")
    def health():
        return {"status": "ok", "pipeline_version": config_mod.PIPELINE_VERSION}

    @app.post("/jobs")
    def post_job(req: JobRequest):
        if not req.video_path or not os.path.isfile(req.video_path):
            raise HTTPException(400, "video_path must be an existing file")
        digest = req.content_sha256 or sha256_file(req.video_path)
        cfg = config_mod.canonical_config(req.config)
        key = config_mod.content_idempotency_key(req.content_sha256 or digest,
                                                 req.config)
        job_id = "job_" + key[:12]
        row, duplicate = app.state.store.create_or_get(
            job_id, req.upload_id, digest, config_mod.PIPELINE_VERSION,
            config_mod.canonical_json(cfg), key, req.video_path)
        if not duplicate and app.state.worker is not None:
            app.state.worker.submit(row["job_id"])
        return {"job_id": row["job_id"], "status": row["status"],
                "idempotency_key": key, "duplicate": duplicate}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        row = app.state.store.get(job_id)
        if row is None:
            raise HTTPException(404, "unknown job")
        return app.state.store.public_view(row)

    return app


app = create_app()
