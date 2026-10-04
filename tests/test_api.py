"""API-level tests: POST /jobs idempotency + GET shape (no worker)."""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from common import config as config_mod
from jobs.api import create_app
from jobs.store import JobStore

ROOT = str(Path(__file__).resolve().parent.parent)
DB = str(Path("/tmp") / "test_api.sqlite3")


def _client():
    if os.path.exists(DB):
        os.remove(DB)
    return TestClient(create_app(store=JobStore(DB), worker=None))


def test_post_twice_returns_same_job_and_duplicate_flag():
    client = _client()
    body = {"upload_id": "u1", "video_path": ROOT + "/input_video.mp4",
            "content_sha256": "fixed-digest-for-api-test"}
    first = client.post("/jobs", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/jobs", json=body)
    assert second.status_code == 200, second.text
    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    got = client.get("/jobs/" + first.json()["job_id"])
    assert got.status_code == 200
    assert got.json()["job_id"] == first.json()["job_id"]
    assert got.json()["status"] == "QUEUED"


def test_config_change_is_new_job():
    client = _client()
    base = {"upload_id": "u2", "video_path": ROOT + "/input_video.mp4",
            "content_sha256": "fixed-digest-2"}
    altered = dict(base)
    altered["config"] = {"anonymizer": {"det_threshold": 0.15}}
    job_a = client.post("/jobs", json=base).json()
    job_b = client.post("/jobs", json=altered).json()
    assert job_a["job_id"] != job_b["job_id"]


def test_missing_video_is_400():
    client = _client()
    resp = client.post("/jobs", json={"upload_id": "u3",
                                      "video_path": "/nope/missing.mp4"})
    assert resp.status_code == 400
