# Anonymise Faces: two-stage blur + verify pipeline

Two independent stages, a job layer that never says "safe" unless both agree,
and an evaluation harness that is honest about shared failure modes.

## Layout

```text
common/    config + hashing + video metadata
pipeline/  A: SCRFD detect -> IoU track -> pixelate -> ffmpeg render
verify/    B: RetinaFace checks on the rendered output file
jobs/      FastAPI service + SQLite store + background worker + decision gate
eval/      sampling tool, label format, scoring
tests/     idempotency + review-gating + API tests
docs/      shared failure modes, limitations, next steps
```

## Setup

```bash
./retina_env/bin/pip install -r requirements.txt
export DEEPFACE_BACKEND_ENGINE=pytorch   # required before importing retinaface
export PYTHONPATH=$PWD                   # run from the repo root
```

Weights resolve automatically: `~/.insightface/buffalo_l` (SCRFD, Pipeline A)
and `~/.deepface/weights/retinaface.pth` (Pipeline B) are already cached.

## Run

```bash
# API + worker (worker runs in-process; stands in for SQS/Celery)
./retina_env/bin/python -m jobs.run_local --video input_video.mp4 --max-frames 60

# tests (fast; do not touch the full video)
./retina_env/bin/python -m pytest tests/ -q

# eval sampling (heuristic strata + track-boundary frames, fixed seed)
./retina_env/bin/python -m eval.sample_frames --help
```

## Results (short-clip validation, max_frames=60)

End-to-end job on the first 60 frames of `input_video.mp4`. Decision, frame
counts and verifier summary are recorded on the job row in SQLite:

```bash
sqlite3 jobs.sqlite3 'select job_id,status,decision,decision_reasons_json from jobs;'
```

## Limitations (short version)

Pipeline B is independent in weights and architecture, not in blind spots:
both stages train on similar face distributions, share input degradation, and
the verifier can only flag faces still visible in the output. Full analysis in
`docs/shared_failure_modes.md`.
