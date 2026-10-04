# Anonymise Faces: two-stage blur + verify pipeline

Two independent stages, a job layer that never says "safe" unless both agree,
and an evaluation harness that is honest about shared failure modes.

## Fresh-clone reproduction

Weights are **not** in the repo. First run downloads them to OS-standard
cache dirs (total ~500MB); keep the machine online for that step:

| model | consumer | cache path after first run | source |
|---|---|---|---|
| `buffalo_l` (SCRFD, Pipeline A) | `pipeline/detect.py` via `insightface` | `~/.insightface/models/buffalo_l/` | auto-download by `insightface` on first `FaceAnalysis.prepare()` |
| `retinaface.pth` (Pipeline B verifier) | `verify/retinaface_check.py` via `retinaface` | `~/.deepface/weights/retinaface.pth` | `https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.pth` (backup: `https://huggingface.co/serengil/deepface/resolve/main/retinaface.pth`) |

RetinaFace needs the PyTorch backend: the repo pins `retinaface==0.0.19`
(PyPI package `retina-face`) in `requirements.txt` and every entry point
sets `DEEPFACE_BACKEND_ENGINE=pytorch` before import.

The **input video is not in the repo** either (`input_video.mp4` is
gitignored). Place your own file at the repo root, or point any `--video`
flag / notebook `INPUT_VIDEO` at an existing path (e.g. a file under
`downloads/`).

## Setup

```bash
python3.12 -m venv retina_env && ./retina_env/bin/pip install -r requirements.txt
export DEEPFACE_BACKEND_ENGINE=pytorch   # required before importing retinaface
export PYTHONPATH=$PWD                   # run from the repo root
```

## Run

```bash
# API + worker (worker runs in-process; stands in for SQS/Celery)
./retina_env/bin/python -m jobs.run_local --video input_video.mp4 --max-frames 60

# tests (fast; do not touch the full video)
./retina_env/bin/python -m pytest tests/ -q

# eval sampling (heuristic strata + track-boundary frames, fixed seed)
./retina_env/bin/python -m eval.sample_frames --video input_video.mp4 \
  --anonymised artifacts/<job-output>.mp4 --tracks-json <tracks-json> --out-dir eval/out

# notebook (10s demo clip; needs INPUT_VIDEO set to a local file)
./retina_env/bin/python -m jupyter nbconvert --to notebook --execute \
  notebooks/detect_anonymise_verify.ipynb --output /tmp/e2e_out.ipynb
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
