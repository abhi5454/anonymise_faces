# Idempotency demo (real service, persisted DB)

Run on 2026-10-03 against the **real** `jobs.sqlite3` in the repo root (not a
temp DB), using the production entry point `uvicorn jobs.api:app` with the
in-process worker enabled. Nothing here is simulated: the transcript below is
the raw terminal output.

## Setup

```bash
cd /Users/abhisheklal/Workspace/anonymise_faces
export PYTHONPATH=/Users/abhisheklal/Workspace/anonymise_faces
export DEEPFACE_BACKEND_ENGINE=pytorch
export ANON_WORKER=1                    # start the worker inside the API

# a fresh input file (20 frames re-encoded from input_video.mp4)
ffmpeg -y -v error -i input_video.mp4 -frames:v 20 -c:v libx264 -crf 20 /tmp/idem_demo.mp4
# sha256 bd9d1ddf5fd5baeeef87404828875517b10dc62bbccaa99d023e226fae8f859f
sqlite3 jobs.sqlite3 'select count(*) from jobs;'   # 2 rows before
ls artifacts/*.mp4 | wc -l                          # 2 outputs before

./retina_env/bin/python -m uvicorn jobs.api:app --port 8099 &
```

## The same POST body, twice

```console
$ curl -s http://127.0.0.1:8099/health
{"status":"ok","pipeline_version":"0.3.0"}

$ curl -s -X POST http://127.0.0.1:8099/jobs -H 'content-type: application/json' \
    -d '{"upload_id":"idem-demo","video_path":"/tmp/idem_demo.mp4","config":{"limit":{"max_frames":20}}}'
{"job_id":"job_95a9c9048d58","status":"QUEUED",
 "idempotency_key":"95a9c9048d58f5434871f0d982f231f25500576d856592bf276164ab7d5cd8e7",
 "duplicate":false}

$ curl -s -X POST http://127.0.0.1:8099/jobs -H 'content-type: application/json' \
    -d '{"upload_id":"idem-demo","video_path":"/tmp/idem_demo.mp4","config":{"limit":{"max_frames":20}}}'
{"job_id":"job_95a9c9048d58","status":"RUNNING",
 "idempotency_key":"95a9c9048d58f5434871f0d982f231f25500576d856592bf276164ab7d5cd8e7",
 "duplicate":true}
```

Identical `job_id`, identical `idempotency_key`; the second POST reports
`duplicate:true` and does **not** enqueue a second run (note it saw the job
already `RUNNING`, i.e. the single run had been claimed while it was in flight).

## Result: one row, one run, one output

```console
$ sqlite3 jobs.sqlite3 'select count(*) from jobs;'
3                      # 2 before -> 3 after: exactly +1 for two POSTs

$ sqlite3 jobs.sqlite3 'select job_id,status,decision,attempt_count from jobs;'
job_66cbb7fede3b | NEEDS_REVIEW | NEEDS_REVIEW | 1
job_2b5b118ad6ab | NEEDS_REVIEW | NEEDS_REVIEW | 1
job_95a9c9048d58 | SAFE         | SAFE         | 1     # attempt_count 1 = processed once

$ ls artifacts/*.mp4 | wc -l
3                      # 2 before -> 3 after: exactly one new output
```

## The finished job (GET /jobs/job_95a9c9048d58)

```json
{"job_id":"job_95a9c9048d58","status":"SAFE","upload_id":"idem-demo",
 "content_sha256":"bd9d1ddf5fd5baeeef87404828875517b10dc62bbccaa99d023e226fae8f859f",
 "pipeline_version":"0.3.0","decision":"SAFE",
 "decision_reasons":["completed","frame_counts_match","verify_coverage=1.0","verifier_hits=0"],
 "artifacts":{"input_path":"/tmp/idem_demo.mp4",
   "output_path":".../artifacts/anon-11f8e5230fe83310.mp4",
   "output_sha256":"11f8e5230fe83310500f25c1a652b1ddf9f2dd8cf2bc8451ebaaadf00e7f71aa"},
 "frame_counts":{"declared":20,"decoded":20,"rendered":20,
   "pipeline_a_decode_errors":0,"verify_decode_errors":0,
   "tracks_json":".../artifacts/job_95a9c9048d58.tracks.json",
   "verify_flags_json":".../artifacts/job_95a9c9048d58.verify_flags.json",
   "bridges_num_bridges":0,"bridges_bridged_frames":0,"bridges_covered_frames":32,
   "bridges_max_bridge_gap":0,"bridges_bridged_frame_fraction":0.0,
   "bridge_policy":"strict"},
 "verify":{"planned_frames":20,"covered_frames":20,"coverage":1.0,"num_flags":0,
   "flags":[],"threshold":0.4,"stride":1,
   "verify_flags_json":".../artifacts/job_95a9c9048d58.verify_flags.json"},
 "error":null}
```

## What this shows

1. **Two identical submissions -> one logical result.** One row (`+1`), one
   execution (`attempt_count=1`), one output file (`+1`).
2. **The gate can pass.** This 20-frame clip reached `SAFE` for the stated
   reasons only: completed, frame counts match, verify coverage `1.0`
   (stride `1`, threshold `0.4`, every planned frame covered), `verifier_hits=0`,
   no bridges, no decode errors. Every other row in the table is
   `NEEDS_REVIEW`, which is the default outcome.
3. **Key derivation:** `sha256(content_sha256 ‖ PIPELINE_VERSION ‖ canonical config)`,
   enforced by a `UNIQUE` constraint, with `QUEUED -> RUNNING` claimed by
   compare-and-set, so two workers racing on the same key cannot double-run.
   `limit.max_frames` is part of the hashed config, so a 20-frame run and a
   full-file run of the same video are different jobs.
