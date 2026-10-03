"""Background worker: claim QUEUED jobs and run A -> render -> B -> decide."""

import json
import os
import queue
import threading
import time
import traceback

from common import config as config_mod
from common.hashing import write_json_atomic
from jobs import decision as decision_mod
from jobs.decision import FAILED, NEEDS_REVIEW, SAFE
from jobs.store import JobStore  # noqa: F401  (re-exported for callers)


class Worker:
    """Single-threaded queue worker (stand-in for SQS/Celery)."""

    def __init__(self, store, out_dir, crops_dir, max_frames=0,
                 supported_resolutions=((640, 480),)):
        self.store = store
        self.out_dir = out_dir
        self.crops_dir = crops_dir
        self.max_frames = max_frames
        self.supported_resolutions = supported_resolutions
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Start the background thread (idempotent)."""
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, job_id):
        """Enqueue a job id for processing."""
        self._queue.put(job_id)

    def _loop(self):
        while True:
            job_id = self._queue.get()
            try:
                self.process(job_id)
            except Exception as exc:  # noqa: BLE001 - worker must not die
                traceback.print_exc()
                try:
                    self.store.finish(
                        job_id, FAILED,
                        error="worker_crash: %s: %s"
                        % (type(exc).__name__, exc))
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._queue.task_done()

    def process(self, job_id):
        """Run one job synchronously. Returns True if this worker did it."""
        if not self.store.claim(job_id):
            return False  # another worker won the race
        row = self.store.get(job_id)
        try:
            from pipeline.run_pipeline_a import run_pipeline_a
            from verify.retinaface_check import run_verify

            user_config = json.loads(row["config_json"]) if row["config_json"] else {}
            run_config = config_mod.canonical_config(user_config)
            # The job's stored config decides behaviour: max_frames is hashed,
            # so a limited run can never share an idempotency key with a full
            # run of the same file.
            max_frames = int(run_config["limit"]["max_frames"]
                             or self.max_frames or 0)
            pipeline_a = run_pipeline_a(
                row["input_path"], self.out_dir, user_config,
                max_frames=max_frames)
            verification = run_verify(
                pipeline_a["output_path"], pipeline_a["tracks"], user_config,
                crops_dir=os.path.join(self.crops_dir, job_id),
                max_frames=max_frames)
            status, reasons = decision_mod.decide(
                pipeline_a, verification,
                supported_resolutions=self.supported_resolutions)
            if status not in (SAFE, NEEDS_REVIEW, FAILED):
                status, reasons = NEEDS_REVIEW, ["unknown_decision:" + status]

            # Side-car artifacts for the eval harness (repo paths, atomic
            # writes; the eval README's --tracks-json / --verifier-flags args
            # read exactly these). They are derived facts, not new behaviour:
            # the decision above is computed before they run and never depends
            # on them existing, so a write failure must not change it.
            try:
                tracks_path = write_json_atomic(
                    os.path.join(self.out_dir, job_id + ".tracks.json"),
                    {
                        "job_id": job_id,
                        "pipeline_version": pipeline_a["pipeline_version"],
                        "input_path": row["input_path"],
                        "tracks": pipeline_a["tracks"],
                        "bridges": pipeline_a["bridges"],
                    })
                flags_path = write_json_atomic(
                    os.path.join(self.out_dir, job_id + ".verify_flags.json"),
                    {
                        "job_id": job_id,
                        "output_path": pipeline_a["output_path"],
                        "threshold": verification.get("threshold"),
                        "stride": verification.get("stride"),
                        "coverage": verification.get("coverage"),
                        "flags": verification.get("flags", []),
                    })
                sidecar_error = None
            except Exception as exc:  # noqa: BLE001 - advisory files only
                traceback.print_exc()
                tracks_path = flags_path = None
                sidecar_error = "%s: %s" % (type(exc).__name__, exc)

            frame_counts = {
                "declared": pipeline_a["input_info"].get("declared_frames"),
                "decoded": pipeline_a["input_info"].get("decoded_frames"),
                "rendered": pipeline_a.get("rendered_frames"),
                "pipeline_a_decode_errors":
                    pipeline_a["input_info"].get("decode_errors", 0),
                "verify_decode_errors": verification.get("decode_errors", 0),
                "tracks_json": tracks_path,
                "verify_flags_json": flags_path,
            }
            if sidecar_error:
                frame_counts["sidecar_write_error"] = sidecar_error
            # Bridge accounting is always stored, whether or not the policy in
            # force lets it block: "how much of this output rests on inferred
            # boxes" is part of the honest record for every job.
            frame_counts.update(
                {"bridges_" + k: v
                 for k, v in decision_mod.bridge_summary(pipeline_a).items()})
            frame_counts["bridge_policy"] = (
                (user_config or {}).get("tracker", {}).get(
                    "bridge_policy", config_mod.DEFAULT_CONFIG["tracker"]
                    ["bridge_policy"]) or "strict")
            verify_summary = {
                "planned_frames": verification.get("planned_frames"),
                "covered_frames": verification.get("covered_frames"),
                "coverage": verification.get("coverage"),
                "num_flags": verification.get("num_flags"),
                "flags": verification.get("flags", [])[:50],
                "threshold": verification.get("threshold"),
                "stride": verification.get("stride"),
                "verify_flags_json": flags_path,
            }
            self.store.finish(
                job_id, status,
                output_path=pipeline_a["output_path"],
                output_sha256=pipeline_a["output_sha256"],
                decision=status,
                decision_reasons_json=json.dumps(reasons),
                frame_counts_json=json.dumps(frame_counts),
                verify_json=json.dumps(verify_summary),
                attempt_count=(row.get("attempt_count") or 0) + 1,
            )
        except Exception as exc:  # noqa: BLE001 - failures become FAILED jobs
            traceback.print_exc()
            self.store.finish(
                job_id, FAILED,
                error="%s: %s" % (type(exc).__name__, exc),
                attempt_count=(row.get("attempt_count") or 0) + 1,
            )
        return True

    def wait_until_done(self, job_id, timeout=3600.0, poll=1.0):
        """Block until a job reaches a terminal state (for tests/scripts)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            row = self.store.get(job_id)
            if row and row["status"] in (SAFE, NEEDS_REVIEW, FAILED):
                return row
            time.sleep(poll)
        raise TimeoutError("job %s did not finish in %.0fs" % (job_id, timeout))
