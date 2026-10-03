"""Local runner: submit a video to the job service and wait for the decision."""

import argparse
import os
import sys

os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "pytorch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config as config_mod  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from jobs.store import JobStore  # noqa: E402
from jobs.worker import Worker  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--db", default="jobs.sqlite3")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--crops-dir", default="crops")
    args = parser.parse_args()

    store = JobStore(args.db)
    worker = Worker(store, out_dir=args.out_dir, crops_dir=args.crops_dir,
                    max_frames=args.max_frames)
    worker.start()
    digest = sha256_file(args.video)
    # max_frames is hashed: a partial run is a different artifact and must not
    # collide with a full run of the same file.
    user_config = {"limit": {"max_frames": args.max_frames}}
    cfg = config_mod.canonical_config(user_config)
    key = config_mod.content_idempotency_key(digest, user_config)
    row, dup = store.create_or_get(
        "job_" + key[:12], os.path.basename(args.video), digest,
        config_mod.PIPELINE_VERSION, config_mod.canonical_json(cfg), key,
        os.path.abspath(args.video))
    print("job=%s duplicate=%s" % (row["job_id"], dup), flush=True)
    if not dup:
        worker.submit(row["job_id"])
    final = worker.wait_until_done(row["job_id"], poll=5.0)
    print("status=%s decision=%s" % (final["status"], final["decision"]))
    print("reasons=%s" % final["decision_reasons_json"])
    print("frames=%s" % final["frame_counts_json"])
    print("output=%s" % final["output_path"])


if __name__ == "__main__":
    main()
