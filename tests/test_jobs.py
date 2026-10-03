"""Unit tests: idempotent concurrent submit + review-by-default gating."""

import json
import os
import threading

import pytest

from common import config as config_mod
from jobs import decision as decision_mod
from jobs.store import JobStore

ROOT = "/Users/abhisheklal/Workspace/anonymise_faces"


def _store(path="/tmp/test_jobs.sqlite3"):
    if os.path.exists(path):
        os.remove(path)
    return JobStore(path)


def test_concurrent_duplicate_submit_creates_one_row():
    store = _store()
    cfg = config_mod.canonical_config(None)
    digest = "abc123"
    key = config_mod.content_idempotency_key(digest, None)
    job_id = "job_" + key[:12]
    results = []

    def submit():
        row, dup = store.create_or_get(
            job_id, "upload-1", digest, config_mod.PIPELINE_VERSION,
            config_mod.canonical_json(cfg), key, ROOT + "/input_video.mp4")
        results.append((row["job_id"], dup))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count_jobs() == 1
    assert {r[0] for r in results} == {job_id}
    assert sum(1 for r in results if r[1]) == 7  # one creator, seven dups


def test_same_video_new_version_is_new_job():
    store = _store("/tmp/test_jobs2.sqlite3")
    cfg = config_mod.canonical_config(None)
    digest = "abc123"
    old_version = config_mod.PIPELINE_VERSION
    key1 = config_mod.content_idempotency_key(digest, None)
    try:
        config_mod.PIPELINE_VERSION = "9.9.9"
        key2 = config_mod.content_idempotency_key(digest, None)
    finally:
        config_mod.PIPELINE_VERSION = old_version
    assert key1 != key2
    row1, _ = store.create_or_get(
        "job_" + key1[:12], "u", digest, old_version,
        config_mod.canonical_json(cfg), key1, "in.mp4")
    row2, dup2 = store.create_or_get(
        "job_" + key2[:12], "u", digest, "9.9.9",
        config_mod.canonical_json(cfg), key2, "in.mp4")
    assert not dup2 and row1["job_id"] != row2["job_id"]
    assert store.count_jobs() == 2


def test_same_video_different_max_frames_is_new_job():
    """A partial run makes a different artifact, so it must not collide."""
    full = config_mod.content_idempotency_key("abc123", None)
    limited = config_mod.content_idempotency_key(
        "abc123", {"limit": {"max_frames": 60}})
    assert full != limited
    # defaults must canonicalise identically whether passed explicitly or not
    assert full == config_mod.content_idempotency_key(
        "abc123", {"limit": {"max_frames": 0}})


def _base_a(**over):
    base = {
        "input_info": {"declared_frames": 100, "decoded_frames": 100,
                       "decode_errors": 0, "width": 1920, "height": 1080},
        "rendered_frames": 100,
        "tracks": [],
        "bridges": [],
        "covered_frames": 100,
    }
    base.update(over)
    return base


def _bridge(gap=10, after=2, before=3, first=27, last=36):
    return {"after_track": after, "before_track": before, "gap_frames": gap,
            "first_bridged_frame": first, "last_bridged_frame": last}


def _with_policy(policy, **tracker):
    tracker.update({"bridge_policy": policy})
    return {"config": {"tracker": tracker}}


def _base_v(**over):
    base = {"coverage": 1.0, "num_flags": 0, "decode_errors": 0}
    base.update(over)
    return base


def test_clean_run_is_safe():
    status, reasons = decision_mod.decide(_base_a(), _base_v())
    assert status == "SAFE"


def test_verifier_hit_forces_review():
    status, _ = decision_mod.decide(_base_a(), _base_v(num_flags=1))
    assert status == "NEEDS_REVIEW"


def test_decode_error_forces_review():
    status, _ = decision_mod.decide(_base_a(), _base_v(decode_errors=1))
    assert status == "NEEDS_REVIEW"


def test_frame_mismatch_forces_review():
    status, _ = decision_mod.decide(_base_a(rendered_frames=99), _base_v())
    assert status == "NEEDS_REVIEW"


def test_incomplete_coverage_forces_review():
    status, _ = decision_mod.decide(_base_a(), _base_v(coverage=0.99))
    assert status == "NEEDS_REVIEW"


def test_pipeline_error_is_failed():
    status, _ = decision_mod.decide({"error": "boom"}, _base_v())
    assert status == "FAILED"


def test_pipeline_a_decode_error_forces_review():
    a = _base_a(input_info={"declared_frames": None, "decoded_frames": 100,
                            "decode_errors": 2, "width": 1920, "height": 1080})
    status, reasons = decision_mod.decide(a, _base_v())
    assert status == "NEEDS_REVIEW"
    assert any("pipeline_a_decode_errors" in r for r in reasons)


def test_bridge_summary_counts_are_always_available():
    a = _base_a(bridges=[_bridge(gap=10), _bridge(gap=4)],
                bridged_frames=14, covered_frames=100)
    summary = decision_mod.bridge_summary(a)
    assert summary["num_bridges"] == 2
    assert summary["bridged_frames"] == 14
    assert summary["max_bridge_gap"] == 10
    assert summary["bridged_frame_fraction"] == 0.14


def test_strict_policy_reviews_on_any_bridge():
    a = _base_a(bridges=[_bridge(gap=2)], bridged_frames=2)
    status, reasons = decision_mod.decide(a, _base_v())
    assert status == "NEEDS_REVIEW"
    assert any("bridged_gaps" in r for r in reasons)


def test_threshold_policy_allows_short_bridge_blocks_long():
    short = _base_a(bridges=[_bridge(gap=2)], bridged_frames=2,
                    **_with_policy("threshold", bridge_gap_review_threshold=3))
    assert decision_mod.decide(short, _base_v())[0] == "SAFE"

    long = _base_a(bridges=[_bridge(gap=4)], bridged_frames=4,
                   **_with_policy("threshold", bridge_gap_review_threshold=3))
    status, reasons = decision_mod.decide(long, _base_v())
    assert status == "NEEDS_REVIEW"
    assert any("bridged_gaps" in r for r in reasons)


def test_fraction_policy_allows_small_fraction_blocks_large():
    small = _base_a(bridges=[_bridge(gap=2)], bridged_frames=2,
                    covered_frames=1000,
                    **_with_policy("fraction",
                                   bridge_frame_fraction_threshold=0.02))
    assert decision_mod.decide(small, _base_v())[0] == "SAFE"

    large = _base_a(bridges=[_bridge(gap=10)], bridged_frames=100,
                    covered_frames=1000,
                    **_with_policy("fraction",
                                   bridge_frame_fraction_threshold=0.02))
    status, reasons = decision_mod.decide(large, _base_v())
    assert status == "NEEDS_REVIEW"
    assert any("bridged_gaps" in r for r in reasons)


def test_none_policy_never_blocks_on_bridges():
    a = _base_a(bridges=[_bridge(gap=30)], bridged_frames=30,
                **_with_policy("none"))
    status, reasons = decision_mod.decide(a, _base_v())
    assert status == "SAFE"
    assert not any("bridged_gaps" in r for r in reasons)


def test_coverage_required_config_is_honored():
    a = _base_a(config={"verifier": {"coverage_required": 0.9}})
    assert decision_mod.decide(a, _base_v(coverage=0.95))[0] == "SAFE"
    assert decision_mod.decide(a, _base_v(coverage=0.85))[0] == "NEEDS_REVIEW"
    strict = _base_a(config={"verifier": {"coverage_required": 1.0}})
    assert decision_mod.decide(strict, _base_v(coverage=0.999))[0] == \
        "NEEDS_REVIEW"
