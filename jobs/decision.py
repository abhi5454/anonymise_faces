"""Review-by-default decision logic. SAFE must be earned, never assumed."""

SAFE = "SAFE"
NEEDS_REVIEW = "NEEDS_REVIEW"
FAILED = "FAILED"

# Bridge policies (see common/config.py DEFAULT_CONFIG["tracker"]).
BRIDGE_STRICT = "strict"
BRIDGE_THRESHOLD = "threshold"
BRIDGE_FRACTION = "fraction"
BRIDGE_NONE = "none"


def bridge_summary(pipeline_a):
    """Counts describing every bridged (inferred) frame. Always recorded.

    A bridged frame is a frame where Pipeline A never detected a face, so the
    blur box there is interpolated from neighbouring tracks. These numbers go
    into the job result regardless of policy, because "how much of this output
    rests on inference" is part of the honest record.
    """
    bridges = pipeline_a.get("bridges") or []
    bridged_frames = pipeline_a.get(
        "bridged_frames", sum(b.get("gap_frames", 0) for b in bridges))
    covered_frames = pipeline_a.get("covered_frames") or 0
    return {
        "num_bridges": len(bridges),
        "bridged_frames": int(bridged_frames),
        "covered_frames": int(covered_frames),
        "max_bridge_gap": max((b.get("gap_frames", 0) for b in bridges),
                              default=0),
        "bridged_frame_fraction": (
            round(bridged_frames / float(covered_frames), 4)
            if covered_frames else 0.0),
    }


def _bridge_reason(pipeline_a, policy):
    """Return the review reason for bridged frames, or None if policy allows."""
    summary = bridge_summary(pipeline_a)
    if summary["bridged_frames"] <= 0:
        return None
    tracker_cfg = ((pipeline_a.get("config") or {}).get("tracker") or {})
    policy = policy or tracker_cfg.get("bridge_policy") or BRIDGE_STRICT
    text = ("bridged_gaps=%d (frames=%d fraction=%.4f max_gap=%d)"
            % (summary["num_bridges"], summary["bridged_frames"],
               summary["bridged_frame_fraction"], summary["max_bridge_gap"]))
    if policy == BRIDGE_NONE:
        return None
    if policy == BRIDGE_THRESHOLD:
        limit = tracker_cfg.get("bridge_gap_review_threshold", 3)
        return text if summary["max_bridge_gap"] > limit else None
    if policy == BRIDGE_FRACTION:
        limit = tracker_cfg.get("bridge_frame_fraction_threshold", 0.02)
        return text if summary["bridged_frame_fraction"] > limit else None
    return text  # BRIDGE_STRICT and any unknown policy: block


def decide(pipeline_a, verification, supported_resolutions=((640, 480),)):
    """Return (decision, reasons). `supported_resolutions` lists (w, h) minima.

    supported_resolutions is a tuple of (min_width, min_height) pairs the
    pipeline has been tested against; anything smaller forces review.
    `reasons` holds blocking facts only; non-blocking bridge accounting is
    available via bridge_summary() and is stored on the job row.
    """
    reasons = []
    if pipeline_a.get("error"):
        return FAILED, ["pipeline_a_failed: %s" % pipeline_a["error"]]
    if verification.get("error"):
        return FAILED, ["verify_failed: %s" % verification["error"]]

    info = pipeline_a.get("input_info", {}) or {}
    declared = info.get("declared_frames")
    decoded = info.get("decoded_frames")
    rendered = pipeline_a.get("rendered_frames")
    if info.get("decode_errors"):
        reasons.append("pipeline_a_decode_errors=%d" % info["decode_errors"])
    if declared and decoded != declared:
        reasons.append("frame_count_mismatch declared=%s decoded=%s"
                       % (declared, decoded))
    if decoded != rendered:
        reasons.append("render_count_mismatch decoded=%s rendered=%s"
                       % (decoded, rendered))
    if verification.get("decode_errors"):
        reasons.append("verify_decode_errors=%d" % verification["decode_errors"])
    coverage_required = _coverage_required(pipeline_a)
    coverage = verification.get("coverage", 0.0)
    if coverage < coverage_required:
        reasons.append("incomplete_verify_coverage=%.3f (required=%.3f)"
                       % (coverage, coverage_required))
    if verification.get("num_flags", 0) > 0:
        reasons.append("verifier_hits=%d" % verification["num_flags"])

    width, height = info.get("width", 0), info.get("height", 0)
    min_w = min(w for w, _ in supported_resolutions)
    min_h = min(h for _, h in supported_resolutions)
    tested = ", ".join("%dx%d" % (w, h) for w, h in supported_resolutions)
    if width < min_w or height < min_h:
        reasons.append("resolution_outside_tested_range=%dx%d tested=[%s]"
                       % (width, height, tested))

    tracks = pipeline_a.get("tracks", []) or []
    gappy = [t for t in tracks if t.get("gap_frames", 0) > 3]
    if gappy:
        reasons.append("gappy_tracks=%d (max_gap_track=%d)" % (
            len(gappy), max(t["track_id"] for t in gappy)))

    bridge_reason = _bridge_reason(pipeline_a, None)
    if bridge_reason:
        reasons.append(bridge_reason)

    if reasons:
        return NEEDS_REVIEW, reasons
    return SAFE, ["completed", "frame_counts_match",
                  "verify_coverage=1.0", "verifier_hits=0"]


def _coverage_required(pipeline_a):
    """Coverage the verifier must reach; from config, defaulting to 100%."""
    cfg = (pipeline_a.get("config") or {}).get("verifier") or {}
    try:
        value = float(cfg.get("coverage_required", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, value))
