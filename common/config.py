"""Shared pipeline configuration, versioning, and idempotency helpers."""

import copy
import hashlib
import json

# Bump whenever detection, tracking, blur, verification, rendering, or
# decision behaviour changes. Same video + new pipeline version = new job.
PIPELINE_VERSION = "0.3.0"

DEFAULT_CONFIG = {
    "pipeline_version": PIPELINE_VERSION,
    "anonymizer": {
        "detector": "insightface-buffalo_l-scrfd",
        "det_threshold": 0.3,
        "det_size": [640, 640],
        "ctx_id": -1,  # CPU
        "max_num": 0,
        "box_expand": 0.35,
        "pixel_blocks": 9,
        "min_block_px": 8,
        "feather": True,
    },
    "tracker": {
        "iou_threshold": 0.3,
        "max_missed": 5,
        "min_score": 0.3,
        "pad_before": 8,
        "pad_after": 12,
        "bridge_gap_frames": 12,
        "bridge_min_iou": 0.0,
        "bridge_max_center_frac": 0.75,
        # Padded/interpolated boxes are grown by this fraction. Measured on the
        # tuning window: 0.1 -> 0.2 -> 0.3 lifts coverage of the residual boxes
        # at track hand-offs (0.909 -> 0.964 -> 1.0); 0.4 adds almost nothing
        # (0.833 vs 0.82). 0.3 is where the gains plateau.
        "interp_dilate": 0.3,
        # A bridged frame is a frame where Pipeline A never detected a face: the
        # blur box there is an inference. The verifier still checks it, but a
        # pass on an inferred region is weaker evidence than on an observed one.
        #   strict    - any bridged frame forces NEEDS_REVIEW (default)
        #   threshold - only bridges longer than bridge_gap_review_threshold do
        #   fraction  - only when bridged frames exceed the fraction threshold
        #   none      - never blocks; counts are still recorded in the job result
        "bridge_policy": "strict",
        "bridge_gap_review_threshold": 3,
        "bridge_frame_fraction_threshold": 0.02,
    },
    "verifier": {
        "model": "retinaface-pytorch",
        "det_threshold": 0.4,
        "stride": 1,
        "boundary_pad": 5,
        "coverage_required": 1.0,
        "cpu_threads": 1,
    },
    "decision": {
        "default": "NEEDS_REVIEW",
        "max_verifier_hits_for_safe": 0,
    },
    "render": {
        "video_codec": "libx264",
        "crf": 20,
        "preset": "veryfast",
        "pix_fmt": "yuv420p",
    },
    # `limit.max_frames` is part of the hashed config on purpose: processing the
    # first N frames produces a different artifact from processing the whole
    # file, so it must not share an idempotency key with a full run.
    "limit": {
        "max_frames": 0,
    },
}


def canonical_config(user_config=None):
    """Merge user config over defaults and return a deterministic copy."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in (user_config or {}).items():
        if section not in config or not isinstance(values, dict):
            config[section] = copy.deepcopy(values)
            continue
        config[section].update(copy.deepcopy(values))
    config["pipeline_version"] = PIPELINE_VERSION
    return config


def canonical_json(config):
    """Deterministic JSON encoding used for hashing and DB storage."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def content_idempotency_key(content_sha256, config=None):
    """Hash of (content sha256, pipeline version, canonical config)."""
    payload = "|".join(
        [content_sha256, PIPELINE_VERSION, canonical_json(canonical_config(config))]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
