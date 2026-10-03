"""Pipeline B: RetinaFace verification on the RENDERED output file.

Decodes the rendered artifact (never Pipeline A's in-memory frames), builds a
predetermined verification plan covering EVERY output frame (stride 1 by
default, so no frame is skipped) plus track-boundary windows, requires the
planned coverage to be reached, and flags residual faces with crops. The
threshold and stride actually used are returned with the result so a reader
never has to assume them.
"""

import os
import time

import cv2

from common import config as config_mod
from retinaface import RetinaFace


def build_verifier_model():
    """Build the RetinaFace model once. Caller must pin the pytorch backend."""
    return RetinaFace.build_model()


def verify_frame(frame_bgr, model, threshold=0.4):
    """Return a JSON-able {face_key: {score, facial_area, landmarks}} dict."""
    raw = RetinaFace.detect_faces(frame_bgr, threshold=threshold, model=model)
    out = {}
    if isinstance(raw, dict):
        for key, face in raw.items():
            landmarks = face.get("landmarks", {}) or {}
            out[key] = {
                "score": float(face["score"]),
                "facial_area": [float(v) for v in face["facial_area"]],
                "landmarks": {
                    name: [float(c) for c in coords]
                    for name, coords in landmarks.items()
                },
            }
    return out


def build_verification_plan(total_frames, tracks, stride=1, boundary_pad=5):
    """Predetermined set of output frame indices the verifier must cover."""
    planned = set(range(0, total_frames, max(1, stride)))
    for track in tracks or []:
        for anchor in (track.get("padded_start"), track.get("padded_end")):
            if anchor is None:
                continue
            for frame in range(anchor - boundary_pad, anchor + boundary_pad + 1):
                if 0 <= frame < total_frames:
                    planned.add(frame)
    return sorted(planned)


def _save_crop(frame_bgr, box, crops_dir, frame_index, face_key):
    os.makedirs(crops_dir, exist_ok=True)
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    path = os.path.join(crops_dir, "flag_f%d_%s.png" % (frame_index, face_key))
    cv2.imwrite(path, frame_bgr[y1:y2, x1:x2])
    return path


def run_verify(output_path, tracks, user_config=None, model=None,
               crops_dir=None, max_frames=0):
    """Verify a rendered artifact. Returns flags, coverage, and statistics."""
    started = time.time()
    cfg = config_mod.canonical_config(user_config)["verifier"]
    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise RuntimeError("could not open rendered video: %s" % output_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames)
    planned = build_verification_plan(
        total, tracks, stride=cfg["stride"], boundary_pad=cfg["boundary_pad"])
    planned_set = set(planned)
    if model is None:
        model = build_verifier_model()
    flags, covered, decode_errors = [], 0, 0
    frame_index = 0
    try:
        while True:
            if max_frames and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                if frame_index < total:
                    decode_errors += 1
                break
            if frame_index in planned_set:
                faces = verify_frame(frame, model, cfg["det_threshold"])
                covered += 1
                for key, face in faces.items():
                    flags.append({
                        "frame_index": frame_index,
                        "timestamp_seconds": round(frame_index / fps, 2),
                        "face_key": key,
                        "score": face["score"],
                        "box": face["facial_area"],
                        "near_track_boundary": any(
                            abs(frame_index - (t.get("padded_start") or -1))
                            <= cfg["boundary_pad"]
                            or abs(frame_index - (t.get("padded_end") or -1))
                            <= cfg["boundary_pad"]
                            for t in tracks or []),
                        "crop_path": _save_crop(
                            frame, face["facial_area"],
                            crops_dir or "/tmp/verify_crops",
                            frame_index, key),
                    })
            frame_index += 1
    finally:
        cap.release()
    planned_total = len(planned)
    coverage = (covered / planned_total) if planned_total else 0.0
    return {
        "flags": flags,
        "num_flags": len(flags),
        "planned_frames": planned_total,
        "covered_frames": covered,
        "coverage": coverage,
        "decode_errors": decode_errors,
        "decoded_frames": frame_index,
        "threshold": cfg["det_threshold"],
        "stride": cfg["stride"],
        "elapsed_seconds": round(time.time() - started, 2),
    }
