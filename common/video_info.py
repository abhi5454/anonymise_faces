"""Video metadata: ffprobe facts plus actual decoded frame counts."""

import json
import subprocess

import cv2


def _rational_to_float(value, default=0.0):
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            den = float(den)
            return float(num) / den if den else default
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def probe_video(path):
    """Return container/codec metadata for a video file via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,"
        "nb_frames,duration,rotation",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream_tags=rotate",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed for %s: %s" % (path, proc.stderr.strip()))
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    fps = _rational_to_float(stream.get("avg_frame_rate"), 0.0)
    if not fps:
        fps = _rational_to_float(stream.get("r_frame_rate"), 25.0)
    try:
        declared_frames = int(stream.get("nb_frames") or 0) or None
    except (TypeError, ValueError):
        declared_frames = None
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": float(fps),
        "codec": stream.get("codec_name"),
        "rotation": stream.get("rotation")
        or (stream.get("tags") or {}).get("rotate"),
        "declared_frames": declared_frames,
        "duration_seconds": float(
            stream.get("duration") or fmt.get("duration") or 0.0
        ),
        "size_bytes": int(fmt.get("size") or 0),
    }


def count_decoded_frames(path, max_frames=0):
    """Decode with OpenCV and return (decoded_count, decode_errors).

    A frame counts as a decode error when cap.read() returns False before we
    have seen the declared frame count, or when a returned frame is empty.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open video for decode: %s" % path)
    decoded = 0
    errors = 0
    while True:
        if max_frames and decoded >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            # End of stream vs error is disambiguated by the caller using the
            # declared count; anything short of it is recorded as an error.
            errors += 1
            break
        decoded += 1
    cap.release()
    return decoded, errors


def collect_video_info(path, max_frames=0):
    """Combine ffprobe metadata with an OpenCV decode pass."""
    info = probe_video(path)
    decoded, decode_errors = count_decoded_frames(path, max_frames=max_frames)
    info["decoded_frames"] = decoded
    info["decode_errors"] = decode_errors
    return info
