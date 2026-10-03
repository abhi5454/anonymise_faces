#!/usr/bin/env python3
"""Detect faces in a video with RetinaFace (PyTorch backend).

Runs RetinaFace on every Nth frame (--stride), reuses one built model for all
frames, and writes a JSON file with one entry per sampled frame:

    {"frame_100": {"timestamp_seconds": 4.17,
                   "faces": {"face_1": {"score": 0.97,
                                        "facial_area": [x1, y1, x2, y2],
                                        "landmarks": {...}}}}}

Options: --preview N saves annotated PNGs, --anonymise OUT.mp4 writes a copy
of the video with detected faces blurred.

IMPORTANT: pass OpenCV frames to RetinaFace as-is (BGR). Do NOT convert with
cv2.cvtColor(..., COLOR_BGR2RGB) first - that degrades detection.

The PyTorch backend is forced via DEEPFACE_BACKEND_ENGINE because TensorFlow
is also installed in retina_env and would otherwise be picked as the default
backend (and it additionally requires the tf-keras package).
"""

import os

os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "pytorch")

import argparse
import json
import time

import cv2

from retinaface import RetinaFace


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect faces in a video with RetinaFace."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--stride", type=int, default=5, help="Detect on every Nth frame."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Confidence in (0, 1]."
    )
    parser.add_argument(
        "--out-json",
        default="retinaface_results.json",
        help="Where to write the detections JSON.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after reading this many frames (0 = whole video).",
    )
    parser.add_argument(
        "--preview", type=int, default=0, help="Save first N face frames as PNGs."
    )
    parser.add_argument(
        "--preview-dir", default="previews", help="Directory for preview PNGs."
    )
    parser.add_argument(
        "--anonymise",
        default=None,
        metavar="OUT_MP4",
        help="Also write a copy of the video with faces blurred.",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=51,
        help="Odd Gaussian blur kernel size (default: 51).",
    )
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be >= 1.")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1].")
    if not os.path.isfile(args.video):
        parser.error("video not found: %s" % args.video)
    return args


def faces_to_jsonable(detections):
    """Convert a RetinaFace result dict into JSON-serialisable plain types."""
    out = {}
    if isinstance(detections, dict):
        for face_key, face in detections.items():
            landmarks = face.get("landmarks", {}) or {}
            out[face_key] = {
                "score": float(face["score"]),
                "facial_area": [int(v) for v in face["facial_area"]],
                "landmarks": {
                    name: [float(c) for c in coords]
                    for name, coords in landmarks.items()
                },
            }
    return out


def clamp_box(box, width, height):
    """Clamp [x1, y1, x2, y2] to frame bounds, return ints."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def draw_detections(frame, faces):
    """Return a copy of frame with boxes, scores and landmarks drawn."""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for face_key, face in (faces or {}).items():
        x1, y1, x2, y2 = clamp_box(face["facial_area"], w, h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "%s %.2f" % (face_key, float(face["score"])),
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        for lx, ly in (face.get("landmarks") or {}).values():
            cv2.circle(annotated, (int(lx), int(ly)), 3, (0, 0, 255), -1)
    return annotated


def blur_detections(frame, faces, blur_size=51):
    """Return a copy of frame with every detected face region blurred."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for face in (faces or {}).values():
        x1, y1, x2, y2 = clamp_box(face["facial_area"], w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 3 or bh < 3:
            continue
        k = min(blur_size, bw, bh)
        if k % 2 == 0:
            k -= 1
        if k < 3:
            continue
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def main(argv=None):
    """Run face detection over a video. Returns process exit code."""
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video: %s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None  # cv2 writer, only opened when --anonymise is given
    if args.anonymise:
        writer = cv2.VideoWriter(
            args.anonymise, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise SystemExit("Could not open output video: %s" % args.anonymise)



    if args.preview:
        os.makedirs(args.preview_dir, exist_ok=True)

    total_str = str(total) if total else "?"
    print(
        "video: %s (%dx%d @ %.2f fps, frames=%s)"
        % (args.video, width, height, fps, total_str),
        flush=True,
    )
    print(
        "backend=pytorch stride=%d threshold=%.2f" % (args.stride, args.threshold),
        flush=True,
    )
    print("Building RetinaFace model ...", flush=True)
    build_start = time.time()
    model = RetinaFace.build_model()
    print("Model ready in %.1fs." % (time.time() - build_start), flush=True)

    results = {}
    last_faces = {}
    saved_previews = 0
    sampled = 0
    infer_seconds = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        if frame_idx % args.stride == 0:
            sampled += 1
            timestamp = frame_idx / fps
            infer_start = time.time()
            # frame is BGR straight from OpenCV - pass it as-is (no BGR2RGB).
            faces = RetinaFace.detect_faces(
                frame, threshold=args.threshold, model=model
            )
            infer_seconds += time.time() - infer_start
            last_faces = faces if isinstance(faces, dict) else {}
            results["frame_%d" % frame_idx] = {
                "timestamp_seconds": round(timestamp, 2),
                "faces": faces_to_jsonable(last_faces),
            }
            avg = infer_seconds / sampled
            suffix = "/%d" % total if total else ""
            print(
                "frame %d%s t=%7.2fs faces=%d (%.2fs/frame)"
                % (frame_idx, suffix, timestamp, len(last_faces), avg),
                flush=True,
            )
            if args.preview and saved_previews < args.preview and last_faces:
                preview_path = os.path.join(
                    args.preview_dir, "frame_%06d.png" % frame_idx
                )
                cv2.imwrite(preview_path, draw_detections(frame, last_faces))
                saved_previews += 1
                print("  preview -> %s" % preview_path, flush=True)

        if writer is not None:
            if last_faces:
                writer.write(blur_detections(frame, last_faces, args.blur_size))
            else:
                writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print("Anonymised video -> %s" % args.anonymise, flush=True)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(
        "Sampled %d frames, wrote %d entries -> %s"
        % (sampled, len(results), args.out_json),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

