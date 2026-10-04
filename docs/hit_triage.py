"""Hit triage (machine-derived, NOT ground truth). Re-runs Pipeline A
detection+tracking IN MEMORY on frozen DEFAULT_CONFIG (0.3.0, no overrides)
and overlaps each persisted verifier flag vs expanded blur boxes.
inside: IoU>=0.5 or coverage>=0.9; partial: IoU>0.02 or cov>0.02;
outside: else. Reads videos+flags JSON only; writes /tmp/*.json only.
"""
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "pytorch")

from common import config as C  # noqa: E402
from pipeline.blur import expand_box  # noqa: E402
from pipeline.detect import build_anonymizer_model, detect_faces_scrfd  # noqa
from pipeline.track import track_detections  # noqa: E402

JOBS = {
    "job_e22858b215cf": ("notebooks/out/clip_10s.mp4", 60,
                         "notebooks/out/job/job_e22858b215cf.verify_flags.json"),
    "job_a69e8d7a950f": ("eval/clips/eval_seg.mp4", 400,
                         "artifacts/job_a69e8d7a950f.verify_flags.json"),
    "job_2c17f0a52a56": ("eval/clips/eval_seg2.mp4", 450,
                         "artifacts/job_2c17f0a52a56.verify_flags.json"),
    "clip300": ("notebooks/out/clip_10s.mp4", 300, None),
}
W, H = 1920, 1080


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def classify(fb, boxes):
    if not boxes:
        return ("outside", 0.0, 0.0)
    fa = max(0.0, fb[2] - fb[0]) * max(0.0, fb[3] - fb[1])
    best, cov = 0.0, 0.0
    for b in boxes:
        best = max(best, iou(fb, b))
        ix = max(0.0, min(fb[2], b[2]) - max(fb[0], b[0]))
        iy = max(0.0, min(fb[3], b[3]) - max(fb[1], b[1]))
        if fa > 0:
            cov = max(cov, ix * iy / fa)
    if best >= 0.5 or cov >= 0.9:
        return ("inside", best, cov)
    if best > 0.02 or cov > 0.02:
        return ("partial", best, cov)
    return ("outside", best, cov)


def run_job(name, model, a, t):
    rel, maxf, fpath = JOBS[name]
    flags = json.load(open(os.path.join(ROOT, fpath))).get("flags", []) if fpath else []
    cap = cv2.VideoCapture(os.path.join(ROOT, rel))
    dets, sizes, scores, fi = {}, [], [], 0
    while fi < maxf:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        dd = detect_faces_scrfd(fr, model, det_threshold=a["det_threshold"])
        for d in dd:
            x1, y1, x2, y2 = d["box"]
            sizes.append([round(x2 - x1, 1), round(y2 - y1, 1)])
            scores.append(round(d["score"], 3))
        dets[fi] = [{"box": d["box"], "score": d["score"]} for d in dd]
        fi += 1
    cap.release()
    active, tracks, bridges = track_detections(
        dets, iou_threshold=t["iou_threshold"], max_missed=t["max_missed"],
        min_score=t["min_score"], pad_before=t["pad_before"],
        pad_after=t["pad_after"], bridge_gap_frames=t["bridge_gap_frames"],
        bridge_min_iou=t["bridge_min_iou"],
        bridge_max_center_frac=t["bridge_max_center_frac"],
        interp_dilate=t["interp_dilate"])
    triage = []
    for f in flags:
        fb = [float(v) for v in f["box"]]
        boxes = [expand_box(e["box"], W, H, a["box_expand"])
                 for e in active.get(f["frame_index"], [])]
        cls, best, cov = classify(fb, boxes)
        s = f["score"]
        sb = "0.40-0.55" if s < 0.55 else ("0.55-0.70" if s < 0.70 else ">=0.70")
        h = fb[3] - fb[1]
        hb = "<40" if h < 40 else ("40-80" if h <= 80 else ">80")
        triage.append({"frame": f["frame_index"], "score": round(s, 3),
                       "box": [round(v, 1) for v in fb], "cls": cls,
                       "sband": sb, "hband": hb, "h": round(h, 1),
                       "best_iou": round(best, 3), "coverage": round(cov, 3),
                       "n_blur_boxes": len(boxes)})
    tag = {"job_e22858b215cf": "e228", "job_a69e8d7a950f": "a69",
           "job_2c17f0a52a56": "2c17", "clip300": "clip300"}[name]
    json.dump({"job": name, "frames_read": fi, "sizes": sizes,
               "scores": scores, "tracks": tracks, "bridges": bridges,
               "triage": triage}, open("/tmp/triage_%s.json" % tag, "w"))
    print("%s done frames=%d dets=%d nflags=%d ntracks=%d" %
          (name, fi, len(sizes), len(triage), len(tracks)), flush=True)


if __name__ == "__main__":
    cfg = C.canonical_config({})
    a, t = cfg["anonymizer"], cfg["tracker"]
    print("PIPELINE_VERSION=%s det_thr=%s det_size=%s" %
          (C.PIPELINE_VERSION, a["det_threshold"], a["det_size"]), flush=True)
    model = build_anonymizer_model(det_size=tuple(a["det_size"]), ctx_id=-1)
    for n in ["job_e22858b215cf", "job_a69e8d7a950f", "job_2c17f0a52a56",
              "clip300"]:
        run_job(n, model, a, t)
    print("ALL DONE", flush=True)

