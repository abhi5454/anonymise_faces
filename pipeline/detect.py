"""Pipeline A detector: InsightFace SCRFD (buffalo_l), tuned for recall.

This is a different detector family from the RetinaFace verifier, so the two
stages are independent in weights and architecture. They still share training
distribution blind spots (WIDER FACE-style data), which is documented in
docs/shared_failure_modes.md rather than assumed away.
"""

from insightface.app import FaceAnalysis


def build_anonymizer_model(det_size=(640, 640), ctx_id=-1):
    """Load the SCRFD detector wrapped in FaceAnalysis (detection only)."""
    app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
    # det_thresh here is only the internal default; per-call filtering uses
    # det_threshold explicitly so experiments stay reproducible.
    app.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=det_size)
    return app


def detect_faces_scrfd(frame_bgr, model, det_threshold=0.3, max_num=0):
    """Detect faces in one BGR frame.

    Returns a list of {"box": [x1, y1, x2, y2], "score": float} sorted by
    descending score. Boxes are clipped to the frame.
    """
    height, width = frame_bgr.shape[:2]
    faces = model.get(frame_bgr, max_num=max_num)
    out = []
    for face in faces or []:
        score = float(getattr(face, "det_score", 0.0))
        if score < det_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"box": [x1, y1, x2, y2], "score": score})
    out.sort(key=lambda item: item["score"], reverse=True)
    return out
