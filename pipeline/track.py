"""Greedy IoU tracker with gap interpolation, cross-track bridging and padding.

Bridges short detection gaps and pads every track a few frames before/after, so
a face seen in frames 10 and 14 is still blurred in 11-13.

The coverage diagnostic on the rendered artifact showed the dominant real miss
was not a weak blur but a *detection dropout*: SCRFD lost a small moving face
for ~10 frames, the short track ended, and the gap was left unblurred. So two
extra mechanisms exist here, both reported rather than hidden:

- ``bridge_gap_frames``: if two consecutive tracks are spatially compatible and
  separated by at most this many frames, the gap is filled with interpolated
  boxes (an inference, not an observation, so it is flagged);
- ``interp_dilate``: interpolated/padded boxes are grown by this fraction to
  absorb motion the two anchor detections do not describe.

It CANNOT invent faces before the first detection of the first track; that
limitation is documented rather than hidden. Dependency-free so
ByteTrack/Kalman can replace it later.
"""

TRACK_FIELDS = ("track_id", "frame_index", "box", "score", "interpolated")


def _dilate_box(box, frac):
    """Grow a box by `frac` of its size about its centre (identity if frac<=0)."""
    if frac <= 0:
        return list(box)
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(1.0, x2 - x1) * (1.0 + frac) / 2.0
    hh = max(1.0, y2 - y1) * (1.0 + frac) / 2.0
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _diag(box):
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def _compatible(box_a, box_b, min_iou=0.0, max_center_frac=0.75):
    """Are two boxes close enough to plausibly be the same face nearby in time?"""
    if _iou(box_a, box_b) >= max(min_iou, 1e-9):
        return True
    ca = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    cb = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    scale = max(1.0, (_diag(box_a) + _diag(box_b)) / 2.0)
    return dist <= max_center_frac * scale


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _lerp_box(box_a, box_b, alpha):
    return [a + (b - a) * alpha for a, b in zip(box_a, box_b)]


def track_detections(detections_by_frame, iou_threshold=0.3, max_missed=5,
                     min_score=0.0, pad_before=3, pad_after=5,
                     bridge_gap_frames=12, bridge_min_iou=0.0,
                     bridge_max_center_frac=0.75, interp_dilate=0.1):
    """Link per-frame detections into padded, gap-bridged tracks.

    Returns (active_boxes, tracks, bridges):
    - active_boxes maps every covered frame to track entries (each with an
      `interpolated` flag and an `inferred` flag for bridged/padded frames);
    - tracks holds start/end/padded bounds plus gap stats used by the verifier;
    - bridges lists the cross-track inferences, because a bridged frame is a
      guess and review-by-default needs to know about it.
    """
    open_tracks, finished, next_id = {}, [], 0
    for frame_index in sorted(detections_by_frame):
        dets = sorted(
            (d for d in detections_by_frame.get(frame_index, [])
             if d.get("score", 0.0) >= min_score),
            key=lambda d: d.get("score", 0.0), reverse=True)
        claimed, matches = set(), {}
        for det_idx, det in enumerate(dets):
            best_id, best_iou = None, iou_threshold
            for track_id, state in open_tracks.items():
                if track_id in claimed:
                    continue
                iou = _iou(det["box"], state["box"])
                if iou >= best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is not None:
                matches[det_idx] = best_id
                claimed.add(best_id)
        matched = set(matches.values())
        for track_id in list(open_tracks):
            if track_id not in matched:
                open_tracks[track_id]["missed"] += 1
                if open_tracks[track_id]["missed"] > max_missed:
                    finished.append(open_tracks.pop(track_id))
        for det_idx, det in enumerate(dets):
            if det_idx in matches:
                state = open_tracks[matches[det_idx]]
                state["box"] = list(det["box"])
                state["score"] = float(det.get("score", 0.0))
                state["missed"] = 0
                state["observations"].append(
                    (frame_index, list(det["box"]), float(det.get("score", 0.0))))
            else:
                open_tracks[next_id] = {
                    "track_id": next_id, "box": list(det["box"]),
                    "score": float(det.get("score", 0.0)), "missed": 0,
                    "observations": [(frame_index, list(det["box"]),
                                      float(det.get("score", 0.0)))]}
                next_id += 1
    finished.extend(open_tracks.values())

    tracks, active_boxes = [], {}
    for state in finished:
        obs = sorted(state["observations"])
        if not obs:
            continue
        start, end = obs[0][0], obs[-1][0]
        padded_start = max(0, start - pad_before)
        padded_end = end + pad_after
        by_frame = {f: b for f, b, _ in obs}
        cur = start
        while cur <= end:
            if cur in by_frame:
                box, interp = by_frame[cur], False
            else:
                prev = max(f for f in by_frame if f < cur)
                nxt = min(f for f in by_frame if f > cur)
                box = _lerp_box(by_frame[prev], by_frame[nxt],
                                (cur - prev) / float(nxt - prev))
                interp = True
            active_boxes.setdefault(cur, []).append({
                "track_id": state["track_id"], "frame_index": cur,
                "box": box, "score": state["score"], "interpolated": interp,
                "inferred": False})
            cur += 1
        pad_box = _dilate_box(by_frame[start], interp_dilate)
        for pad in range(padded_start, start):
            active_boxes.setdefault(pad, []).append({
                "track_id": state["track_id"], "frame_index": pad,
                "box": list(pad_box), "score": state["score"],
                "interpolated": True, "inferred": True})
        pad_box = _dilate_box(by_frame[end], interp_dilate)
        for pad in range(end + 1, padded_end + 1):
            active_boxes.setdefault(pad, []).append({
                "track_id": state["track_id"], "frame_index": pad,
                "box": list(pad_box), "score": state["score"],
                "interpolated": True, "inferred": True})
        tracks.append({
            "track_id": state["track_id"], "start": start, "end": end,
            "padded_start": padded_start, "padded_end": padded_end,
            "gap_frames": (end - start + 1) - len(by_frame),
            "observations": len(by_frame)})

    tracks.sort(key=lambda t: t["track_id"])
    bridges = _bridge_gaps(
        tracks, active_boxes, bridge_gap_frames, bridge_min_iou,
        bridge_max_center_frac, interp_dilate)
    return active_boxes, tracks, bridges


def _bridge_gaps(tracks, active_boxes, max_gap, min_iou, max_center_frac,
                 interp_dilate):
    """Fill short gaps between consecutive tracks with interpolated boxes."""
    bridges = []
    ordered = sorted(tracks, key=lambda t: (t["start"], t["end"]))
    for previous, following in zip(ordered, ordered[1:]):
        gap = following["start"] - previous["end"] - 1
        if gap <= 0 or gap > max_gap:
            continue
        prev_box = _edge_box(active_boxes, previous["end"], previous["track_id"])
        next_box = _edge_box(active_boxes, following["start"],
                             following["track_id"])
        if prev_box is None or next_box is None:
            continue
        if not _compatible(prev_box, next_box, min_iou, max_center_frac):
            continue
        for frame in range(previous["end"] + 1, following["start"]):
            alpha = (frame - previous["end"]) / float(following["start"]
                                                      - previous["end"])
            box = _dilate_box(_lerp_box(prev_box, next_box, alpha),
                              interp_dilate)
            active_boxes.setdefault(frame, []).append({
                "track_id": previous["track_id"], "frame_index": frame,
                "box": box, "score": min(previous.get("score", 0.0),
                                         following.get("score", 0.0)),
                "interpolated": True, "inferred": True})
        bridges.append({
            "after_track": previous["track_id"],
            "before_track": following["track_id"],
            "gap_frames": gap,
            "first_bridged_frame": previous["end"] + 1,
            "last_bridged_frame": following["start"] - 1,
        })
    return bridges


def _edge_box(active_boxes, frame_index, track_id):
    """Recover the last observed (non-inferred) box of a track at a frame."""
    for entry in active_boxes.get(frame_index, []):
        if entry["track_id"] == track_id and not entry.get("inferred"):
            return list(entry["box"])
    return None
