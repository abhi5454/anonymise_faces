"""Pipeline A driver: probe -> SCRFD -> track -> blur -> ffmpeg render."""

import os
import time

import cv2

from common import config as config_mod
from common.hashing import atomic_rename, content_addressed_path, sha256_file
from common.video_info import probe_video
from pipeline.blur import obscure_frame
from pipeline.detect import build_anonymizer_model, detect_faces_scrfd
from pipeline.render import render_blurred_video, temp_artifact_path
from pipeline.track import track_detections


def _decode_and_detect(src_path, model, cfg, max_frames=0, progress_every=200):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("could not open video: %s" % src_path)
    detections, decoded, errors = {}, 0, 0
    try:
        while True:
            if max_frames and decoded >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                if max_frames and decoded >= max_frames:
                    break
                errors += 1
                break
            dets = detect_faces_scrfd(
                frame,
                model,
                det_threshold=cfg["anonymizer"]["det_threshold"],
                max_num=cfg["anonymizer"]["max_num"],
            )
            if dets:
                detections[decoded] = dets
            decoded += 1
            if progress_every and decoded % progress_every == 0:
                print("  pipeline A decoded=%d" % decoded, flush=True)
    finally:
        cap.release()
    return detections, decoded, errors


def run_pipeline_a(src_path, out_dir, user_config=None, max_frames=0, model=None):
    """Run Pipeline A. Returns a dict with artifact paths and statistics."""
    started = time.time()
    cfg = config_mod.canonical_config(user_config)
    info = probe_video(src_path)
    fps = info["fps"] or 25.0
    width, height = info["width"], info["height"]

    own_model = False
    if model is None:
        model = build_anonymizer_model(
            det_size=tuple(cfg["anonymizer"]["det_size"]),
            ctx_id=cfg["anonymizer"]["ctx_id"],
        )
        own_model = True

    detections, decoded, decode_errors = _decode_and_detect(
        src_path, model, cfg, max_frames=max_frames)
    print("pipeline A: tracked-input frames=%d decoded=%d" % (
        len(detections), decoded), flush=True)

    active_boxes, tracks, bridges = track_detections(
        detections,
        iou_threshold=cfg["tracker"]["iou_threshold"],
        max_missed=cfg["tracker"]["max_missed"],
        min_score=cfg["tracker"]["min_score"],
        pad_before=cfg["tracker"]["pad_before"],
        pad_after=cfg["tracker"]["pad_after"],
        bridge_gap_frames=cfg["tracker"]["bridge_gap_frames"],
        bridge_min_iou=cfg["tracker"]["bridge_min_iou"],
        bridge_max_center_frac=cfg["tracker"]["bridge_max_center_frac"],
        interp_dilate=cfg["tracker"]["interp_dilate"],
    )

    def blur_fn(frame, frame_index, boxes):
        return obscure_frame(
            frame,
            boxes.get(frame_index),
            expand=cfg["anonymizer"]["box_expand"],
            pixel_blocks=cfg["anonymizer"]["pixel_blocks"],
            feather=cfg["anonymizer"]["feather"],
            min_block_px=cfg["anonymizer"]["min_block_px"],
        )

    tmp_out = temp_artifact_path(".mp4")
    rendered = render_blurred_video(
        src_path, active_boxes, tmp_out, fps, width, height,
        cfg["render"], blur_fn, max_frames=max_frames)
    digest = sha256_file(tmp_out)
    final_path = content_addressed_path(out_dir, "anon-" + digest[:16], ".mp4")
    atomic_rename(tmp_out, final_path)

    return {
        "output_path": final_path,
        "output_sha256": digest,
        "config": cfg,
        "pipeline_version": config_mod.PIPELINE_VERSION,
        "input_info": dict(info, decoded_frames=decoded,
                           decode_errors=decode_errors),
        "rendered_frames": rendered,
        "tracks": tracks,
        "bridges": bridges,
        "num_tracks": len(tracks),
        "num_bridged_gaps": len(bridges),
        "bridged_frames": sum(b["gap_frames"] for b in bridges),
        "covered_frames": len(active_boxes),
        "frames_with_detections": len(detections),
        "elapsed_seconds": round(time.time() - started, 2),
    }
