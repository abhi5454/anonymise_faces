# Limitations

- CPU-only on the reference machine (Apple-silicon laptop, PyTorch, no GPU):
  measured 2026-10-04 on a 1080p clip — RetinaFace ~3.5 s/frame (warm model,
  threshold 0.4, full-res or 640-wide input alike), SCRFD ~0.2-0.4 s/frame.
  Full 1080p RetinaFace sweeps are hours-scale per pass on a long clip (the
  42-min Pass 2 was this cost), so the demo notebook narrows work with
  `DET_STRIDE=5`, `DET_WIDTH=640`, and the verify gate instead of a second
  full sweep. GPU batching is still the real fix (see next steps).
- Tracker bridges gaps inside a track but cannot find faces before a track's
  first detection (no backward pass yet; see next steps).
- Offline padding reuses edge boxes; fast motion at track edges can leak.
- Pixelation strength scales with face size; tiny faces get coarse blocks.
- SQLite + in-process worker stand in for SQS/Celery; one worker at a time.
- `input_video.mp4` is AV1 1080p; ffprobe reports `nb_frames=N/A`, so frame
  accounting relies on actual decode counts, not container metadata.
