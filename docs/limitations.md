# Limitations

- Tracker bridges gaps inside a track but cannot find faces before a track's
  first detection (no backward pass yet; see next steps).
- Offline padding reuses edge boxes; fast motion at track edges can leak.
- Pixelation strength scales with face size; tiny faces get coarse blocks.
- SQLite + in-process worker stand in for SQS/Celery; one worker at a time.
- CPU-only; ~0.5 s/frame RetinaFace verify, SCRFD faster per frame.
- `input_video.mp4` is AV1 1080p; ffprobe reports `nb_frames=N/A`, so frame
  accounting relies on actual decode counts, not container metadata.
