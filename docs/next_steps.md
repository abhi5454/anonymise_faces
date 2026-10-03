# Next steps

1. Backward-pass tracking: propagate boxes before first detection.
2. Ensemble second detector with union voting; person-detector fallback that
   blurs head regions when face confidence is low.
3. Tiled inference for small faces; low-light enhancement before detection.
4. Chunked processing for long recordings: overlapping keyframe-aligned
   segments, per-chunk idempotency keys, stitch step, per-chunk review flags.
5. Calibrated thresholds from labeled data; per-camera miss-proxy metrics.
6. GPU batching, checkpointing, audit log of model version per output.
