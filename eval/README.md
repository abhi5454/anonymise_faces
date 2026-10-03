# Eval harness

1. Sample: `./retina_env/bin/python -m eval.sample_frames --video input_video.mp4
   --anonymised artifacts/<job-output>.mp4 --tracks-json <tracks> --out-dir eval/out`
   (fixed `--seed`, heuristic strata + track-boundary frames).
2. Label originals per `label_format.md` -> `labels.csv`.
3. Judge outputs -> `output_labels.csv` (`covered|partial-visible|visible`).
4. Score: `./retina_env/bin/python -m eval.score --manifest eval/out/manifest.csv
   --labels labels.csv --output-labels output_labels.csv
   --verifier-flags <job-verify.json>`.

Outputs the per-stratum table from the plan:

```text
Failure type | GT faces | Missed by A | Residual visible | Caught by B | Clip outcome
```

Metrics: Pipeline A miss rate, end-to-end residual visible rate (the number
that matters), verifier recall on residual faces, false-alarm rate on clean
frames, job-level SAFE-with-residual and review rates — all with Wilson 95%
intervals. No labels exist yet; do not report numbers before labeling.
