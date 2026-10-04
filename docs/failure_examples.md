# Failure examples (recorded data only, no footage)

Visuals from the real footage are **withheld**: no video frames or crops are
published here. The images in `docs/examples/` are synthetic schematics drawn
on a gray canvas by `make_schematics.py`. All numbers below come from recorded
pipeline output (job rows in `jobs.sqlite3`, code docstrings).

| # | Failure | Recorded evidence | Cause | Status |
|---|---------|-------------------|-------|--------|
| 1 | Rim leak (pre-fix) | `pipeline/blur.py` docstring: RetinaFace flags scored up to **0.95** on faces whose box the pipeline had "blurred" | Pre-fix code pixelated a feathered ellipse *inscribed* in the box, leaving box corners and border sharp | Fixed: every pixel of the expanded box is now fully pixelated; feathering happens only in a ring *outside* the box (`box_expand=0.35`, `pixel_blocks=9`, `min_block_px=8`). Schematic: `rim_leak_schematic.png` |
| 2 | Detection dropout gap | `pipeline/track.py` docstring: SCRFD lost a small moving face for **~10 frames**, the short track ended, the gap stayed unblurred. Job `job_66cbb7fede3b`: `bridged_gaps=1 (frames=10 fraction=0.1389 max_gap=10)`, decision `NEEDS_REVIEW` | Short track end + unbridged gap left frames with no blur box | Mitigated, still reviewed: cross-track bridging (`bridge_gap_frames=12`) fills compatible gaps and any bridged frame still blocks `SAFE` under the `strict` policy. Schematic: `dropout_gap_schematic.png` |
| 3 | Residual verifier hits (bulk) | Job `job_a69e8d7a950f`: `verifier_hits=28`, coverage `1.0` (planned 400, stride 1, threshold 0.4); e.g. frame 26 `face_1` score **0.688** box **[565.0, 446.0, 611.0, 514.0]** `near_boundary=False`. Job `job_2c17f0a52a56`: `verifier_hits=27`, coverage `1.0` (planned 450); e.g. frame 260 `face_1` score **0.934** box **[993.0, 184.0, 1114.0, 320.0]** `near_boundary=True`. Both decisions `NEEDS_REVIEW` | Faces still visible in the rendered output, caught by the independent RetinaFace pass | Working as designed: any `num_flags > 0` forces `NEEDS_REVIEW`. Schematic: `verifier_flag_schematic.png` |
| 4 | Boundary flag, disposition unknown | Job `job_66cbb7fede3b` frame 10 `face_1` score **0.443** box **[473.0, 689.0, 540.0, 775.0]** `near_track_boundary=True`; job `job_2b5b118ad6ab` frame 4 `face_1` score **0.456** box **[558.0, 497.0, 588.0, 548.0]** `near_track_boundary=True` | Flag sits at a track-boundary window where misses cluster; without human labels it cannot be called a false positive or a true leak | Open: no `labels.csv` / `output_labels.csv` exist yet, so no false-positive claim is made. See `eval/README.md` |

## Regenerate the schematics

```bash
./retina_env/bin/python docs/examples/make_schematics.py
```

Writes `rim_leak_schematic.png`, `dropout_gap_schematic.png`,
`verifier_flag_schematic.png` into this directory. The script uses only
OpenCV/Numpy drawing on a blank canvas; it reads no video, no crops.
