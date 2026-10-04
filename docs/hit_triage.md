# Hit triage (machine-derived, NOT ground truth)

Method: `docs/hit_triage.py` re-runs Pipeline A detection+tracking **in
memory** on frozen `DEFAULT_CONFIG` (`PIPELINE_VERSION=0.3.0`, no overrides;
`det_threshold=0.3`, `det_size=[640,640]`) and overlaps each persisted
verifier flag box against the expanded blur boxes (`box_expand=0.35`,
clamped to 1920x1080) for the same frame. Reads source videos + persisted
`*verify_flags.json` only; writes `/tmp/triage_*.json` only. No pipeline,
config, or version change.

Spatial classes (no identity/leak claim — that needs human labels):
`inside` = best IoU >= 0.5 or flag-area coverage >= 0.9;
`partial` = IoU > 0.02 or coverage > 0.02; `outside` = else, including
frames with zero blur boxes (`nbox=0`).
Score bands: 0.40-0.55 / 0.55-0.70 / >=0.70 (verifier threshold 0.4).
Height bands: <40 / 40-80 / >80 px in rendered 1080p coords.

## A. 58 vs 11 resolution

`notebooks/out/job/e2e_jobs.sqlite3` holds exactly **one** job on the
downloads clip: `job_e22858b215cf`, created 2026-10-03 13:46:52,
`num_flags=11`, planned/verified 60/60, decoded/rendered 60/60. No `58`
appears in any notebook, doc, or JSON (`grep` clean); no PDF exists in the
repo. `notebooks/out/crops/` holds 66 crops, so an older unpersisted run
likely produced more flags and was overwritten. **No two persisted runs on
the same clip+config with different hit counts exist to compare —
11 is the only persisted number.**

## B1. Triage: job_e22858b215cf (notebook, downloads clip, n=11)

| frame | score | band | h | hband | class | IoU | cov | nbox | box |
|---|---|---|---|---|---|---|---|---|---|
| 13 | 0.499 | 0.40-0.55 | 118 | >80 | outside | 0.00 | 0.00 | 0 | [903,960,1048,1078] |
| 14 | 0.452 | 0.40-0.55 | 114 | >80 | outside | 0.00 | 0.00 | 0 | [907,964,1053,1078] |
| 15 | 0.591 | 0.55-0.70 | 120 | >80 | outside | 0.00 | 0.00 | 0 | [933,958,1065,1078] |
| 25 | 0.551 | 0.55-0.70 | 374 | >80 | outside | 0.00 | 0.00 | 1 | [891,704,1333,1078] |
| 26 | 0.489 | 0.40-0.55 | 25 | <40 | outside | 0.00 | 0.00 | 1 | [1867,289,1894,314] |
| 27 | 0.534 | 0.40-0.55 | 26 | <40 | outside | 0.00 | 0.00 | 1 | [1872,285,1900,311] |
| 27 | 0.412 | 0.40-0.55 | 350 | >80 | outside | 0.00 | 0.00 | 1 | [939,728,1335,1078] |
| 28 | 0.532 | 0.40-0.55 | 337 | >80 | outside | 0.00 | 0.00 | 1 | [958,741,1338,1078] |
| 33 | 0.546 | 0.40-0.55 | 42 | 40-80 | outside | 0.00 | 0.00 | 1 | [1693,92,1727,134] |
| 33 | 0.458 | 0.40-0.55 | 83 | >80 | outside | 0.00 | 0.00 | 1 | [1510,386,1570,469] |
| 34 | 0.728 | >=0.70 | 43 | 40-80 | outside | 0.00 | 0.00 | 1 | [1698,90,1732,133] |

Summary: inside 0, partial 0, **outside 11/11**; score 8/2/1;
height 2/2/7 (<40/40-80/>80). Recompute: 33 dets (all h>80), 1 track
len 33, 0 bridges. Caveat: recomputed boxes come from re-decoded
`clip_10s.mp4`; the persisted job rendered from the same clip name, so
frame indices should align, but `active_boxes` were not persisted.

## B2. Triage: job_a69e8d7a950f (eval seg1, n=28)

Summary: **inside 3, partial 0, outside 25**; score 12/7/9
(0.40-0.55/0.55-0.70/>=0.70); height 12/14/2 (<40/40-80/>80).
Inside hits (all 0.55-0.70, 40-80h, cov>=0.96 — flag sits within the blur
box footprint, i.e. verifier fired over a blurred region):
f26 s=0.688 box [565,446,611,514] iou=0.37;
f30 s=0.563 box [568,447,614,513] iou=0.40;
f37 s=0.562 box [584,449,630,514] iou=0.41.
Outside 25 include a f117-143 run of tiny (<40h) faces with nbox=0
(missed-by-A candidates, machine only). Recompute: 238 dets, 7 tracks
lens [1,1,1,2,56,89,89] (4 <5f), 1 bridge (gap 6, f151-156).

## B3. Triage: job_2c17f0a52a56 (eval seg2, n=27)

Summary: **inside 1, partial 2, outside 24**; score 5/10/12;
height 0/1/26. f229 s=0.438 inside (iou=0.56 cov=0.96, nbox=4);
f251 s=0.538 partial (iou=0.26 cov=0.40); f260 s=0.934 partial
(iou=0.03 cov=0.06); remaining 24 outside incl. f261-283 run (all >80h,
nbox=0 — missed-by-A candidates, machine only). Recompute: 124 dets,
13 tracks lens [1x8,2,7,16,90] (10 <5f), 1 bridge (gap 2, f218-219).

## Pipeline A precision proxy (300-frame recompute, clip_10s.mp4)

335 detections over 300 frames; w med/min/max 118/46/165 px,
h med/min/max 156/65/245 px (h: 328 >80, 7 in 40-80, 0 <40);
score med/min/max 0.83/0.50/0.90 (277 >=0.7, 58 in 0.5-0.7, none below
0.5 by construction at det_threshold 0.3... note floor is 0.5 in this
window). 32 tracks, lens: 24x1f + 6f + 48/52/52/53/100f;
**26 of 32 tracks <5 frames** (fragmentation proxy). 1 bridge (gap 5,
f215-219). Caveat: notebook `DET_STRIDE=5/DET_WIDTH=640` exploratory
cells differ from this stride-1 full-decode recompute.

## covered_frames: before or after bridging?

After. `pipeline/run_pipeline_a.py:113`:
`"covered_frames": len(active_boxes),` where `active_boxes` already
contains padded + interpolated + bridged entries (`pipeline/track.py`
lines 142-158 append observed/padded, 188-197 append bridged boxes).
So "293/300"-style coverage counts frames with a blur box present
post-bridging, not raw detection frames.

## Sources: two different files

`ffprobe`: `input_video.mp4` = 1920x1080, 24000/1001 fps (~23.98),
186.52 s. `downloads/face_detection_test.mp4` = 1920x1080, 30000/1001
fps (~29.97), 171.17 s. Different fps+duration = different files
(downloads clip is not a cut of input_video).
Segments/eval come from `input_video.mp4`: seg1 = source frames 120+400
(`eval_seg.map.json`: source_frame = segment_frame + 120),
seg2 = 1650+450 (`eval_seg2.mp4.map.json`: +1650). Notebook
`clip_10s.mp4` (300 frames, 29.97 fps) comes from the downloads file.
Eval manifest `source_frame` values map into input_video coordinates.

## Throughput

`docs/limitations.md` ~3.5 s/frame = RetinaFace CPU at 640-wide/stride
settings on 1080p (verify-scale cost). 409 s / 300 frames = ~1.36 s/frame
is a blended Pipeline A rate (SCRFD ~0.2-0.4 s/frame + track/blur/render
on the 300-frame clip), not a RetinaFace full-res sweep. Notebook comment
(~8 s/frame full-1080p RetinaFace, ~40 min per 300-frame sweep) is the
full-resolution Pass-1/Pass-2 cost; `DET_STRIDE=5` + `DET_WIDTH=640` cut
each sweep ~5x and ~3-4x respectively. Compare like with like: state which
pass, stride, and width each number used.

## Media checks

`grep -l "image/png" notebooks/*.ipynb` prints nothing (exit 1);
`git ls-files | grep -Ei 'mp4|mov|png|jpg|jpeg|sqlite|crops|uploads|artifacts'`
prints nothing (exit 1). No embedded images, no tracked media.

