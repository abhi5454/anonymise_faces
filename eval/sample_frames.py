"""Stratified frame sampler: heuristic strata + track-boundary frames.

Writes original/output pairs and a manifest CSV. Strata are sampling heuristics,
NOT labels; a human still judges every pair per eval/label_format.md.

Two honesty rules are enforced here:

* ``--min-frame`` excludes frames below it, so the frames used to *tune* the
  pipeline can never leak into the evaluation sample.
* ``--synthetic-frames`` adds deliberately degraded variants (downscale, darken,
  blur) of already-sampled frames, tagged ``synthetic=1`` + ``variant=`` in the
  manifest. The supplied material is a single clip at a single resolution, so
  resolution/lighting robustness can only be probed synthetically - and a
  synthetic row must never be mistaken for real footage.

Pairing is done by sequential lockstep decode of both videos, not by seeking:
OpenCV frame seeking on compressed video is unreliable, and a silently shifted
pair would produce labels for the wrong output frame.
"""

import argparse
import csv
import json
import os
import random

import cv2
import numpy as np

REAL_STRATA = ("motion", "blur", "low_light", "occlusion", "easy",
               "track_boundary")
SYNTHETIC_VARIANTS = ("720p", "480p", "darken", "blur")
MANIFEST_FIELDS = ("sample_id", "stratum", "frame", "source_frame", "synthetic",
                   "variant", "orig_crop", "anon_crop", "seed")


def laplacian_sharpness(frame):
    """Higher = sharper. Used for the blur heuristic."""
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var()


def mean_luma(frame):
    """Mean grayscale brightness, 0-255. Used for the low-light heuristic."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def classify(frame, previous, frame_no, boundaries):
    """Return the single stratum this frame belongs to, or None."""
    if frame_no in boundaries:
        return "track_boundary"
    luma = mean_luma(frame)
    sharp = laplacian_sharpness(frame)
    if previous is None:
        motion = 0.0
    else:
        motion = float(np.mean(cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY))))
    if luma < 45:
        return "low_light"
    if sharp < 60:
        return "blur"
    if motion > 12:
        return "motion"
    if frame_no % 37 == 0:
        return "occlusion"
    return "easy"


def apply_variant(frame, variant):
    """Degrade a frame for a synthetic row. Deterministic per variant."""
    if variant == "720p":
        return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    if variant == "480p":
        return cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA)
    if variant == "darken":
        return cv2.convertScaleAbs(frame, alpha=0.45, beta=0)
    if variant == "blur":
        return cv2.GaussianBlur(frame, (0, 0), sigmaX=4.0)
    raise ValueError("unknown variant %r" % (variant,))


def save(path_no_ext, frame):
    path = path_no_ext + ".jpg"
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return path



def main():
    parser = argparse.ArgumentParser(description="Sample eval frames.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--anonymised", required=True)
    parser.add_argument("--tracks-json", default=None)
    parser.add_argument("--source-start", type=int, default=0,
                        help="source_frame = frame + this (segment offset)")
    parser.add_argument("--out-dir", default="eval/out")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-frame", type=int, default=0,
                        help="exclude frames below this (leakage guard for the "
                             "frames used to tune the pipeline)")
    parser.add_argument("--synthetic-frames", type=int, default=0,
                        help="add degraded variants of N sampled frames")
    parser.add_argument("--synthetic-variants",
                        default=",".join(SYNTHETIC_VARIANTS))
    parser.add_argument("--extra-frames", default="",
                        help="comma-separated frames to force into the sample "
                             "as stratum `verifier_flagged` (used so the "
                             "verifier's own hits can be scored as caught/missed)")
    args = parser.parse_args()

    tracks = []
    if args.tracks_json and os.path.isfile(args.tracks_json):
        data = json.load(open(args.tracks_json))
        # Accept both the raw list of tracks and the worker's side-car dict
        # ({job_id, pipeline_version, tracks, bridges}).
        tracks = data.get("tracks", []) if isinstance(data, dict) else data
    boundaries = set()
    for track in tracks:
        for anchor in (track.get("padded_start"), track.get("padded_end")):
            if anchor is not None:
                boundaries.update(range(max(0, anchor - 3), anchor + 4))

    rng = random.Random(args.seed)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames:
        total = min(total, args.max_frames)
    strata = {name: [] for name in REAL_STRATA}
    previous = None
    for frame_no in range(total):
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        previous_for_motion, previous = previous, frame
        if frame_no < args.min_frame:
            continue
        stratum = classify(frame, previous_for_motion, frame_no, boundaries)
        if stratum:
            strata[stratum].append({"frame": frame_no})
    cap.release()
    empty = [name for name, entries in strata.items() if not entries]
    if empty:
        print("WARNING empty strata after --min-frame %d: %s"
              % (args.min_frame, ", ".join(empty)), flush=True)

    picks = []
    for name in REAL_STRATA:
        entries = strata[name]
        for entry in rng.sample(entries, min(args.per_stratum, len(entries))):
            picks.append([name, entry["frame"], 0, ""])
    for raw in args.extra_frames.split(","):
        raw = raw.strip()
        if not raw:
            continue
        frame_no = int(raw)
        if frame_no < args.min_frame:
            continue
        picks.append(["verifier_flagged", frame_no, 0, ""])
    picks.sort(key=lambda item: (item[1], item[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = {row[1] for row in picks}
    stratum_of = {row[1]: row[0] for row in picks}
    pairs = {}
    src, anon = cv2.VideoCapture(args.video), cv2.VideoCapture(args.anonymised)
    src_frames = int(src.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    anon_frames = int(anon.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if src_frames != anon_frames:
        raise SystemExit("frame count mismatch: original=%d output=%d"
                         % (src_frames, anon_frames))
    # Sequential lockstep decode, never seeking: a shifted pair would label the
    # wrong output frame, and seeking on compressed video can silently do that.
    frame_no = 0
    while wanted:
        ok_o, orig = src.read()
        ok_a, blurred = anon.read()
        if not ok_o or not ok_a or orig is None or blurred is None:
            break
        if frame_no in wanted:
            stratum = stratum_of[frame_no]
            pairs[frame_no] = (
                save(os.path.join(args.out_dir,
                                  "orig_%s_%05d" % (stratum, frame_no)), orig),
                save(os.path.join(args.out_dir,
                                  "anon_%s_%05d" % (stratum, frame_no)), blurred))
            wanted.discard(frame_no)
        frame_no += 1
    src.release()
    anon.release()
    if wanted:
        raise SystemExit("could not read frames %s from the pair"
                         % sorted(wanted))
    rows = []
    sample_id = 0
    for stratum, frame_no, synthetic, variant in picks:
        orig_p, anon_p = pairs[frame_no]
        rows.append([sample_id, stratum, frame_no,
                     frame_no + args.source_start, synthetic, variant,
                     orig_p, anon_p, args.seed])
        sample_id += 1

    # Synthetic degradation of already-sampled frames. Tagged in the manifest,
    # never mixed in silently: the supplied material is one clip at one
    # resolution, so resolution/lighting robustness can only be probed this way
    # and a synthetic row must never be mistaken for real footage.
    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    if args.synthetic_frames and variants:
        bases = [row for row in rows if not row[4]][:args.synthetic_frames]
        for base in bases:
            frame_no, source_frame, stratum = base[2], base[3], base[1]
            for variant in variants:
                orig, blurred = cv2.imread(base[6]), cv2.imread(base[7])
                rows.append([
                    sample_id, "synthetic", frame_no, source_frame, 1, variant,
                    save(os.path.join(args.out_dir,
                                      "synorig_%s_%05d" % (variant, frame_no)),
                         apply_variant(orig, variant)),
                    save(os.path.join(args.out_dir,
                                      "synanon_%s_%05d" % (variant, frame_no)),
                         apply_variant(blurred, variant)),
                    args.seed])
                sample_id += 1

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        writer.writerows(rows)
    print("wrote %s with %d samples (%d real, %d synthetic)"
          % (manifest_path, len(rows),
             sum(1 for r in rows if not r[4]),
             sum(1 for r in rows if r[4])), flush=True)
    print("strata: " + ", ".join("%s=%d" % (name, len(strata[name]))
                                 for name in REAL_STRATA), flush=True)
    print("excluded frames < %d (tuning leakage guard); source_frame = "
          "frame + %d" % (args.min_frame, args.source_start), flush=True)




if __name__ == "__main__":
    main()
