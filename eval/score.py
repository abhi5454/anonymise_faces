"""Score labeled eval pairs: miss rate, residual rate, verifier recall.

Inputs: manifest.csv from sample_frames.py, labels.csv + output_labels.csv in
the label_format.md schema, and the job's verifier flags JSON.
Prints a per-stratum table with Wilson 95% intervals.
"""

import argparse
import csv
import json
import math
from collections import defaultdict


def wilson(p, n, z=1.96):
    """Wilson score interval for proportion p with n trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_labels(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True,
                        help="labels.csv on original frames")
    parser.add_argument("--output-labels", required=True,
                        help="covered|partial-visible|visible per face")
    parser.add_argument("--verifier-flags", default=None)
    args = parser.parse_args()

    manifest = {row["sample_id"]: row for row in
                csv.DictReader(open(args.manifest))}
    gt = defaultdict(list)   # sample_id -> [face rows]
    for row in load_labels(args.labels):
        gt[row[0]].append(row)
    judged = {}              # (sample_id, face_idx) -> verdict
    for row in load_labels(args.output_labels):
        judged[(row[0], row[1])] = row[2]

    flagged_frames = set()
    if args.verifier_flags and open(args.verifier_flags):
        flags = json.load(open(args.verifier_flags))
        flagged_frames = {f["frame_index"] for f in
                          flags.get("flags", flags) if isinstance(f, dict)}

    stats = defaultdict(lambda: {"gt": 0, "missed": 0, "residual": 0,
                                 "caught": 0, "false_alarm_frames": 0,
                                 "clean_frames": 0})
    for sample_id, faces in gt.items():
        stratum = manifest.get(sample_id, {}).get("stratum", "unknown")
        frame_no = int(manifest.get(sample_id, {}).get("frame", -1))
        stat = stats[stratum]
        clean = True
        for idx, face in enumerate(faces):
            cls = face[5] if len(face) > 5 else "face"
            if cls == "no-face":
                continue
            if cls == "partial":
                continue  # reported separately, not forced into rates
            clean = False
            stat["gt"] += 1
            verdict = judged.get((sample_id, str(idx)), "visible")
            if verdict != "covered":
                stat["missed"] += 1
            if verdict in ("partial-visible", "visible"):
                stat["residual"] += 1
                if frame_no in flagged_frames:
                    stat["caught"] += 1
        if clean:
            stat["clean_frames"] += 1
            if frame_no in flagged_frames:
                stat["false_alarm_frames"] += 1

    print("%-14s %6s %6s %8s %8s %8s" % (
        "stratum", "gt", "missed", "residual", "caught", "fa/clean"))
    for stratum in sorted(stats):
        s = stats[stratum]
        miss_lo, miss_hi = wilson(s["missed"] / s["gt"] if s["gt"] else 0,
                                  s["gt"])
        print("%-14s %6d %6d %8d %8d %5d/%d  missCI=[%.2f,%.2f]" % (
            stratum, s["gt"], s["missed"], s["residual"], s["caught"],
            s["false_alarm_frames"], s["clean_frames"], miss_lo, miss_hi))
    print("\nn is small: treat intervals as honesty bounds, not precision.")


if __name__ == "__main__":
    main()
