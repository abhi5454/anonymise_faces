"""Merge per-segment eval manifests into one manifest with per-stratum caps.

The supplied material is a single clip, so the eval sample is drawn from two
cut segments of it (one contains no blur/low-light frames, the other does).
Each segment was sampled independently with the same seed; this merges them,
keeps every stratum that exists, trims the abundant strata to a cap so no one
stratum dominates, renumbers sample_ids, and reports the final composition.

Usage:
  merge_manifests.py out.csv --in seg1.csv --in seg2.csv \\
      --cap stratum=N --cap stratum=N ... [--synthetic-caps stratum=N]

Rows for a stratum are thinned evenly (not from the front) so the surviving
frames stay spread across the segment rather than clustering at its start.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

DEFAULT_CAPS = {
    "motion": 10,
    "easy": 10,
    "track_boundary": 10,
    "occlusion": 4,
    "blur": 8,
    "low_light": 8,
    "verifier_flagged": 10,
    "synthetic": 8,
}


def parse_caps(values, defaults):
    caps = dict(defaults)
    for value in values or []:
        name, _, count = value.partition("=")
        if not count.isdigit():
            raise SystemExit("bad cap %r (want stratum=N)" % value)
        caps[name.strip()] = int(count)
    return caps


def thin_evenly(rows, cap):
    """Keep at most `cap` rows, spread evenly across the input order."""
    if len(rows) <= cap:
        return list(rows)
    if cap <= 0:
        return []
    if cap == 1:
        return [rows[len(rows) // 2]]
    step = (len(rows) - 1) / float(cap - 1)
    return [rows[int(round(i * step))] for i in range(cap)]


def thin_balanced(rows, cap):
    """Thin rows that carry a `variant` label, keeping every variant present.

    Plain even-thinning over [baseA v1..v4, baseB v1..v4, ...] happens to pick
    the same two variants repeatedly, so a 4-variant stratum would come back
    with lopsided coverage. Balance across variants instead: equal share each,
    remainder spread over the variants with the most rows.
    """
    variants = OrderedDict()
    for row in rows:
        variants.setdefault(row.get("variant") or "", []).append(row)
    if len(variants) <= 1:
        return thin_evenly(rows, cap)
    names = sorted(variants, key=lambda name: (-len(variants[name]), name))
    base, remainder = divmod(cap, len(names))
    kept = []
    for index, name in enumerate(names):
        share = base + (1 if index < remainder else 0)
        kept.extend(thin_evenly(variants[name], share))
    kept.sort(key=lambda r: (r.get("variant") or "", int(r["frame"])))
    return kept


def main():
    parser = argparse.ArgumentParser(description="Merge eval manifests.")
    parser.add_argument("out")
    parser.add_argument("--in", dest="inputs", action="append", required=True,
                        help="manifest to merge (repeatable)")
    parser.add_argument("--cap", action="append",
                        help="stratum=N, override a per-stratum cap")
    args = parser.parse_args()

    caps = parse_caps(args.cap, DEFAULT_CAPS)
    by_stratum = OrderedDict()
    fields = None
    for path in args.inputs:
        if not os.path.isfile(path):
            raise SystemExit("missing manifest %s" % path)
        with open(path) as handle:
            reader = csv.DictReader(handle)
            fields = fields or reader.fieldnames
            if reader.fieldnames != fields:
                raise SystemExit("manifest header mismatch: %s vs %s"
                                 % (path, fields))
            for row in reader:
                by_stratum.setdefault(row["stratum"], []).append(row)

    kept = []
    for stratum in sorted(by_stratum):
        cap = caps.get(stratum, 10)
        rows = by_stratum[stratum]
        has_variants = len({r.get("variant") or "" for r in rows}) > 1
        chosen = thin_balanced(rows, cap) if has_variants else thin_evenly(rows, cap)
        chosen.sort(key=lambda r: (r["stratum"], int(r["frame"])))
        kept.extend(chosen)
        if len(rows) != len(chosen):
            print("  %-18s kept %d/%d (cap %d)"
                  % (stratum, len(chosen), len(rows), cap), flush=True)

    kept.sort(key=lambda r: (r.get("source_frame", r["frame"])))
    for index, row in enumerate(kept):
        row["sample_id"] = str(index)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    real = [r for r in kept if r.get("synthetic") in ("", "0", 0)]
    synth = [r for r in kept if r.get("synthetic") in ("1", 1)]
    counts = OrderedDict()
    for row in kept:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    print("wrote %s: %d rows (%d real, %d synthetic)"
          % (args.out, len(kept), len(real), len(synth)), flush=True)
    print("composition: " + ", ".join("%s=%d" % kv for kv in counts.items()),
          flush=True)
    frames = [int(r["source_frame"]) for r in kept]
    print("source_frame range: %d..%d (tuning frames 0-119 excluded: %s)"
          % (min(frames), max(frames),
             "yes" if min(frames) >= 120 else "NO - LEAKAGE"), flush=True)
    if min(frames) < 120:
        sys.exit("refusing to write a manifest containing tuning frames")


if __name__ == "__main__":
    main()