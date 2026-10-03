"""Build contact sheets from an eval manifest: original | output, side by side.

Sheets are for *human* labeling: every cell is tagged with its sample_id,
stratum and (segment) frame, and the original is always shown next to the output
of the same sample so a judge never has to flip between files.

Usage: contact_sheet.py <manifest.csv> <out_dir> [--per-sheet N]
"""

import argparse
import csv
import os
import sys

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_rows(manifest):
    with open(manifest) as handle:
        return [row for row in csv.DictReader(handle)
                if row and row.get("sample_id") is not None]


def letterbox(img, width, height):
    """Scale to fit width x height keeping aspect ratio, pad with dark grey."""
    h, w = img.shape[:2]
    scale = min(width / float(w), height / float(h))
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np_full((height, width, 3), 32)
    oh, ow = resized.shape[:2]
    y, x = (height - oh) // 2, (width - ow) // 2
    canvas[y:y + oh, x:x + ow] = resized
    return canvas


def np_full(shape, value):
    import numpy as np
    return np.full(shape, value, dtype="uint8")


def main():
    parser = argparse.ArgumentParser(description="Build eval contact sheets.")
    parser.add_argument("manifest")
    parser.add_argument("out_dir")
    parser.add_argument("--per-sheet", type=int, default=6)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=360)
    parser.add_argument("--prefix", default="sheet")
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if not rows:
        sys.exit("no rows in %s" % args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)

    header = 46
    gutter = 6
    label_h = 26
    cell_w, cell_h = args.cell_width, args.cell_height
    cols = 2                      # original | output
    written = []
    for start in range(0, len(rows), args.per_sheet):
        chunk = rows[start:start + args.per_sheet]
        sheet_h = header + len(chunk) * (cell_h + label_h + gutter)
        sheet = np_full((sheet_h, cols * cell_w + gutter, 3), 16)
        cv2.putText(sheet,
                    "%s  rows %d-%d of %d   (LEFT: original | RIGHT: output)"
                    % (os.path.basename(args.manifest), start,
                       start + len(chunk) - 1, len(rows)),
                    (10, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y = header
        for row in chunk:
            orig = cv2.imread(row["orig_crop"])
            anon = cv2.imread(row["anon_crop"])
            if orig is None or anon is None:
                raise SystemExit("missing image for sample %s (%s / %s)"
                                 % (row["sample_id"], row["orig_crop"],
                                    row["anon_crop"]))
            tag = "id%s src%s %s frame%s%s%s" % (
                row["sample_id"], row.get("source_frame", "?"),
                row["stratum"], row["frame"],
                " SYNTH" if row.get("synthetic") in ("1", 1) else "",
                " [" + row["variant"] + "]" if row.get("variant") else "")
            for col, img in enumerate(((orig, anon))):
                sheet[y:y + cell_h,
                      gutter + col * cell_w: gutter + col * cell_w + cell_w] = \
                    letterbox(img, cell_w, cell_h)
            cv2.putText(sheet, tag, (gutter + 4, y + cell_h + 20), FONT, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            y += cell_h + label_h + gutter
        path = os.path.join(args.out_dir, "%s_%02d.jpg"
                            % (args.prefix, start // args.per_sheet))
        cv2.imwrite(path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append(path)
        print("wrote %s (%d rows)" % (path, len(chunk)), flush=True)
    print("%d sheets for %d rows" % (len(written), len(rows)), flush=True)


if __name__ == "__main__":
    main()