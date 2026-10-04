"""Draw synthetic failure-mode schematics on a blank gray canvas.

No video frames or crops are read or written. Outputs go to this directory:
  rim_leak_schematic.png      - pre-fix inscribed ellipse left the box rim sharp
  dropout_gap_schematic.png   - tracked/blurred segments with an unblurred gap
  verifier_flag_schematic.png - pixelated box with a verifier detection overlay

Run: ./retina_env/bin/python docs/examples/make_schematics.py
"""

import os

import cv2
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
W, H = 640, 360
GRAY = (90, 90, 90)
LIGHT = (150, 150, 150)
WHITE = (235, 235, 235)
RED = (60, 60, 220)
GREEN = (80, 180, 80)
YELLOW = (80, 200, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def canvas():
    return np.full((H, W, 3), GRAY, dtype=np.uint8)


def pixel_fill(img, x1, y1, x2, y2, blocks=9):
    """Suggest pixelation with a coarse block grid (schematic only)."""
    bw = max(1, (x2 - x1) // blocks)
    bh = max(1, (y2 - y1) // blocks)
    rng = np.random.RandomState(7)
    for i in range(blocks):
        for j in range(blocks):
            v = int(110 + rng.rand() * 90)
            cv2.rectangle(img, (x1 + i * bw, y1 + j * bh),
                          (min(x2, x1 + (i + 1) * bw), min(y2, y1 + (j + 1) * bh)),
                          (v, v, v), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 2)


def rim_leak():
    img = canvas()
    x1, y1, x2, y2 = 220, 60, 420, 300
    pixel_fill(img, x1, y1, x2, y2)
    # pre-fix inscribed ellipse: rim between ellipse and box edge stayed sharp
    cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2),
                ((x2 - x1) // 2 - 8, (y2 - y1) // 2 - 8), 0, 0, 360, YELLOW, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
    cv2.putText(img, "expanded box (fully pixelated now)", (x1 - 190, y1 - 12),
                FONT, 0.5, WHITE, 1)
    cv2.putText(img, "pre-fix ellipse: rim leaked (scores to 0.95)",
                (x1 - 200, y2 + 24), FONT, 0.5, YELLOW, 1)
    cv2.imwrite(os.path.join(OUT_DIR, "rim_leak_schematic.png"), img)


def dropout_gap():
    img = canvas()
    y, h = 150, 60
    cv2.rectangle(img, (40, y), (280, y + h), GREEN, -1)
    cv2.rectangle(img, (400, y), (600, y + h), GREEN, -1)
    cv2.rectangle(img, (280, y), (400, y + h), RED, -1)
    cv2.putText(img, "tracked+blurred", (70, y + 36), FONT, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "GAP ~10f unblurred", (262, y - 14), FONT, 0.55, RED, 2)
    cv2.putText(img, "tracked+blurred", (425, y + 36), FONT, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "dropout: SCRFD lost face, gap left unblurred -> NEEDS_REVIEW",
                (40, y + h + 40), FONT, 0.5, WHITE, 1)
    cv2.imwrite(os.path.join(OUT_DIR, "dropout_gap_schematic.png"), img)


def verifier_flag():
    img = canvas()
    x1, y1, x2, y2 = 240, 70, 400, 290
    pixel_fill(img, x1, y1, x2, y2)
    # verifier detection box slightly shifted/smaller: residual visible face
    cv2.rectangle(img, (x1 + 18, y1 + 10), (x2 - 8, y2 - 30), RED, 2)
    cv2.putText(img, "RetinaFace hit e.g. score 0.688", (x1 - 60, y1 - 12),
                FONT, 0.5, RED, 1)
    cv2.putText(img, "pixelated box", (x1 + 30, y2 + 24), FONT, 0.5, WHITE, 1)
    cv2.imwrite(os.path.join(OUT_DIR, "verifier_flag_schematic.png"), img)


if __name__ == "__main__":
    rim_leak()
    dropout_gap()
    verifier_flag()
    print("wrote 3 schematics to", OUT_DIR)
