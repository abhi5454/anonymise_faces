"""Strong, size-adaptive face obscuring: pixelate the whole box, feather outside.

Earlier revision pixelated into a *feathered ellipse inscribed in the box*, so
the box corners and border stayed sharp. RetinaFace flags came back at scores up
to 0.95 on faces whose box the pipeline had "blurred" - the leak was the rim, not
a missed detection. Two invariants now hold by construction:

1. Every pixel of the expanded box is fully pixelated. Feathering happens only
   in a ring *outside* the expanded box (the ring exists so the transition to
   sharp background is not a hard rectangle; it never un-blurs face pixels).
2. `min_block_px` makes the block size adaptive: tiny faces get blocks that are
   a fixed fraction of their size, so the pixel grid cannot preserve a
   face-like low-frequency layout.
"""

import cv2
import numpy as np


def expand_box(box, width, height, expand=0.35):
    """Expand [x1,y1,x2,y2] by a fraction, clamped to the frame."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = bw * (1.0 + expand) / 2.0, bh * (1.0 + expand) / 2.0
    return (
        max(0, int(cx - hw)),
        max(0, int(cy - hh)),
        min(width - 1, int(cx + hw)),
        min(height - 1, int(cy + hh)),
    )


def _block_grid(region_shape, pixel_blocks, min_block_px=0.0):
    """Blocks-per-axis for a region, optionally capped by a minimum block size."""
    rh, rw = region_shape[:2]
    blocks = max(1, int(pixel_blocks))
    if min_block_px and min_block_px > 0:
        cap = int(min(rw, rh) // float(min_block_px))
        blocks = max(1, min(blocks, cap))
    return blocks


def _pixelate(roi, pixel_blocks, min_block_px=0.0):
    """Replace roi with its pixelated version (nearest-neighbour upscale)."""
    rh, rw = roi.shape[:2]
    blocks = _block_grid(roi.shape, pixel_blocks, min_block_px)
    bw = max(1, rw // blocks)
    bh = max(1, rh // blocks)
    small = cv2.resize(roi, (max(1, rw // bw), max(1, rh // bh)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def _ring_mask(grown_shape, ring):
    """Mask that is 1.0 everywhere except a soft `ring`-wide border (see below).

    The mask rectangle equals the whole grown region, and the grown region is
    the expanded box plus `ring` pixels on every side, so blurring this mask
    keeps the expanded box area at ~1.0 and only fades in the outer ring.
    """
    rh, rw = grown_shape[:2]
    mask = np.ones((rh, rw), dtype=np.float32)
    if ring <= 0:
        return mask
    sigma = max(0.8, ring / 2.0)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)


def obscure_face(frame_bgr, box, expand=0.35, pixel_blocks=9, feather=True,
                 min_block_px=0.0):
    """Pixelate the expanded face region in place and return the frame.

    Guarantees the expanded box is fully pixelated; `feather` only softens the
    ring outside it. `min_block_px` adapts block size for small faces.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = expand_box(box, w, h, expand)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return frame_bgr
    rw, rh = x2 - x1, y2 - y1
    ring = max(2, int(0.2 * min(rw, rh))) if feather else 0
    gx1, gy1 = max(0, x1 - ring), max(0, y1 - ring)
    gx2, gy2 = min(w, x2 + ring), min(h, y2 + ring)
    roi = frame_bgr[gy1:gy2, gx1:gx2]
    pix = _pixelate(roi, pixel_blocks, min_block_px)
    if ring <= 0:
        roi[:, :] = pix
        return frame_bgr
    mask = _ring_mask(roi.shape, ring)[..., None]
    roi[:, :] = (pix.astype(np.float32) * mask
                 + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    return frame_bgr


def obscure_frame(frame_bgr, entries, expand=0.35, pixel_blocks=9, feather=True,
                  min_block_px=0.0):
    """Blur every tracked face entry on a copy of the frame."""
    out = frame_bgr.copy()
    for entry in entries or []:
        obscure_face(out, entry["box"], expand, pixel_blocks, feather,
                     min_block_px)
    return out
