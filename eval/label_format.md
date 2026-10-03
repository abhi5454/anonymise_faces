# Label format (eval/label_format.md)

Label on the **original** frame; judge the **output** frame separately.

## Original-frame label

One row per face: `sample_id, x1, y1, x2, y2, cls` where `cls` is:

- `face` — identifiable by a human: >= ~50% of the face visible, or both
  eyes/nose/mouth region discernible.
- `partial` — a face is present but ambiguous (heavy occlusion, extreme pose,
  tiny). Never force yes/no; use this class.
- `no-face` — no human could identify a face here.

Boxes are pixel `[x1, y1, x2, y2]` in original-frame coordinates.

## Output-frame judgment

Per labeled face, one of:

- `covered` — no identifying features visible.
- `partial-visible` — identifiability remains despite blur (the number that
  matters; include box of the visible part in notes).
- `visible` — effectively unblurred.

## Rules

- Labeler name + date go in `labels.csv` header comments.
- Ambiguous cases get `partial` + a note, not a coin flip.
- Tools: CVAT/Label Studio, or any OpenCV box script; format above is what
  `score.py` consumes.
