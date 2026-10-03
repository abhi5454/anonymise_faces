# Shared failure modes

Pipeline B (RetinaFace verifier) is **independent in weights and
architecture, not in blind spots**. A verifier "pass" is weaker evidence than
it looks. This file states the correlations explicitly.

## 1. Same data distribution

Both SCRFD and RetinaFace train largely on WIDER FACE-style data. Both are
weaker on:

- tiny faces (few pixels after downscaling),
- extreme poses and profile views,
- infrared / low-light imagery,
- heavy motion blur,
- occluded faces (hands, masks, foreground objects).

A face that is hard for Pipeline A for these reasons is also more likely to
be missed by Pipeline B.

## 2. Same input degradation

If a frame is too dark, blurred, or compressed for one detector, it is likely
so for the other. Failures are correlated, not independent. Mitigations:

- run the verifier at a different scale / stride from Pipeline A;
- sample track-boundary neighbourhoods, where misses cluster;
- keep a human review queue for flagged and low-confidence segments.

## 3. Same preprocessing risk

If both stages downscale to ~640 px, small faces vanish in both. Mitigations:

- different detector input sizes between stages;
- tiled inference for small faces (future work);
- explicit review trigger for resolutions outside the tested range.

## 4. Non-face identifying content (out of scope)

The verifier cannot see tattoos, name tags, license plates, clothing logos, or
the back of a head. None of these produce a verifier hit. If they matter for
the deployment, they need their own stage (e.g. person/head-region fallback).

## 5. Verifier blindness after heavy blur

Heavy pixelation removes the signal RetinaFace needs, so a fully blurred face
is correctly silent. Pipeline B therefore catches **leaky, partial, shifted,
or entirely missing blur over a still-visible face** — which is the main
failure that matters. A face missed by A that stays sharp in the output is
exactly what B is positioned to catch; a face missed by A *and* invisible to
B (tiny/dark/occluded) is the residual risk carried by review sampling.

## 6. What the numbers must not claim

- A 0% verifier-hit rate is not a 0% miss rate.
- Report Pipeline A miss rate, end-to-end residual visible rate, and verifier
  recall separately, with Wilson intervals and the honest caveat that n is
  small. See `eval/README.md`.
