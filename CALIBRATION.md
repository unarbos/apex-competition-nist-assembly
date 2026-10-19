# NIST-Lite v0.2.0 calibration (anti-scripting)

Measured on this box (MuJoCo 3.16). `OBS_DIM=184`.
Episodes/round: `32` (8 strata × 4). Seeds: 0–9.

## Why RL is required

v0.1.5 round-4 king was a hand-written ONNX controller (~5.8k elementwise ops,
no matmuls). v0.2 defeats that class:

- **Pose-estimate bias**: peg/slot channels carry a fixed per-episode bias of
  4–10 mm and
  ~7–16°,
  plus white noise and dropout. Exact hole/peg poses are never revealed.
- **Wrist force/torque** for contact-driven search.
- **Hidden dynamics**: friction, mass, gain, mild backlash, chamfer, rare delay,
  mid-episode shove/drop.
- **Held-out shapes**: `('hex', 'dflat')` on the reserved split only.

## Difficulty (live)

- clearance `0.25–0.80 mm`
- STEPS_PER_PEG `280`

## Junk (zero policy)

- zero: mean=0.0274 range=[0.0104,0.0395]

### baseline (nearest-first, obs)
- raw: mean=0.0900 sd=0.0166 min=0.0607 max=0.1131 (n=10)
- peg seat frac: mean=0.0929
- in-process wall: mean=76.7s [72.5,78.6]
- per-K: 3=0.089, 4=0.106, 5=0.095, 6=0.070
- by clearance: loose(>=0.60mm)=0.097, mid(0.40-0.60mm)=0.089, tight(<0.40mm)=0.079
- by clutter: high=0.090, low=0.090
- by start mix: all_upright=0.100, has_tray=0.096, mixed_lying=0.084

### obs-only scripted (anti-scripting gate)
- raw: mean=0.0167 sd=0.0099 min=0.0048 max=0.0324 (n=10)
- peg seat frac: mean=0.0179
- in-process wall: mean=58.5s [52.3,65.7]
- per-K: 3=0.020, 4=0.017, 5=0.016, 6=0.014
- by clearance: loose(>=0.60mm)=0.027, mid(0.40-0.60mm)=0.012, tight(<0.40mm)=0.012
- by clutter: high=0.020, low=0.013
- by start mix: all_upright=0.020, has_tray=0.013, mixed_lying=0.021

### privileged reference (tools/; true state)
- raw: mean=0.5102 sd=0.0406 min=0.4264 max=0.5533 (n=10)
- peg seat frac: mean=0.5581
- in-process wall: mean=87.3s [79.7,93.2]
- per-K: 3=0.548, 4=0.530, 5=0.490, 6=0.473
- by clearance: loose(>=0.60mm)=0.617, mid(0.40-0.60mm)=0.494, tight(<0.40mm)=0.406
- by clutter: high=0.494, low=0.527
- by start mix: all_upright=0.564, has_tray=0.596, mixed_lying=0.411

## ONNX baseline (2 workers)

- seed 1: raw=0.0134 wall=83.7s

## Held-out (throwaway secret)

- reserved axes: `{'backlash': [0.0002548, 0.0003073], 'pose_bias_xy': [0.0056421, 0.0065421]}`
- reference live: 0.4942
- reference reserved (held-out shapes): 0.3312
- drop: +0.1630

## Targets
- privileged reference 0.50–0.75 → **met** (mean=0.510)
- obs-only scripted ≤ ~0.25 → **met** (mean=0.017)
- baseline ≤ ~0.15 → **met** (mean=0.090)
- junk ~0 → met
- eval ≤ ~400 s @ 2 workers → see ONNX wall

## Learning evidence
- quick_learn hillclimb best=0.0305 (warm=0.0123); held delta vs obs=+0.0000

## Solvers
- `tools/reference_solver.py` — privileged; not in player
- `ObsOnlyScripted` — strongest obs-only hand-coded controller

