# NIST-Lite v0.2.0 calibration (tiered live mix)

Measured on this box (MuJoCo 3.16). `OBS_DIM=184`.
Episodes/round: `64` (8 strata × 8). Seeds: 0–4.
Per-stratum difficulty mix ≈ 25% easy / 35% medium / 40% hard.

## Why RL is required

v0.1.5 round-4 king was a hand-written ONNX controller (~5.8k elementwise ops,
no matmuls). v0.2 defeats that class:

- **Pose-estimate bias**: peg/slot channels carry a fixed per-episode bias of
  4–10 mm and
  ~7–16°,
  plus an independent peg−slot residual, white noise, dropout, and mild drift.
- **Wrist force/torque** for contact-driven search.
- **Hidden dynamics**: friction, mass, gain, mild backlash, chamfer, rare delay,
  mid-episode shove/drop.
- **Held-out shapes**: `('hex', 'dflat')` on the reserved hard tier only.

## Difficulty (live, tiered)

- clearance `0.25–0.80 mm`
- STEPS_PER_PEG `280`
- **easy** (~25%): grasped / near-hole starts, reduced noise, residual bias kept
  large enough that −rel scripts miss; no drift/disturbance
- **medium** (~35%): mixed upright starts, moderated anti-scripting
- **hard** (~40%): full residual bias / drift / disturbances

### Junk (zero policy)
- raw: mean=0.0117 sd=0.0039 min=0.0078 max=0.0156 (n=2)
- peg seat frac: mean=0.0117
- in-process wall: mean=139.2s [135.8,142.5]
- per-K: 3=0.000, 4=0.031, 5=0.000, 6=0.016
- by clearance: loose(>=0.60mm)=0.017, mid(0.40-0.60mm)=0.008, tight(<0.40mm)=0.000
- by clutter: high=0.013, low=0.010
- by start mix: all_upright=0.013, has_tray=0.008, mixed_lying=0.012
- by difficulty tier: easy=0.021, hard=0.012, medium=0.005

### baseline (nearest-first, obs)
- raw: mean=0.1002 sd=0.0069 min=0.0917 max=0.1106 (n=5)
- peg seat frac: mean=0.1042
- in-process wall: mean=145.0s [139.7,150.2]
- per-K: 3=0.124, 4=0.102, 5=0.084, 6=0.091
- by clearance: loose(>=0.60mm)=0.100, mid(0.40-0.60mm)=0.104, tight(<0.40mm)=0.092
- by clutter: high=0.097, low=0.103
- by start mix: all_upright=0.106, has_tray=0.090, mixed_lying=0.097
- by difficulty tier: easy=0.124, hard=0.077, medium=0.107

### obs-only scripted (anti-scripting gate)
- raw: mean=0.1097 sd=0.0127 min=0.0889 max=0.1243 (n=5)
- peg seat frac: mean=0.1129
- in-process wall: mean=142.7s [131.2,152.3]
- per-K: 3=0.141, 4=0.097, 5=0.095, 6=0.106
- by clearance: loose(>=0.60mm)=0.113, mid(0.40-0.60mm)=0.111, tight(<0.40mm)=0.099
- by clutter: high=0.108, low=0.112
- by start mix: all_upright=0.115, has_tray=0.094, mixed_lying=0.109
- by difficulty tier: easy=0.157, hard=0.075, medium=0.113

### privileged reference (tools/; true state)
- raw: mean=0.5684 sd=0.0113 min=0.5551 max=0.5893 (n=5)
- peg seat frac: mean=0.6204
- in-process wall: mean=153.3s [144.7,165.1]
- per-K: 3=0.620, 4=0.553, 5=0.528, 6=0.572
- by clearance: loose(>=0.60mm)=0.658, mid(0.40-0.60mm)=0.493, tight(<0.40mm)=0.479
- by clutter: high=0.580, low=0.557
- by start mix: all_upright=0.625, has_tray=0.501, mixed_lying=0.408
- by difficulty tier: easy=0.691, hard=0.520, medium=0.535

## ONNX baseline (2 workers)

- seed 1: raw=0.0196 wall=85.4s (measured before the final env retune; stale)
- seed 1 on the released v0.2.0 images (podman, 2 CPU / 2 GiB, `fixtures/input.json`):
  raw=0.0077, 64 episodes, referee wall 113 s. The memoryless ONNX twin in
  `tools/make_policies.py` is much weaker than the stateful Python `NearestFirstBaseline`
  (0.100 above); it is a plumbing / floor check, not the score bar.

## Held-out (throwaway secret)

- reserved axes: `{'bias_drift': [1.46e-05, 1.84e-05], 'pose_bias_xy': [0.0056421, 0.0065421]}`
- reference live: 0.5629
- reference reserved (held-out shapes on hard tier): 0.5003
- drop: +0.0626

## Targets
- baseline ~0.10–0.15 → **met** (mean=0.100)
- obs-only scripted ≤ ~0.25 → **met** (mean=0.110)
- privileged reference ~0.55–0.75 → **met** (mean=0.568)
- junk ~0 → met
- eval ≤ ~400 s @ 2 workers → see ONNX wall
- PPO beats baseline and obs-only on live mix → see Learning evidence

## Learning evidence
- Residual PPO on `ObsOnlyScripted` (zero-init + light privileged residual demos),
  ~100k steps / 2 workers on the tiered live mix.
- Live curve (3-seed eval): 0.106 → **0.115** best; final 5-seed:
  **ppo=0.1066** vs obs=0.0959 vs base=0.0903
  (Δ ppo−obs=+0.011, Δ ppo−base=+0.016).
- Curve steps/scores: `out/ppo_curve.npz`.

## Solvers
- `tools/reference_solver.py` — privileged; not in player
- `ObsOnlyScripted` — strongest obs-only hand-coded controller
- `tools/train_ppo.py` — residual PPO (scripted base + privileged demos) on tiered live mix

