# NIST-Lite v0.3.0 calibration (tiered live mix)

Measured on this box (MuJoCo 3.16). `OBS_DIM=184`.
Episodes/round: `64` (8 strata × 8). Seeds: 0–4.
Per-stratum difficulty mix ≈ 25% easy / 35% medium / 40% hard.

v0.3.0 vs v0.2.0: the observation noise model changed (independent per-slot / per-clutter /
per-peg estimate offsets) and the extra draws shift every episode, so these numbers are a
fresh sample, not a like-for-like rerun. v0.2.0 means (same harness): zero 0.012,
baseline 0.100, obs-only scripted 0.110, privileged reference 0.568. The privileged/obs-only
gap is unchanged (0.566 vs 0.124 here).

## Why RL is required

v0.1.5 round-4 king was a hand-written ONNX controller (~5.8k elementwise ops,
no matmuls). v0.2 defeats that class:

- **Pose-estimate bias**: peg/slot/clutter channels carry per-episode estimate offsets of
  4–10 mm and
  ~7–16°,
  drawn independently per slot, clutter block and peg (v0.3.0 observation
  hardening), plus a per-peg peg−slot residual, white noise, dropout, and mild drift.
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
- raw: mean=0.0226 sd=0.0122 min=0.0104 max=0.0348 (n=2)
- peg seat frac: mean=0.0227
- in-process wall: mean=158.2s [153.9,162.5]
- per-K: 3=0.052, 4=0.016, 5=0.012, 6=0.010
- by clearance: loose(>=0.60mm)=0.022, mid(0.40-0.60mm)=0.016, tight(<0.40mm)=0.048
- by clutter: high=0.016, low=0.029
- by start mix: all_upright=0.034, has_tray=0.007, mixed_lying=0.007
- by difficulty tier: easy=0.032, hard=0.004, medium=0.035

### baseline (nearest-first, obs)
- raw: mean=0.1234 sd=0.0165 min=0.0985 max=0.1455 (n=5)
- peg seat frac: mean=0.1283
- in-process wall: mean=185.0s [168.9,194.3]
- per-K: 3=0.167, 4=0.139, 5=0.113, 6=0.074
- by clearance: loose(>=0.60mm)=0.125, mid(0.40-0.60mm)=0.132, tight(<0.40mm)=0.090
- by clutter: high=0.105, low=0.142
- by start mix: all_upright=0.141, has_tray=0.117, mixed_lying=0.062
- by difficulty tier: easy=0.162, hard=0.088, medium=0.133

### obs-only scripted (anti-scripting gate)
- raw: mean=0.1238 sd=0.0212 min=0.0947 max=0.1574 (n=5)
- peg seat frac: mean=0.1274
- in-process wall: mean=180.1s [168.8,186.6]
- per-K: 3=0.177, 4=0.137, 5=0.100, 6=0.081
- by clearance: loose(>=0.60mm)=0.131, mid(0.40-0.60mm)=0.120, tight(<0.40mm)=0.106
- by clutter: high=0.108, low=0.140
- by start mix: all_upright=0.143, has_tray=0.117, mixed_lying=0.065
- by difficulty tier: easy=0.173, hard=0.082, medium=0.133

### privileged reference (tools/; true state)
- raw: mean=0.5660 sd=0.0243 min=0.5317 max=0.6043 (n=5)
- peg seat frac: mean=0.6167
- in-process wall: mean=186.9s [175.1,196.3]
- per-K: 3=0.596, 4=0.574, 5=0.528, 6=0.566
- by clearance: loose(>=0.60mm)=0.651, mid(0.40-0.60mm)=0.501, tight(<0.40mm)=0.427
- by clutter: high=0.555, low=0.577
- by start mix: all_upright=0.643, has_tray=0.541, mixed_lying=0.307
- by difficulty tier: easy=0.693, hard=0.482, medium=0.565

## ONNX baseline (2 workers)

- seed 1: raw=0.0773 wall=122.2s (in-process `tools/local_run.py`, dev build)
- `baselines/baseline.onnx` is now a memoryless rule-based twin of `NearestFirstBaseline`
  (`tools/make_policies.py`; target = nearest unseated peg, phases read from the obs).
  The v0.2 linear distillation scored 0.0077 on seed 1; the stateful Python baseline is
  0.123 above. It is a plumbing / floor check, not the score bar.

## Held-out (throwaway secret)

- reserved axes: `{'bias_drift': [1.46e-05, 1.84e-05], 'pose_bias_xy': [0.0056421, 0.0065421]}`
- reference live: 0.5423
- reference reserved (held-out shapes on hard tier): 0.4991
- drop: +0.0431

## Targets
- baseline ~0.10–0.15 → **met** (mean=0.123)
- obs-only scripted ≤ ~0.25 → **met** (mean=0.124)
- privileged reference ~0.55–0.75 → **met** (mean=0.566)
- junk ~0 → met
- eval ≤ ~400 s @ 2 workers → see ONNX wall
- PPO beats baseline and obs-only on live mix → see Learning evidence

## Learning evidence (measured on v0.2.0; not rerun for v0.3.0)
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

