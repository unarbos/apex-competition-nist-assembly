# NIST-Lite v0.5.0 calibration (generalization hardening)

Measured on this box (MuJoCo 3.16, CPU) with `tools/eval_policies.py` (same episode draw and
per-episode RNG streams as the referee, dev build). Live mix, 64 episodes per seed
(8 strata × 8), seeds 1–5 unless noted. `OBS_DIM=184`.

## Summary: before (v0.4.0 sim) → after (v0.5.0)

| Policy | v0.4.0 | v0.5.0 mean | by tier (easy / medium / hard) | peg seat rate by family |
|---|---|---|---|---|
| zero (no-op) | 0.023 | **0.017** (seeds 0.008–0.032, n=320) | e 0.03 / m 0.01 / h 0.01 | dflat 0.02, ngon 0.01, rect 0.02, round 0.02 |
| baseline (Python, nearest-first) | 0.123 | **0.090** (seeds 0.070–0.105, n=320) | e 0.12 / m 0.10 / h 0.06 | dflat 0.10, ngon 0.05, rect 0.06, round 0.15 |
| baseline.onnx (memoryless twin, seeds 1–3) | n/a | **0.043** (seeds 0.038–0.046, n=192) | e 0.05 / m 0.05 / h 0.04 | dflat 0.02, ngon 0.02, rect 0.01, round 0.11 |
| obs-only scripted (`ObsOnlyScripted`) | 0.124 | **0.083** (seeds 0.066–0.104, n=320) | e 0.12 / m 0.09 / h 0.05 | dflat 0.07, ngon 0.04, rect 0.05, round 0.17 |
| obs-only reference port (`tools/obs_refport.py`) | n/a (new) | **0.210** (seeds 0.191–0.235, n=320) | e 0.32 / m 0.29 / h 0.06 | dflat 0.16, ngon 0.22, rect 0.23, round 0.32 |
| privileged reference (true state) | 0.566 | **0.697** (seeds 0.675–0.719, n=320) | e 0.78 / m 0.70 / h 0.64 | dflat 0.71, ngon 0.71, rect 0.70, round 0.85 |
| top-3 round-5 scripted submissions (private fixtures, seeds 1–3) | 0.79 / 0.39 / 0.22 | **0.019 / 0.014 / 0.010** | | |

v0.4.0 column: v0.3.0/v0.4.0 calibration (same simulator) and the round-5 audit on the v0.4.0 sim
(seeds 501–502). The v0.4.0 reference number predates the reference-solver insertion fix in this
release; the fix alone does not change the task.

Bars for this release (all on the live mix):

| Bar | Target | v0.5.0 | Met |
|---|---|---|---|
| obs-only scripted controllers | ≤ ~0.25 | best 0.210 (`obs_refport`), others ≤ 0.09; top round-5 scripted submissions 0.010–0.019 | yes |
| privileged solver | 0.5–0.75 | 0.697 | yes |
| quick PPO beats best scripted | significant margin | not demonstrated (see below) | **no (open)** |
| eval wall time, 64 episodes, 2 CPU | ≪ 1800 s | 154 s (`tools/local_run.py`, baseline.onnx, `taskset -c 0-1`) | yes |

`obs_refport` is the privileged reference's insertion logic driven only by the
observation (smoothed slot/peg estimates, no family knowledge, no true state). With
zero estimate error it scores ~0.46, so the remaining gap to the reference (0.70) is the
cost of the per-slot/per-peg estimate biases and per-episode layouts, which a policy must
learn to compensate for from contact and motion rather than from a fixed template.

## Reserved split

Throwaway build secret (the real secret is never used here). Privileged reference on the
reserved split, seeds 1–2, 128 episodes: **0.727** (easy 0.87 / medium 0.74 / hard 0.62).
Held-out families are solvable: peg seat rate stadium 0.76, trapezoid 0.74 (live families
0.72–0.92).

## Quick PPO (paired, same 128 episodes, `tools/train_ppo.py`)

| Run | Base | Steps | PPO | Base score | Δ (paired se) |
|---|---|---|---|---|---|
| residual on obs-only scripted | `ObsOnlyScripted` | 300k | 0.091 | 0.086 | +0.004 (0.007) |
| residual on obs-only scripted | `ObsOnlyScripted` | 1M | 0.086 | 0.086 | −0.001 (0.005) |
| residual on obs reference port, privileged shaping (train only) | `obs_refport` | 300k | 0.197 | 0.199 | −0.003 (0.010) |
| insert-phase-only residual, scale 0.5 | `obs_refport` | 400k | 0.179 | 0.199 | −0.021 (0.013) |

None of the quick runs (≤ 1M steps, CPU, a few tens of minutes) produce a significant
gain over its scripted base. This is reported as an **open trade-off**: v0.5.0 removes
the advantage of fixed-template scripting (the scripted ceiling drops from ~0.79 to ~0.21
while the privileged solver stays at ~0.70), but a cheap RL recipe that clears the scripted
ceiling has not been demonstrated yet. There is 0.49 of headroom between the best
observation-only controller and the privileged solver; closing it most likely needs longer
training, longer histories in the policy input, or a learned estimator, not a
change to the task.

## Reproduce

```bash
python3 tools/eval_policies.py --policy reference --seeds 1 2 3 4 5 --per 8 --workers 8
python3 tools/eval_policies.py --policy refport   --seeds 1 2 3 4 5 --per 8 --workers 8
python3 tools/train_ppo.py --base refport --parallel-eval --priv-shaping --steps 300000
NIST_SCRIPTED_FIXTURES=/path/to/private/onnx pytest tests/test_scripted_regression.py
```

`tests/test_scripted_regression.py` asserts every scripted ONNX in
`$NIST_SCRIPTED_FIXTURES` and the in-repo obs-only scripted controller score ≤ 0.30; it is
skipped when the fixture directory is absent (fixtures are not part of the public repo).

