# NIST-Lite v0.5.1 calibration (observation hardening)

Measured on this box (MuJoCo 3.16, CPU) with `tools/eval_policies.py` (same episode draw and
per-episode RNG streams as the referee, dev build). Live mix, 64 episodes per seed
(8 strata × 8), seeds 1–5 unless noted. `OBS_DIM=184`.

## Why v0.5.1

On v0.5.0, hand-coded controllers submitted in live rounds reached 0.89–0.94, above our privileged
solver (0.70). They are compiled rule-based controllers (no learned layers) that servo on the
peg-minus-hole estimate and search the last few millimetres. v0.5.1 makes that residual several
times larger than the clearance, adds drift and glitches, and adds more lying-down starts. The
v0.5.0 column for the revealed submissions is their live/local v0.5.0 score.

## Summary: v0.5.0 → v0.5.1

| Policy | v0.5.0 | v0.5.1 mean | by tier (easy / medium / hard) | peg seat rate by family |
|---|---|---|---|---|
| zero (no-op) | 0.017 | **0.019** (seeds 0.010–0.027, n=320) | e 0.05 / m 0.01 / h 0.01 | dflat 0.02, ngon 0.01, rect 0.03, round 0.02 |
| baseline (Python, nearest-first) | 0.090 | **0.011** (seeds 0.006–0.017, n=320) | e 0.02 / m 0.01 / h 0.01 | dflat 0.00, ngon 0.01, rect 0.01, round 0.02 |
| baseline.onnx (memoryless twin, seeds 1–3) | 0.043 | **0.021** (seeds 0.015–0.031, n=192) | e 0.04 / m 0.01 / h 0.02 | dflat 0.01, ngon 0.01, rect 0.00, round 0.05 |
| obs-only scripted (`ObsOnlyScripted`) | 0.083 | **0.017** (seeds 0.003–0.027, n=320) | e 0.03 / m 0.01 / h 0.01 | dflat 0.01, ngon 0.01, rect 0.01, round 0.04 |
| obs-only reference port (`tools/obs_refport.py`) | 0.210 | **0.027** (seeds 0.021–0.032, n=320) | e 0.07 / m 0.01 / h 0.02 | dflat 0.04, ngon 0.02, rect 0.03, round 0.03 |
| privileged reference (true state) | 0.697 | **0.664** (seeds 0.607–0.715, n=320) | e 0.86 / m 0.64 / h 0.56 | dflat 0.67, ngon 0.73, rect 0.68, round 0.77 |
| revealed v0.5.0 scripted submission: round-14 king (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.257** (seeds 0.238–0.277, n=128) | e 0.47 / m 0.23 / h 0.14 | dflat 0.26, ngon 0.29, rect 0.27, round 0.32 |
| revealed v0.5.0 scripted submission: round-14 entry, same logic (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.261** (seeds 0.238–0.283, n=128) | e 0.46 / m 0.23 / h 0.16 | dflat 0.28, ngon 0.30, rect 0.26, round 0.32 |
| revealed v0.5.0 scripted submission: round-12 entry (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.268** (seeds 0.257–0.280, n=128) | e 0.51 / m 0.21 / h 0.17 | dflat 0.24, ngon 0.31, rect 0.29, round 0.34 |
| revealed v0.5.0 scripted submission: round-8 king (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.153** (seeds 0.134–0.171, n=128) | e 0.34 / m 0.09 / h 0.09 | dflat 0.19, ngon 0.15, rect 0.17, round 0.20 |
| revealed v0.5.0 scripted submission: round-14 entry (separate export) (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.273** (seeds 0.240–0.305, n=128) | e 0.46 / m 0.25 / h 0.17 | dflat 0.29, ngon 0.31, rect 0.29, round 0.31 |
| revealed v0.5.0 scripted submission: round-8 entry (pytorch export) (private fixture, seeds 1–2) | 0.85–0.94 (live) | **0.091** (seeds 0.085–0.098, n=128) | e 0.22 / m 0.06 / h 0.03 | dflat 0.09, ngon 0.12, rect 0.09, round 0.14 |

Bars for this release (live mix):

| Bar | Target | v0.5.1 | Met |
|---|---|---|---|
| revealed v0.5.0 scripted submissions | ≤ ~0.35 | 0.091–0.273 | yes |
| obs-only scripted controllers (ours) | ≤ 0.25 | refport 0.027, others ≤ 0.02 | yes |
| privileged solver | ≥ 0.5 | 0.664 | yes |
| zero / baseline | low | zero 0.019, baseline 0.011, baseline.onnx 0.021 | yes |
| quick PPO beats best scripted | significant margin | not demonstrated (see below) | **no (open)** |
| eval wall time, 128 episodes, 2 CPU | ≤ ~600 s | see HANDOFF (podman smoke) | |

The privileged solver only loses ground on lying-down starts (it does not handle them), so its drop
from 0.70 to 0.66 is the larger lying share, not the observation changes.

## Score spread

Per-episode score SD is larger on v0.5.1 because more episodes land mid-range: about 0.32 for
the strongest scripted fixture (0.14 on v0.5.0), so the 128-episode round mean has a sampling SE of
about 0.028 across rounds. Within a round every submission faces the same episodes, but a near-identical
policy still re-rolls most trajectories (contact-rich sim), so a flat 1% takeover margin is well inside
evaluation noise at these score levels. Measured: the strongest fixture and a copy of it that differs
only by a negligible input offset, on the same 128 episodes, differ on 77 of 128 episodes; the mean
difference is +0.015 (paired SE 0.014, per 64-episode seed +0.042 and −0.013), against a 1% margin of
about 0.003. See the platform discussion on paired, significance-based
takeover; more episodes per stratum (a round input) cut the noise roughly as 1/sqrt(n).

## Quick PPO (paired, same 128 episodes, `tools/train_ppo.py`)

| Run | Base | Steps | PPO | Base score | Δ (paired se) |
|---|---|---|---|---|---|
| residual on obs-only scripted, privileged demos + shaping (train only) | `ObsOnlyScripted` | 300k | 0.024 | 0.016 | +0.008 (0.004) |
| residual on obs reference port, privileged shaping (train only) | `obs_refport` | 300k | 0.025 | 0.024 | +0.001 (0.003) |
| residual on the strongest revealed scripted policy (two settings) | ONNX fixture | 16k–65k | ~0 | 0.33 (32-episode eval) | collapses after the first update |

None of the quick runs (CPU, ≤ 300k steps, tens of minutes) beat its scripted base. The strongest
revealed scripted policies cannot serve as a residual base: they keep their controller state in the
low digits of their own actions and read it back via the `last_action` features, so any residual
breaks them (both residual runs on that base collapsed to ~0 after the first update). The gap
between the best observation-only controllers (≤ 0.27) and the privileged solver (0.66) is the
headroom; closing it most likely needs longer training, a learned estimator, or a privileged
teacher distilled into an observation-only student.

## Reproduce

```bash
python3 tools/eval_policies.py --policy reference --seeds 1 2 3 4 5 --per 8 --workers 8
python3 tools/eval_policies.py --policy refport   --seeds 1 2 3 4 5 --per 8 --workers 8
python3 tools/train_ppo.py --base refport --parallel-eval --priv-shaping --steps 300000
python3 tools/fingerprint.py --probe-seed <private> a.onnx b.onnx
NIST_SCRIPTED_FIXTURES=/path/to/private/onnx pytest tests/test_scripted_regression.py
```

`tests/test_scripted_regression.py` asserts every scripted ONNX in `$NIST_SCRIPTED_FIXTURES` and the
in-repo obs-only scripted controllers score ≤ 0.30 on a fixed 16-episode slice; it is skipped when
the fixture directory is absent (fixtures are not part of the public repo).
