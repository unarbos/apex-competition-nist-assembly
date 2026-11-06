# NIST-Lite calibration

## v0.7.1 (Assembly III fixes, profile `v07`, live default)

Live referee default is `v07` (spec `0.7.1`). Image bake is `NIST_PROFILE=v07`. Hardness is in
**obs-indexed keys**, **rework extract**, **stack rings**, denser graphs, and residual noise.
Referee bounds cost with sim horizon only (`MAX_HORIZON=1400`, `deadline_ms=80`); **no**
`round_wall_s` or per-episode wall clock. Platform 1800 s is a non-scoring abort only.

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold`). Public seeds only. Image bake verified via
`tools/image_calib_v07.py` (zero ONNX through built referee+player; `nist_profile=v07` in
`result.json` / records).

Reproduce:

```bash
NIST_PROFILE=v07 python tools/calib_v06.py --profile v07 --seeds 7 --per 8 --workers 4 \
  --policies reference v06_style compound_scripted zero --ref reference \
  --live-eps 64 --cpus 2 --out out/calib_v071
# or: NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
# through images: python tools/image_calib_v07.py --seed 7 --full
```

| Policy | mean (seed 7, 8/stratum, 64 eps) | wall_s | notes |
|---|---|---|---|
| privileged reference (honest) | **0.503** | ~702 | true-state; normal predicates |
| v0.6-style scripted | **0.273** | (same harness) | insert/collar/twist only |
| obs-only compound_scripted | **0.033** | (same harness) | in-process |
| zero (built images) | **0.006** | **144** | `nist_profile=v07` in result/records |

64-ep zero through referee+player images: **144 s ≪ 1800 s**. Platform timeout is a non-scoring abort only.

## v0.7.0 (superseded — oracle-inflated privileged bar)

The v0.7.0 privileged row used oracle helpers in `tools/compound_solver.py` and is **not** a
valid solvability proof. Image also baked `NIST_PROFILE=v06` and cut rounds on wall clocks.
See v0.7.1.

## v0.6.2 (mass-event fix)

A `mass` event now refreshes MuJoCo's derived mass constants (`mj_setConst`), so the next grasp
no longer oscillates and ejects the peg (reported by msulab). Same harness and episodes as v0.6.1
(`tools/calib_v06.py --profile v06`, seeds 1–2, 4/stratum, 64 paired episodes):

| Policy | v0.6.1 | v0.6.2 |
|---|---|---|
| zero | 0.010 | **0.010** |
| obs-only scripted | 0.015 | **0.015** |
| baseline ONNX | 0.021 | **0.021** |
| privileged compound solver | 0.388 | **0.403** (c1/c2/c3/c4 0.58 / 0.47 / 0.31 / 0.26) |

Episodes with a scheduled mass event (dev seeds 3210–3213, 8/stratum: 91 of 256 episodes, 47 where
the event fired), privileged solver: `unstable` endings **3 → 0**; mean score 0.275 → 0.272
(within noise; the solver re-grasps carefully, so the effect is larger for policies that
grasp hard right after a change). Policies that rarely finish a step almost never trigger an
event and are unchanged. Projected live eval time 259 s for the slowest policy (≤ 400 s bar).

## v0.6.1 (pre-launch Macrocosmos fixes)

Same harness as v0.6.0 (`tools/calib_v06.py --profile v06`, seeds 1–2, 4/stratum, 64 paired
episodes). Rubric: out-of-order credit floored at the partial cap (0.30); collar 0.20/0.30
partial requires the peg in its hole. Scale is essentially unchanged vs v0.6.0:

| Policy | v0.6.0 | v0.6.1 | notes |
|---|---|---|---|
| zero | 0.008 | **0.010** | |
| obs-only scripted | 0.018 | **0.015** | |
| baseline ONNX | (not in that table) | **0.021** | |
| privileged compound solver | 0.406 | **0.388** | c1/c2/c3/c4: 0.58 / 0.42 / 0.32 / 0.24 |

**Eval wall time (this box, 2 workers):** 64 episodes, baseline ONNX, seed 1: **676 s**
(fits the 1800 s timeout with margin). Privileged mean ~22 s/episode in the calib harness.

**Lying-peg ceiling (stated for launch).** Privileged solver seats lying inserts **0/64**
(seeds 11–13, 96 episodes). Lying pegs are ~13% of insert steps and also delay the waves
behind them. Treat lying inserts and the collar/twist steps that depend on them as
**unproven headroom**, not demonstrated ceiling. Reach-and-grasp works
(`tests/test_lying_grasp.py`); re-orient-and-seat on the multi-peg board does not, for any
controller we have.

## v0.6.0 (compound tasks)

Measured with `tools/calib_v06.py --profile v06` (same episode draw, per-episode RNG streams,
`CompoundEpisode` and `compound_score` as the referee; dev build), live mix, seeds 1–2,
4 episodes per stratum: 64 paired episodes. Physics at 500 Hz, action rounding off.

| Policy | Overall | c1 / c2 / c3 / c4 | steps done | credit by kind (collar / insert / twist) |
|---|---|---|---|---|
| zero (no-op) | **0.008** | 0.01 / 0.01 / 0.01 / 0.01 | 0.01 | 0 / 0.03 / 0 |
| obs-only scripted (`ObsOnlyScripted`, 184-d prefix) | **0.018** | 0.03 / 0.01 / 0.01 / 0.01 | 0.03 | 0 / 0.06 / 0 |
| revealed v0.5.0 scripted submission: round-15 king (private fixture, 184-d prefix) | **0.202** | 0.30 / 0.26 / 0.12 / 0.13 | 0.49 | 0 / 0.66 / 0 |
| revealed v0.5.0 scripted submission: round-14 entry (private fixture, 184-d prefix) | **0.190** | 0.29 / 0.24 / 0.12 / 0.11 | 0.47 | 0 / 0.62 / 0 |
| naive BC of the privileged solver (128 demos, 3×256 MLP, memoryless, 288-d) | **0.006** | 0.01 / 0.01 / 0.00 / 0.00 | 0.00 | 0 / 0.02 / 0 |
| privileged compound solver (`tools/compound_solver.py`, true state) | **0.406** | 0.60 / 0.52 / 0.26 / 0.24 | 0.49 | 0.34 / 0.55 / 0.34 |

The revealed submissions vs the privileged solver: paired Δ −0.20 to −0.22 (SE 0.03, z ≈ −7). No
simulation blow-ups in 320 episodes. Revealed v0.5 controllers still insert pegs well. They earn
nothing on collars and twists, which carry most of the weight.

Bars for this release:

| Bar | Target | v0.6.0 | Met |
|---|---|---|---|
| privileged solver overall (solvability) | ≥ 0.30 | 0.406 | yes |
| privileged solver on c4 (ceiling headroom) | ≤ 0.70 | 0.24 | yes |
| best fixed scripted controller | ≤ 0.5 × privileged (0.203) | 0.202 | yes (just) |
| zero / obs-only scripted | low | 0.008 / 0.018 | yes |
| learned policy beats best scripted (paired z ≥ 3) | | naive BC 0.006; DAgger / PPO not yet run | **no (open)** |
| eval wall time | fits 1800 s with margin on 2 CPU | see HANDOFF §4 | |

**Lying-peg ceiling.** Over 96 episodes (seeds 11–13), the privileged solver completed these
inserts by start type: grasped 61/69 (0.88), tray 11/16 (0.69), near-hole 147/231 (0.64), loose
77/124 (0.62), lying **0/64**. Lying pegs can be reached and grasped
(`tests/test_lying_grasp.py`), but no controller of ours re-orients and seats one on the
multi-peg board. Lying pegs are about 13% of insert steps. Their inserts, and the collar/twist
steps on the same pegs, should be read as unproven headroom. They are not a demonstrated part of
the ceiling.

**Glitch-burst share.** Estimate glitch bursts (8–30 steps, 4× noise plus a 2–6 mm offset) run at
half the v0.5.1 rate on v0.6. On the live mix they cover **25%** of control steps: easy 16%,
medium 25%, hard 29% (seeds 1–3, 12 per stratum, burst process only).

**Cost.** A v0.6 episode averages ~1,810 control steps (v0.5.1 ~1,260). Per step, 500 Hz physics
with 8-segment collars costs about the same as v0.5.1 at 1 kHz (1.66 vs 1.76 ms idle on this box).
So a v0.6 episode costs about 1.4× a v0.5.1 episode. See HANDOFF §4 for the round size.

Reproduce:

```bash
NIST_PROFILE=v06 python tools/calib_v06.py --profile v06 --seeds 1 2 --per 4 --workers 8 \
  --policies reference zero obs_scripted --ref reference --out out/calib_v06
```

## v0.5.1 calibration (observation hardening, kept for history)

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
