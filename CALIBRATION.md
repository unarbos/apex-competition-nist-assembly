# NIST-Lite v0.8.1 calibration

## Scope

This PR addresses the four scoring/envelope fixes requested in Macrocosmos
[issue #45, comment 7003185456](https://github.com/macrocosm-os/apex-competitions-builder/issues/45#issuecomment-7003185456).
The `v08` profile, 320-float observation, five-float action, 0.36 per-step
partial-credit cap, milestone weights, 920-action horizon, physical perturbation,
and image digest pins are unchanged. `spec.yaml` declares **0.8.1**.

Controller and PPO/SAC tuning are deferred to a later PR. The measured privileged
reference below is the existing **legacy assisted privileged solver**: it reads
true state and may use simulator-state assists. It is not an action-only
controller result and is not a gate for this scoring-fix PR.

## Four fixes

1. **Completed keys.** A completed key retains credit from its physical
   completion latch and completion-time precision while its peg remains seated
   upright. Final wrist yaw is not re-checked against every earlier key window.
   Missing completion precision fails closed. Tests cover both two-key and
   three-key graphs, a wrist moved to the last key's window, final perturbation,
   and a key whose peg is pulled out losing credit.
2. **Dependent partials.** Unfinished key, twist, and screw steps cannot earn
   generic insert-part lift/transport/insert/alignment/stability partial credit
   by re-grasping or lifting an already-inserted peg. Evidence collection skips
   those steps and the scoring function also rejects stale generic evidence.
   The regression inserts the peg, re-lifts it, and requires zero dependent
   partial credit for key, twist, and screw.
3. **Result envelope.** The top-level metadata keys are exactly
   `schema_version`, `summary`, `units`, `metrics`, `capabilities`, and
   `details`. `nist_profile` remains in `details` and per-episode history
   records, not at the metadata root or in result units. Assembly IV is labeled
   `IV`, not `II`, in details and records. The serialized `local_run.py`
   integration test checks this contract.
4. **Collar/stack finalization.** `collar_run` is refreshed during settling.
   Final collar credit requires current physical seating rather than a stale
   counter. Final stack credit checks the measured ring pose; finalization does
   not re-pin a mated ring after the kick. Regression tests use the legacy
   privileged reference to earn collar/stack completion, then verify stable and
   displaced physical fixtures for both kinds. Final peg seating requires both
   depth and upright orientation. The bounded diagnostic privileged search
   completed a collar but no stack; stack stable/displaced cases are explicitly
   labeled physical fixtures, not a claimed privileged stack demonstration.

Non-round key-window sampling is intentionally unchanged in this scoped PR:
symmetry-midpoint windows remain predictable and are not a blind search. This is
an open design choice, not a new randomization change.

## Validation

Test and evaluation artifacts remain outside the repository under
`/home/bb/nist-v080-artifacts/`. Full-suite and matched results will be inserted
from completed runs on this exact scorer, not copied from earlier WIP reports.

| Check | Result |
|---|---|
| Full test suite | Pending on the scoped PR branch |
| Four-fix focused regressions | **7 passed, 32 deselected in 95.68 s** |
| Built-image 64-episode / two-CPU wall | CI `referee-image` job on the PR; no local container claim |

## Matched fixed-scorer results

Seeds 7, 19, and 31; 64 episodes per seed (192 per policy). Confidence intervals
are episode-bootstrap 95% intervals, 10,000 resamples, seed 20261117. Every
report records the simulator-source bundle hash, `per=8`, model/controller hash
where applicable, and contiguous episode indices.

| Policy | Seed 7 | Seed 19 | Seed 31 | Pooled mean [95% CI] | Round sample std |
|---|---:|---:|---:|---:|---:|
| Zero | Pending | Pending | Pending | Pending | Pending |
| Uniform random | Pending | Pending | Pending | Pending | Pending |
| Best observation-only script | Pending | Pending | Pending | Pending | Pending |
| CEM, selected theta `[-0.002, 0.028, 0.105]` | Pending | Pending | Pending | Pending | Pending |
| Legacy assisted privileged reference | Pending | Pending | Pending | Pending | Pending |

The required zero and uniform-random scores must be **exactly zero**. Any
nonzero random result will be investigated rather than silently accepted.

## Determinism

Independent matched repeats will compare complete report bytes. Pending.

## Historical results

Earlier v0.8.0 and controller-tuning measurements are not evidence for this PR.
In particular, prior stack-finalization comparisons that re-pinned rings during
settling are not reused. Generated models, demonstrations, and measurement logs
are not committed. No miner ONNX, hotkeys, secrets, merge, tag, release, image
push, or digest change is included.
