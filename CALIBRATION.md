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
controller result and is not a gate for this scoring-fix PR. Across the 192
matched episodes it completes 3 stack steps, 11 collars, and 27 keys.

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
   not re-pin a mated ring after the kick. An earned interference-fit stack
   ring is exempt from the direct collar impulse; the peg is still kicked and
   the ring's measured pose is verified after settling. Unmated loose rings
   still receive the impulse. Default-kick tests verify that a mated stable
   ring keeps credit, a displaced ring loses credit, and an unmated ring is
   still kicked. Final peg seating requires both depth and upright orientation.
   Stack fixtures are explicitly physical finalization fixtures, not a claimed
   privileged stack demonstration.

Non-round key-window sampling is intentionally unchanged in this scoped PR:
symmetry-midpoint windows remain predictable and are not a blind search. This is
an open design choice, not a new randomization change.

## Validation

Test and evaluation artifacts remain outside the repository under
`/home/bb/nist-v080-artifacts/`. The matched reports use simulator-source bundle
SHA-256 `4b1b96662b08c7903da4468f5ace815716d167f13360e57a244c0a9071ad79ec`.

| Check | Result |
|---|---|
| Full test suite, final scoring-fix head `2e8a4b3` | **202 passed, 7 skipped, 4 warnings in 848.26 s** (CI run 44537159324) and **202 passed, 7 skipped, 4 warnings in 1161.30 s** (CI run 44537171527); Stable-Baselines3 is absent |
| Four-fix focused regressions | **7 passed, 1 skipped, 31 deselected in 31.66 s** in the CI-only venv (no Stable-Baselines3) |
| Built-image 64-episode / two-CPU wall, final scoring-fix head | **293.619 s** (CI run 44537159324) and **223.005 s** (CI run 44537171527), cpuset 0,1; no player fault; both pass 600 s |

Per-round wall time is not aggregated across resumable measurement chunks. The
built-image CI runs above are the wall-time evidence; they are separate from the
in-process policy comparison below.

## Matched fixed-scorer results

Seeds 7, 19, and 31; 64 episodes per seed (192 per policy). Confidence intervals
are episode-bootstrap 95% intervals, 10,000 resamples, seed 20261117. Every
report records the simulator-source bundle hash, `per=8`, model/controller hash
where applicable, and contiguous episode indices.

| Policy | Seed 7 | Seed 19 | Seed 31 | Pooled mean [95% CI] | Round sample std |
|---|---:|---:|---:|---:|---:|
| Zero | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 [0.000000000000, 0.000000000000] | 0.000000000000 |
| Uniform random | 0.000000000000 | 0.000000000000 | 0.000001140097 | 0.000000380032 [0.000000000000, 0.000001140097] | 0.000000658236 |
| Best observation-only script (`script`) | 0.011770318832 | 0.009212072863 | 0.005786258549 | 0.008922883415 [0.007189550607, 0.010763208236] | 0.003002493509 |
| CEM, selected theta `[-0.002, 0.028, 0.105]` | 0.013930091274 | 0.015594990778 | 0.009920159394 | 0.013148413815 [0.011304683038, 0.015014859784] | 0.002917052030 |
| Legacy assisted privileged reference | 0.132743658480 | 0.142243356531 | 0.152491145668 | 0.142492720227 [0.127751994114, 0.157873185200] | 0.009876104964 |

Zero is exactly zero in all 192 episodes. **The strict uniform-random exact-zero gate is not met:** seed 31, episode 11 scores 0.000072966237. The exact evaluator call reproduces it. The only nonzero evidence is `lift_fraction=0.007296623697` on an unfinished rework step: the closed gripper contacts the peg while it rises about 0.772 mm over two frames, above the existing 0.5 mm fractional-lift threshold. This is under the current lift rule, not dependent-step or completion credit. The score is retained rather than hidden by rounding or a scoring change made to improve the table.

CEM minus the best observation-only script is **0.004225530401 [0.002339346329, 0.006145725924]** (paired episode bootstrap; interval excludes zero). The alternative `strong_script` pools to **0.008413835474 [0.006707563416, 0.010208918610]**. Ordinary-minus-strong is 0.000509047941 [-0.001760417831, 0.002759936812], which includes zero.

## Determinism

All five reported policies were independently rerun on all three seeds: **960 repeat episodes**.
Every per-seed canonical report and every complete three-seed report is **byte-identical**
to its original, including scores, terminal reasons, completed kinds, and milestone fields.
Timing is null in both reports; wall time is measured separately by the built-image CI jobs.

| Policy | Complete repeated report SHA-256 |
|---|---|
| `zero` | `be415cab635680a6ce34d1b857858f3b92aca437f6cbe64c6265056ff6a655e8` |
| `random` | `61ab6f272e7dfbd888c7cbca939f336cc5f415b7ed0e3c08ef4dc31aba759a0f` |
| `script` | `e90a224263624efd1545cf75a32a18f576c9eef7f7aa8d278914eddf4543efa9` |
| `cem` | `5915c6f5f20f7cb661713b5908403cb970c4082490a872e0f0e1263895adaf4c` |
| `privileged` | `cb7b13bfb30b4917aad57252eb6162ed8d922fe351b8a3fe36b7360355e4dddf` |

## Historical results

Earlier v0.8.0 and controller-tuning measurements are not evidence for this PR.
In particular, prior stack-finalization comparisons that re-pinned rings during
settling are not reused. Generated models, demonstrations, and measurement logs
are not committed. No miner ONNX, hotkeys, secrets, merge, tag, release, image
push, or digest change is included.
