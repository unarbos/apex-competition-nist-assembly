# NIST-Lite v0.8.2 calibration

## Scope

This PR fixes stack finalization reported in Macrocosmos
[issue #49](https://github.com/macrocosm-os/apex-competitions-builder/issues/49).
A ring mated in play and left undisturbed retains credit. A ring that was never
mated, was knocked off in play, was displaced, or is gripped scores zero.
`spec.yaml` declares **0.8.2**.

The `v08` profile, 320-float observation, five-float action, 0.36 per-step
partial-credit cap, milestone weights, 920-action horizon, physical task rules,
and image digest pins are unchanged. Controller and PPO/SAC tuning are outside
this patch. The assisted reference below is the existing **legacy assisted
privileged solver**: it reads true state and may use simulator-state assists.
It is not an action-only controller result or a release gate and was not tuned
to improve its score.

## Fix

1. Snapshot completion claims using the physical predicates before finalization.
   A passive end pose cannot synthesize an unearned completion.
2. For a claimed, physically valid stack, skip the carrier peg's lateral impulse
   so the calibration impulse does not throw its supported ring off. Draw the
   kick angle in the original order. Unrelated seated pegs retain their kicks,
   and never-mated rings retain their direct impulses.
3. Place stack rings with their octagonal faces supporting the peg at its
   measured slot yaw. Rectangular and regular-polygon cross-sections use their
   shared symmetry to avoid unsupported face alignments.
4. Continue measuring peg seating, ring position, ring height, upright pose, and
   gripper clearance after settling. Never re-pin a mated ring during settling.
   Invalid claims do not exempt the carrier peg and score zero.

The board-42 investigation also exposed a reset-validation defect: rework
contacts bypassed unrelated clutter/contact validation, allowing deeply embedded
clutter to enter settling and become a projectile. Validation now limits the
contact exemption to the intentional wrong-slot peg/hole pair, checks unrelated
parts and active rings, and validates the collision-free fallback start before
returning an episode. Spawn bounds, penetration, drift, and speed limits are not
relaxed. Existing scoring tolerances and task difficulty are unchanged.

## Independent v0.8.1 reproduction

The public issue reproduction was run locally and adapted into regression tests;
the original script is not committed. Published cases on v0.8.1 were reproduced:

| Published case | Measured retention |
|---|---:|
| Board 1 step 9, round, yaw 0, with kick | 8/8 |
| Board 1 step 9, round, slot yaw, with kick | 8/8 |
| Board 62 step 15, rectangular, slot yaw, with kick | 0/8 |
| Board 62 step 15, rectangular, yaw 0, with kick | 8/8, but 1.38 mm hole-wall overlap |
| Board 35 step 10, low-friction round, slot yaw, with kick | 0/8 |

The independent all-69-unparked-stack survey used measured slot yaw and four kick
directions. Each final-source row has 276 trials; the separate committed test
requires all 69 no-kick cases to keep credit and at least 95% in every kicked
shape bucket.

| Shape | Stacks | Before, no kick | Before, with kick | After, with kick |
|---|---:|---:|---:|---:|
| D-flat | 14 | 13/14 | 0/56 | **56/56 (100%)** |
| n-gon | 18 | 18/18 | 1/72 | **72/72 (100%)** |
| rect | 12 | 9/12 | 3/48 | **48/48 (100%)** |
| round | 25 | 24/25 | 66/100 | **100/100 (100%)** |
| **Total** | **69** | **64/69** | **70/276 (25.36%)** | **276/276 (100%)** |

The final external no-kick scan independently retained all **276/276** trials.
The adapted committed regression uses one no-kick fixture per stack plus four
kick-seed trials per stack, verifies all 69 no-kick cases and every 95% shape
bucket, explicitly covers boards 38/52/54 and board 42 without a kick, and
rejects re-pinning. Valid-stack RNG/impulse checks and never-mated, displaced,
and gripped zero-score fixtures are also included.

## Validation

Test and evaluation artifacts remain outside the repository under
`/home/bb/nist-v080-artifacts/`. The final matched reports use simulator-source
bundle SHA-256
`012ad46950fc9609e7c0c31fb03ff2b50d53468ceade3701db8a9f800080172d`.

| Check | Result |
|---|---|
| Final-source focused stack/reset regressions | **9 passed, 43 deselected in 99.50 s** |
| Full test suite, exact CI environment | **218 passed, 5 skipped in 995.02 s** (`NIST_PROFILE=v08`); GitHub Actions pending |
| Supplemental no-profile full-suite runs | 217 passed, 5 skipped, 1 failed in legacy seed-1/episode-55 no-op stability in each of two runs; isolated patched and `origin/main` scans passed. This is not used as release-gate evidence. |
| All 64 sampled boards × finalization RNGs 23–26 | **0 construction/validation failures in 256 resets** |
| Built-image 64-episode / two-CPU wall | Pending CI; must be below 600 s |
| Digest pins | Unchanged from v0.8.1 |

<!-- FINAL_VALIDATION_TABLE: insert local/CI results and exact run IDs. -->

Per-round wall time is not aggregated across resumable measurement chunks.
Built-image CI is the wall-time evidence; in-process policy runs measure scores
and determinism separately.

## Assisted-reference stack retention

Each seed contains 64 episodes. Eligibility means a physically valid stack that
was completed in play before finalization; these are not passive pose fixtures.

| Seed | Before: eligible | Before: kept | After: eligible | After: kept |
|---|---:|---:|---:|---:|
| 7 | 15 | 0 | 16 | 16 |
| 19 | 23 | 3 | 22 | 22 |
| 31 | 24 | 0 | 22 | 22 |
| **Total** | **62** | **3 (4.84%)** | **60** | **60 (100%)** |

| Shape | Before: eligible | Before: kept | After: eligible | After: kept |
|---|---:|---:|---:|---:|
| D-flat | 7 | 0 | 5 | 5 |
| n-gon | 20 | 1 | 21 | 21 |
| rect | 9 | 0 | 7 | 7 |
| round | 26 | 2 | 27 | 27 |
| **Total** | **62** | **3** | **60** | **60** |

The before/after eligible counts are reported independently. Reset correction
can change the assisted solver's subsequent trajectory; denominators are not
assumed identical across scorer versions. All three final-source rounds measured
**0** initially valid gripped stacks and **0** initially completed but physically
invalid stacks.

## Matched fixed-scorer results

Zero and uniform random use seeds 7, 19, 31, 43, and 59; all other policies use
seeds 7, 19, and 31. Every round has 64 episodes. Confidence intervals are
episode-bootstrap 95% intervals, 10,000 resamples, seed 20261117. Every report
records the simulator-source bundle hash, `per=8`, model/controller hash where
applicable, and contiguous episode indices. CEM uses selected theta
`[-0.002, 0.028, 0.105]`, model SHA-256
`6497939c270b733731851024dd4375d1421a438efc7f29420ebb09121741f4f2`.

| Policy | Seed 7 | Seed 19 | Seed 31 | Pooled mean [95% CI] | Round sample std |
|---|---:|---:|---:|---:|---:|
| Zero | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 [0.000000000000, 0.000000000000] | 0.000000000000 |
| Uniform random | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 [0.000000000000, 0.000000000000] | 0.000000000000 |
| Strong observation-only script | 0.007480294772 | 0.008054087099 | 0.008400895447 | 0.007978425772 [0.006322663418, 0.009737406456] | 0.000464940725 |
| CEM | 0.013378643432 | 0.015659481659 | 0.010350975092 | 0.013129700061 [0.011209426954, 0.015060606764] | 0.002662994573 |
| Legacy assisted privileged reference | 0.149044175561 | 0.162007288183 | 0.163256833316 | 0.158102765687 [0.141916240736, 0.174857568180] | 0.007869808260 |

Zero and random also scored exactly zero on seeds 43 and 59, so each completed
five-round report contains 320 exact-zero episodes. CEM minus strong script is
**0.005151274289** with paired episode-bootstrap 95% CI
**[0.003079704190, 0.007145657930]** over 192 matched episodes. The assisted
reference completed 60 stack steps, all retained, as reported independently
above.

## Red-team pass

Generic attack rollouts use seed 31 and eight episodes per attack; each listed
rollout scored exactly zero. The red-team summary and physical fixtures remain
in the external artifact directory. The committed regressions additionally
require zero target credit for never-mated, displaced, and gripped stack rings.

| Attack class | Measured result |
|---|---|
| Contact jitter / push without grasp | `jitter`, `vibrate_parts`, `slam_down`, and `transport_wrist_spoof`: all zero. |
| Wrist-yaw sweeps / key precision farming | `yaw_spin` and `cue_hillclimb_sweep`: all zero. Natural-seat key freebie completed **0/20** keys. |
| Dependent-step re-grasp / re-lift | Key and screw target credit is exactly zero, including with injected stale insertion evidence. The separate prerequisite insert credit of 0.002727272727 in each fixture is not dependent-step credit. No unparked twist step exists in that fixture; the screw remains explicit. |
| False / barely mated and knocked-off stack rings | Post-mate knock-off, 12 mm lateral false mate, and -20 mm vertical false mate: zero with final physical rejection. A +12 mm vertical displaced pose fails immediately but can settle back onto the peg; its fixture score remains zero because prerequisites were not completed. |
| Invalid/extreme actions and observation spoofing | NaN, extreme-action, sensor-spoof, early-termination, solver-blowup, and drop-near-hole rollouts: all zero. Invalid or unstable episodes terminate without earning target credit. |
| Gripped / displaced / never-mated rings | Committed physical finalization regressions: zero target score; invalid claims cannot exempt the carrier peg, which still receives its kick. |

The analytic single-kind-only bound across the tested blueprints is
**0.427639751553**. This is a scoring bound, not an achieved attack score.

## Determinism

Zero, uniform random, strong observation-only script, CEM, and the assisted
privileged reference are independently repeated on seeds 7, 19, and 31:
**960 repeat episodes**. Every policy passed exact episode-record comparison and
byte-identical complete canonical reports on the frozen scorer.

| Policy | Canonical report SHA-256 | Repeat episodes |
|---|---|---:|
| Zero | `63232e8c02178020c3ab942ef344999639d2e69b834781a72d80ac3bd7620e0b` | 192 |
| Uniform random | `dd1deb327071f3dc1b784a72a2914c9dcab2e5795b81ee9213713a4007833403` | 192 |
| Strong observation-only script | `86019cfe333e98514274853b168fcb6c8995b508cbcbfbc2ca17940860b546e3` | 192 |
| CEM | `5cd2ee6223ac71f08a0237b0d9a928cee742d68d6fd84722c1c71d420d56030b` | 192 |
| Legacy assisted privileged reference | `20f071746222237e0b7c7e99ee37e068d042b22c0fb157ba0c1369cdb33f07bd` | 192 |

## Historical results

Earlier v0.8.0/v0.8.1 and controller-tuning measurements are not evidence for
this patch. In particular, v0.8.1 policy scores, CI walls, and red-team results
are not carried forward as v0.8.2 evidence. Generated models, demonstrations,
and measurement logs are not committed. No miner ONNX, hotkeys, secrets, merge,
tag, release, image push, or digest change is included.
