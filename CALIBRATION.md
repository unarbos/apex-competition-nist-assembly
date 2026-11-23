# NIST-Lite v0.8.3 calibration

## v0.8.3 reset-failure guard

This PR addresses the reset-failure rollout gate in Macrocosmos
[issue #45, comment 7084678153](https://github.com/macrocosm-os/apex-competitions-builder/issues/45#issuecomment-7084678153).
`spec.yaml` declares **0.8.3**. It contains construction exceptions inside the
referee worker, keeps the affected episode as a deterministic zero in the fixed
scoring denominator, and continues the remaining episodes. Records identify the
episode with `terminal_reason` and `failure_reason` equal to `reset_failure`.
Workers report contained reset failures before stepping, so a later worker death
cannot replace them with a generic worker failure. Exception text and private
simulator inputs are not recorded.

The parent closes child pipe endpoints after forking and polls readability and
process liveness around every receive. A one-byte readiness channel confirms
that each child has closed inherited parent endpoints before the next fork,
without waiting for its episode construction to finish. The parent waits on
pending startup pipes rather than serializing behind one child's construction,
so an early-dead peer can still be collected. EOF or a dead child produces
complete `worker_failure` zero records for that child's episodes instead of
waiting for the platform timeout. A failed readiness launch explicitly closes
its current pipe and terminates and reaps every started worker. Summary validation also rejects non-finite or oversized
numeric values and non-JSON-serializable public record/trajectory payloads before
aggregation or record writing. Healthy episode construction and simulation have
no additional wall-clock cutoff. Only a partial live frame or a backpressured
parent-to-worker frame with no transport progress for five seconds is contained
as a worker failure. Reset-only workers are excluded from action and `fail_all`
delivery but still receive their final summary request.

A claimed stack's carrier peg is exempt from its finalization impulse only when
`seated_now` is true, not merely `_peg_in_hole`. Kick-angle draws are unchanged.
Ordinary, non-stack collars continue to receive 0.63 m/s kicks. The wrist-sweep
and key-precision observations in the issue comment remain design notes, not
changes in this patch.

The `v08` profile, 320-float observation, five-float action, 0.36 per-step
partial-credit cap, milestone weights, 920-action horizon, physical task rules,
and image digest pins are unchanged. Controller and PPO/SAC tuning are outside
this patch.

### Released-image reset sweep

The sweep constructs all 64 Assembly IV episodes for each of 100 independently
derived live-split round seeds using the released v0.8.2 referee application
layers and isolated live build inputs. Its report contains anonymous sample
indices only; no held-out round seed, range, or build salt is printed or stored.
Each construction uses the normal round and per-episode seed derivation and an
independent condition copy. Instrumentation counts calls to `_force_safe_start`
and `_teleport_safe_grid`, plus any exception raised during construction.

Final result: **100 samples × 64 episodes = 6,400 constructions**. Across those
episodes, **45 (0.703125%) reached `_force_safe_start`**, **0 reached
`_teleport_safe_grid`**, and **0 raised during reset/construction**. The sweep
completed in 326.29 s with two workers. Application-layer digests were verified
during retrieval, and the released public `nist_sim` source matched tag v0.8.2;
the simulator-source file hashes are retained in the external sweep report.

Each entry below is `anonymous sample: force-safe / safe-grid / raised`. No entry
contains a round seed or any private build input.

```text
0: 0/0/0 | 1: 0/0/0 | 2: 0/0/0 | 3: 0/0/0 | 4: 0/0/0
5: 0/0/0 | 6: 0/0/0 | 7: 0/0/0 | 8: 1/0/0 | 9: 0/0/0
10: 0/0/0 | 11: 0/0/0 | 12: 0/0/0 | 13: 0/0/0 | 14: 0/0/0
15: 0/0/0 | 16: 0/0/0 | 17: 2/0/0 | 18: 0/0/0 | 19: 1/0/0
20: 0/0/0 | 21: 0/0/0 | 22: 0/0/0 | 23: 0/0/0 | 24: 0/0/0
25: 0/0/0 | 26: 1/0/0 | 27: 2/0/0 | 28: 0/0/0 | 29: 0/0/0
30: 0/0/0 | 31: 0/0/0 | 32: 0/0/0 | 33: 2/0/0 | 34: 0/0/0
35: 2/0/0 | 36: 0/0/0 | 37: 1/0/0 | 38: 1/0/0 | 39: 0/0/0
40: 0/0/0 | 41: 1/0/0 | 42: 0/0/0 | 43: 0/0/0 | 44: 0/0/0
45: 0/0/0 | 46: 1/0/0 | 47: 1/0/0 | 48: 2/0/0 | 49: 0/0/0
50: 1/0/0 | 51: 0/0/0 | 52: 0/0/0 | 53: 0/0/0 | 54: 0/0/0
55: 0/0/0 | 56: 0/0/0 | 57: 0/0/0 | 58: 0/0/0 | 59: 1/0/0
60: 2/0/0 | 61: 0/0/0 | 62: 0/0/0 | 63: 1/0/0 | 64: 1/0/0
65: 0/0/0 | 66: 2/0/0 | 67: 0/0/0 | 68: 0/0/0 | 69: 0/0/0
70: 1/0/0 | 71: 0/0/0 | 72: 0/0/0 | 73: 1/0/0 | 74: 1/0/0
75: 0/0/0 | 76: 0/0/0 | 77: 0/0/0 | 78: 1/0/0 | 79: 0/0/0
80: 0/0/0 | 81: 1/0/0 | 82: 0/0/0 | 83: 0/0/0 | 84: 1/0/0
85: 0/0/0 | 86: 1/0/0 | 87: 0/0/0 | 88: 0/0/0 | 89: 1/0/0
90: 0/0/0 | 91: 0/0/0 | 92: 3/0/0 | 93: 0/0/0 | 94: 1/0/0
95: 3/0/0 | 96: 0/0/0 | 97: 1/0/0 | 98: 1/0/0 | 99: 2/0/0
```

### Ordinary-collar retention survey

The independent v0.8.3 survey exercised ordinary, non-stack collars that are
active at reset, using physically valid collar placements and four kick seeds
(23–26) per collar. Parked collars are excluded; they do not receive an impulse.
All **72/72 observed impulses were 0.63 m/s**, and **60/72 (83.33%)** retained
physically verified collar credit after settling. These fixtures verify the
ordinary-collar kick and report its retention; they do not modify the physical
rules or relax retention requirements.

### v0.8.3 validation

Guard tests use the actual fork/pipe implementation and an eight-episode,
two-worker round. They force selected constructions to fail, repeat the same
round for exact record/metadata equality, and terminate one child before its
first observation. Additional cases inject truncated and malformed pipe frames
and terminate a worker after it accepts an action, checking that surviving
workers do not consume stale acknowledgements or silently skip their next action.
Additional cases use three workers to kill the middle worker after an earlier
worker has acknowledged, kill a worker during observation collection in a
three-step round, return a decoded malformed observation, and return incomplete
summary rows. The normal action phase stores one action batch and completes every
surviving worker's unacknowledged delivery before requesting another observation.
Assertions cover deterministic zero scores, reason flags, fixed episode count
and denominator, normal per-kind schema keys, public-record privacy,
player-fault absence, complete output, bounded completion, and pipe/liveness-
based dead-worker recovery. The carrier regression requires a non-upright,
in-hole carrier to receive its kick while all random-angle draws remain
unchanged. A normal two-worker regression independently requires all eight
episodes to receive all 920 actions, protecting the non-failure lockstep path.

The final guard selection contains **26 cases**, including early-death during a
peer's construction, readiness-failure cleanup, oversized-number,
non-JSON-summary, nonzero-reset-summary, and reset-then-worker-death
regressions. Both partial-write and initially fully backpressured sends are
covered. Final-source focused results are **26 passed in 123.81 s** for the
guard module, including per-test `v08` profile restoration after legacy module
reloads, and **53 passed, 3 warnings in 150.28 s** for Assembly IV. The clean
ordered repository-wide suite passed: **245 passed, 5 skipped, 4 warnings in
1209.55 s (0:20:09)**, with `NIST_PROFILE=v08` and single-threaded BLAS/OpenMP.
Stopped or pre-fix runs are not used as final evidence.
GitHub Actions built-image wall is pending at PR creation.

Final-source ONNX-player/referee checks used public development seed 7 with
eight episodes and two workers, running zero and uniform random sequentially.
Both returned code 0, completed all eight episodes at 920 steps, made 920 player
calls, and had no player fault. **Zero scored exactly 0.0 in 30.557978339 s;
uniform random scored exactly 0.0 in 30.786305332 s.** Both scores are unchanged
from the v0.8.2 zero/random baseline. These local runs measure score behavior,
not the required 64-episode built-image wall.

The v0.8.2 measurements retained in the following sections are historical
baseline evidence, not reattributed to v0.8.3.


## v0.8.2 baseline: stack finalization

The previous PR fixed stack finalization reported in Macrocosmos
[issue #49](https://github.com/macrocosm-os/apex-competitions-builder/issues/49).
A ring mated in play and left undisturbed retained credit. A ring that was never
mated, was knocked off in play, was displaced, or was gripped scored zero.

1. Completion claims were snapshotted using physical predicates before
   finalization; passive end poses could not synthesize completions.
2. A claimed, physically valid stack skipped its carrier peg's lateral impulse
   to avoid throwing its supported ring off. Angles retained their original
   order; unrelated pegs and never-mated rings retained their impulses. v0.8.3
   narrows this exemption to an upright (`seated_now`) carrier.
3. Stack rings were placed with their octagonal faces supporting the peg at its
   measured slot yaw, using shared cross-section symmetries.
4. Finalization continued to measure peg seating, ring position, height,
   upright pose, and gripper clearance after settling without re-pinning.

The board-42 investigation also fixed reset validation around intentional rework
contacts: unrelated clutter/contact validation is enforced, fallback starts are
validated, and spawn bounds, penetration, drift, speed limits, and scoring
tolerances are not relaxed.

The following assisted-reference measurements used the existing **legacy
assisted privileged solver**, which reads true state and may use simulator-state
assists. They are not action-only controller results or release gates.

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
| Full test suite, CI environment locally | **218 passed, 5 skipped, 4 warnings in 995.02 s** (`NIST_PROFILE=v08`) |
| GitHub Actions full suite | **216 passed, 7 skipped, 4 warnings in 1263.09 s** (`NIST_PROFILE=v08`), no failures; [pull-request run 44933410964](https://github.com/unarbos/apex-competition-nist-assembly/actions/runs/44933410964), pytest job 136934748411 on `bc9da2b`. Skip totals are reported separately for the two environments; there were no GitHub test failures. |
| Supplemental no-profile full-suite runs | 217 passed, 5 skipped, 1 failed in legacy seed-1/episode-55 no-op stability in each of two runs; isolated patched and `origin/main` scans passed. This is not used as release-gate evidence. |
| All 64 sampled boards × finalization RNGs 23–26 | **0 construction/validation failures in 256 resets** |
| Built-image 64-episode / two-CPU wall | **156.832025864 s** on cpuset `0,1`, below 600 s; Docker return code 0, all 64 episodes scored, no player fault. Pull-request run 44933410964, `referee-image` job 136934749420, artifact `referee-wall-64` on `bc9da2b`. |
| Digest pins | Unchanged from v0.8.1 |

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
