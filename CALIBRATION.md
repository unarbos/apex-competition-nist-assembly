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
matched episodes it completes 3 stack steps, 11 collars, 27 keys, 434 inserts,
and 114 reworks.

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

Fractional lift now requires bilateral `fl`/`fr` finger contact with the part,
gripper closure, and a rise of at least 4 mm for eight consecutive control frames
(0.32 s at 25 Hz). A dip below threshold resets the hold. Fractional credit starts
only above 4 mm and is bounded by the unchanged full-lift milestone at 8 mm. This
tightens evidence: sub-mm contact jitter, palm or unilateral pushes, and brief
spikes earn nothing. The full-lift milestone, 0.36 cap, milestone weights, horizon,
and physical task rules are unchanged. Assembly IV task history omits
`key_yaw`/`key_tol`; retarget events omit old/new window values.

## Validation

Test and evaluation artifacts remain outside the repository under
`/home/bb/nist-v080-artifacts/`. The matched reports use simulator-source bundle
SHA-256 `fbe94f221465134f598bbd1b63144d421991333544af04e45c9fe75d8ae73473`.

| Check | Result |
|---|---|
| Full test suite, evidence head `6939708` | **208 passed, 7 skipped, 4 warnings in 841.41 s** (CI run 44581816589) and **208 passed, 7 skipped, 4 warnings in 915.45 s** (CI run 44581828716); Stable-Baselines3 is absent |
| Full test suite, scorer head `4eed6ad` | **208 passed, 7 skipped, 4 warnings in 917.68 s** (CI run 44574980029) and **208 passed, 7 skipped, 4 warnings in 1189.74 s** (CI run 44574994762); Stable-Baselines3 is absent |
| Final-source focused regressions | **74 passed, 1 skipped in 616.70 s**; additional event-privacy and serialized-envelope checks **4 passed, 54 deselected in 28.89 s** |
| Built-image 64-episode / two-CPU wall, evidence head `6939708` | **128.524 s** (CI run 44581816589) and **318.906 s** (CI run 44581828716), cpuset 0,1; no player fault; both pass 600 s |
| Built-image 64-episode / two-CPU wall, scorer head `4eed6ad` | **232.121 s** (CI run 44574994762) and **232.520 s** (CI run 44574980029), cpuset 0,1; no player fault; both pass 600 s |

Per-round wall time is not aggregated across resumable measurement chunks. The
built-image CI runs above are the wall-time evidence; they are separate from the
in-process policy comparison below.

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
| Ordinary observation-only script | 0.011692006070 | 0.009105692788 | 0.005630799528 | 0.008809499462 [0.007074604350, 0.010651619112] | 0.003041439470 |
| Strong observation-only script | 0.008312436355 | 0.008266258032 | 0.008550083894 | 0.008376259427 [0.006663470605, 0.010172337927] | 0.000152296810 |
| CEM | 0.013811296527 | 0.015439669214 | 0.009779517504 | 0.013010161082 [0.011147214975, 0.014878161420] | 0.002913879347 |
| Legacy assisted privileged reference | 0.132693435266 | 0.142225513227 | 0.152449479002 | 0.142456142498 [0.127728299866, 0.157831118611] | 0.009880040912 |

**Both zero and uniform random are exactly zero in all 320 episodes across all
five seeds.** The additional seed-43 and seed-59 means are also exactly zero.
The former seed-31 episode-11 sub-mm contact excursion earns no credit under the
sustained bilateral-grasp rule; no scores are hidden by rounding.

Ordinary-minus-strong script is **0.000433240035 [-0.001837687157, 0.002691805617]**
(paired episode bootstrap; interval includes zero). CEM minus the best
observation-only script is **0.004200661619 [0.002315596179, 0.006127752842]**
(interval excludes zero). The privileged reference is not a release gate and was
not tuned to increase its score.

## Red-team pass

The following results are measured on the tightened scorer. Generic attack
rollouts use seed 31 and eight episodes per attack; each listed generic rollout
scored exactly zero. The red-team summary, per-attack reports, and physical
fixtures remain in the external artifact directory.

| Attack class | Measured result |
|---|---|
| Contact jitter / push without grasp | `jitter`, `vibrate_parts`, `slam_down`, and `transport_wrist_spoof`: all zero. A sub-mm rise, unilateral or palm contact, or a rise that breaks the eight-frame hold earns no fractional lift. |
| Wrist-yaw sweeps / key precision farming | `yaw_spin` and `cue_hillclimb_sweep`: all zero. Natural-seat key freebie completed **0/20** keys. Transient visits fail the hold; completion precision cannot improve after moving to another key yaw. |
| Dependent-step re-grasp / re-lift | Key and screw target credit is exactly zero, including with injected stale insertion evidence. The separate prerequisite insert credit of 0.002727272727 in each fixture is not dependent-step credit. |
| False / barely mated and knocked-off stack rings | Post-mate knock-off, 12 mm lateral false mate, and -20 mm vertical false mate: zero with final physical rejection. A +12 mm vertical displaced pose fails immediately but can settle back onto the peg; its fixture score remains zero because prerequisites were not completed. This is not described as a knock-off exemption bypass. |
| NaN, infinity, extreme actions, and `/act` timing | NaN, ±infinity, wrong shape, nonnumeric, extreme-action, sensor-spoof, early-termination, solver-blowup, and drop-near-hole rollouts: zero. Invalid rows terminate their episodes with `invalid_action` and zero score; a one-step extreme action scores zero. Late `/act` returns `act failed: PlayerError`, zero score, in **1.358 s**. Wrong batch shape and unreadable responses also fail closed with zero score. |
| Observation / player-visible envelope recovery | Observation-privacy and serialized-envelope tests pass. Assembly IV task history and retarget events contain no key-window values; private dynamics, salt, and reserved endpoints are absent. The explicitly public training-range header remains separate. |

The analytic single-kind-only bound across the tested blueprints is
**0.427639751553**, below 0.55. This is a scoring bound, not an achieved attack
score.

## Determinism

All six policies were independently repeated on seeds 7, 19, and 31: **1,152 repeat
episodes**. Every per-seed report matches the original episode records, and every
complete three-seed report is byte-identical to its canonical original, including
scores, terminal reasons, completed kinds, and milestone fields. For zero and
random, the comparison uses the three matched seeds from the original five-seed
report. A redundant per-round scorer-hash field added by the repeat worker is
validated then omitted for canonical comparison; the top-level scorer identity
and all episode values are unchanged. Timing is null in both reports; wall time
is measured separately by the built-image CI jobs.

| Policy | Complete repeated report SHA-256 |
|---|---|
| `zero` | `80f216a4479bf3a15032f185c9745117c7e13b47f945e0ded9a644444b77b45a` |
| `random` | `f614a7c67651c558cbe79b062266a3277d897b70e75124d0849d87383b65261c` |
| `script` | `c048e9cc9f697b0156c988936f86fb2fe1d4ea70ee7c1a0fcfdaec52a243f3c4` |
| `strong_script` | `0d6b84f2a537399fca920ba3a7b40785d6214e7bbea505d9134efd2d3d043cba` |
| `cem` | `2b3c76a4ca2a6ec0601c799f53bb44342f9f4679ca3f42f6a562d9bfda316405` |
| `privileged` | `24060af8661b61800718ada2cfcb52b3b56fc121f386bdb59ed396fadb658918` |

## Historical results

Earlier v0.8.0 and controller-tuning measurements are not evidence for this PR.
In particular, prior stack-finalization comparisons that re-pinned rings during
settling are not reused. Generated models, demonstrations, and measurement logs
are not committed. No miner ONNX, hotkeys, secrets, merge, tag, release, image
push, or digest change is included.
