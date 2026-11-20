# NIST-Lite v0.8 "Assembly IV": design (profile `v08`)

## Goal

A hard reset for miners with frontier-lab RL skill and compute. Prefer too hard
(near-impossible ceiling) over another easy saturation. Procedural generalization
over memorization; exploit-resistant end-state scoring; no answer-leaking cues.

Closes the v0.7.5 R32 hole classes from audit draft PR #40 by design: (1) no
hill-climbable per-step cues, (2) no key windows on natural seat / slot symmetries,
(3) no single heavily weighted subtask that can carry the score.

## What changes vs v0.7.x

- **No free-start credit.** v0.8 removes grasped and near-hole spawns, including
  extra pegs added by graph generation. Reset-only states and passive end poses
  do not complete a step or earn partial credit.
- **Longer blueprints:** typically ~8–28 steps (mean ~12 on public seeds) with denser
  precedence, more keys (up to 3), stacks, screws, reworks, and waves (up to 4 on c4).
- **Equal per-step weight.** Farming inserts/collars cannot dominate. Score is the mean
  over blueprint steps of verified credit.
- **End-state + perturbation scoring.** Credit requires a physically verified configuration
  at episode end that survives a small XY impulse settle. Insert latch alone does not
  score if the part was pulled out.
- **No answer cues.** Key closeness, wrist detent, and retarget pulse are removed from the
  observation. Absolute key yaw stays hidden. A present-mark (0.5) may appear after touch
  on an active key — it does not encode window geometry.
- **Anti-aligned key windows.** Index yaw is sampled away from slot symmetries and wrist-zero
  (closes the v0.7.5 freebie class where natural insert yaw completed keys).
  Completion records the key's own index precision; finalization verifies that its peg
  remains seated and does not require the wrist to remain in that key's window.
  An unfinished key, twist, or screw earns no generic insert-part partial credit.
- **No D-flat auto-arm.** Leave/reenter arming is mandatory for every key family.
- **Dynamics stretch.** Per-episode mass / friction / gain / latency stretches force
  in-episode system identification. Mid-episode physics events remain.
- **Observation 320-d** (288-d Assembly III layout + 32-d blueprint encoding). Action stays
  5-d. Old 288/184 ONNX policies are rejected by the player image.
- **Horizon** hard-capped at 920 control steps (sim-cost only; no wall-clock score cuts).

## Scoring

For each blueprint step:

1. Physically verify the end state (and stability after perturbation).
2. If verified and completed in dependency order:  
   `credit = (0.80 + 0.20 · time_term) · (0.70 + 0.30 · precision)`.
3. If verified but out of order: `credit = 0.03`.
4. If an insert, rework, collar, or stack step is unfinished, unblocked and its
   prerequisites are done, at most `0.36` from five bounded,
   action-earned physical evidence buckets: `0.12` for sustained grasp/lift, and `0.06`
   each for held transport toward the real hole, held insertion, near-target
   alignment, and stable release through the final perturbation-settle window.
   An earned mated interference-fit stack ring is exempt from the direct collar
   impulse; its peg is still perturbed and the measured ring pose is verified
   after settling. Finalization never re-pins the ring. Unmated loose rings
   still receive the direct impulse.
   Transport requires earned lift and closed contacting jaws; its best-so-far
   fraction measures actual part motion toward the target, not wrist proximity.
   Insertion additionally requires transport, an upright held part, and depth
   inside the actual hole region. Their continuous fractions are bounded by
   `0.60`. Within the lift bucket, a best-so-far fraction of at most `0.024`
   recognizes a real grasp that raises the part 4–8 mm above its settled height
   for eight consecutive control frames (0.32 s at 25 Hz). Both finger bodies
   must contact the part while the jaws are closed. This rejects the observed
   ~0.8 mm soft-contact excursions that previously produced fractional lift:
   palm pushes, unilateral contact, sub-mm jitter, and brief spikes are not a
   grasp-and-raise. The threshold remains below the 8 mm full-lift milestone.
   Oscillation cannot accumulate credit after valid evidence is earned, and
   released/falling parts, wrist motion, contact alone, absolute spawn height,
   passive poses and settling frames earn none. Legacy scalar progress is not a
   scoring input.
   These buckets are alternatives to, not additions to, done credit. Unfinished
   key, twist, and screw steps earn no generic insertion partial credit.
5. Episode score = mean over steps. Round score = mean over episodes.

A perfect 1.0 requires every step done early, in order, with perfect precision and
stability — treated as unreachable.

## Release bars (calibration gates)

Measured on held-out public seeds (7, 19), ≥2 × 64 episodes, in-process code path that
the images bake:

| Gate | Target |
|---|---|
| Honest privileged (actions only; true state; no qpos snaps) | ~0.2–0.35 on the 8-ep slice; far below 1.0 |
| Strong obs-only scripted | ~0.01–0.05 |
| Revealed v0.7.x leader ONNX | ≤ ~0.02 or cleanly rejected (width ≠ 320) |
| Zero / random | exactly ~0 (within 1e-4) |
| Short residual / CEM learner | beat scripted beyond seed noise (report CIs) |
| Determinism | bit-identical repeats |
| Wall | < 600 s / 64 episodes / 2 CPU |
| Red-team (≥5 hacks) | each near 0 |

## Interface

- Observation `[B, 320]` → action `[B, 5]` at 25 Hz.
- Blueprint encoding occupies the last 32 floats (tier, step kinds, peg ids, blocked/done,
  graph size). See `compound_v08.py` module docstring.
- Records remain `nist_lite_record/4` with `nist_profile: v08` and per-step precision/stability.
  Assembly IV history omits `key_yaw` and `key_tol` from task steps: history is
  player-visible after a round, not a private replay cache. The older profiles'
  record behavior is unchanged. Public training-draw ranges remain distinct
  from reserved endpoints and per-episode hidden dynamics.

## Non-goals for this PR

- Do not tag, release, push images, or change digest pins to new builds.
- Do not merge; owner bot reviews and handles Macrocosmos rollout.
- Privileged-controller and PPO/SAC tuning are deferred to a later PR. The
  release gate for this scoring-fix PR is the four Macrocosmos issue-45 fixes.
- Non-round key-window randomization is unchanged; the predictable symmetry
  midpoint behavior remains a documented open design choice, not a blind search.
