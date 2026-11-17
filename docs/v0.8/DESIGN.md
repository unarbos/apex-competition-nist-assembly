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
4. If unfinished but prerequisites are done, at most `0.18` from three bounded,
   action-earned physical milestones: `0.06` for a sustained closed-jaw grasp
   that lifts the part, `0.06` for holding that previously lifted part within
   tolerance of its target pose, and `0.06` for retaining that pose after release
   through the final perturbation-settle window. Legacy scalar progress is not a
   scoring input. These milestones are alternatives to, not additions to, done credit.
5. Episode score = mean over steps. Round score = mean over episodes.

A perfect 1.0 requires every step done early, in order, with perfect precision and
stability — treated as unreachable.

## Release bars (calibration gates)

Measured on held-out public seeds (7, 19), ≥2 × 64 episodes, in-process code path that
the images bake:

| Gate | Target |
|---|---|
| Honest privileged (actions only; true state; no qpos snaps) | ≤ ~0.35, clearly > 0 |
| Strong obs-only scripted | ≤ ~0.03 |
| Revealed v0.7.x leader ONNX | ≤ ~0.02 or cleanly rejected (width ≠ 320) |
| Zero / random | ~0 |
| Short PPO/SAC baseline | preferably > scripted (report even if not) |
| Determinism | bit-identical repeats |
| Wall | < 600 s / 64 episodes / 2 CPU |
| Red-team (≥5 hacks) | each near 0 |

## Interface

- Observation `[B, 320]` → action `[B, 5]` at 25 Hz.
- Blueprint encoding occupies the last 32 floats (tier, step kinds, peg ids, blocked/done,
  graph size). See `compound_v08.py` module docstring.
- Records remain `nist_lite_record/4` with `nist_profile: v08` and per-step precision/stability.

## Non-goals for this PR

- Do not tag, release, push images, or change digest pins to new builds.
- Do not merge; owner bot reviews and handles Macrocosmos rollout.
