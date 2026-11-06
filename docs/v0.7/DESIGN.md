# NIST-Lite v0.7 "Assembly III": design (profile `v07`, live as of 0.7.0 / 0.7.1)

## Goal

Far more randomness and compound complexity than v0.6, with an extremely high ceiling.
`c4` should be near-impossible even for a strong privileged controller; partial credit keeps
the gradient. Perception noise stays a minor knob. Eval cost is bounded by **sim horizon**
(and the platform timeout), not wall-clock caps inside the referee — same policy + same seed
must score the same on any machine load.

## What changes vs v0.6

- **New step kinds** (on top of insert / collar / twist):
  - `rework`: peg starts seated in a wrong empty slot; extract, then insert into the true slot.
  - `screw`: twist-while-pressing (multi-regrasp); yaw only counts while the gripper presses down.
  - `stack`: second-level ring that must rest on the *top* of a seated peg (not on the block).
  - `key`: yaw-indexed mating — a seated peg only counts when peg yaw is inside a random key
    window. Non-round keys are drawn near a seated symmetry of the slot so the window is
    reachable; key never shares a peg with twist/screw (shared obs channels).
- **Denser graphs:** compound-step share, denser random precedence, waves up to **2 on c4**
  (code: `waves=(2, 2)` in `TIER_CFG_V07`; not 5).
- **Late reveals:** some twist/screw/key parameters are zeroed in the observation until a
  prerequisite step finishes.
- **Stronger between-step events:** higher fire rate and wider scales; an extra `swap_bias`
  event reshuffles peg-vs-slot residuals.
- **Tighter time slack on c3/c4** (and shorter per-step budgets) so failed episodes do not
  explode sim steps. Hard horizon `MAX_HORIZON_V07 = 1400`.
- **Observation:** still 288-d with the v0.6 layout as a compatible prefix. New kinds reuse
  collar/twist/task channels with documented encodings (see HANDOFF / README).
- **Profile clearance:** `Profile.clearance` for `v07` is a design target only — the simulator
  still draws from `conditions.RANGES["clearance"]` (0.25–0.8 mm). It is **not** wired.

## Scoring

Same rubric family as v0.6 (done = 0.70 + 0.30·time; out-of-order floored at partial; unfinished
≤ 0.30). Weights (code `STEP_WEIGHT_V07`):

| kind | weight |
|---|---|
| insert | 0.30 |
| collar | 0.25 |
| twist | 0.25 |
| screw | 0.40 |
| rework | 2.10 |
| stack | 2.60 |
| key | 3.40 |

## Release bars (calibration, honest privileged)

Privileged reference must run with `oracle=False` / `tools/compound_solver.py --honest`
(no helpers that set `step_done` / `collar_run` / `_key_hold` directly).

- Privileged (honest) ≥ ~0.50 overall.
- v0.6-style / revealed v0.6 kings ≤ ~0.25 (through built images).
- Obs-only / zero low (≤ ~0.15).
- 64-ep live eval well under the 1800 s platform timeout (target ≤ ~600 s for slow kings);
  no referee wall-clock score cuts — platform timeout is a non-scoring abort only.

## v0.7.1 fixes (Macrocosmos preview blockers)

- Referee image bakes `NIST_PROFILE=v07` (was still `v06`).
- Removed `round_wall_s` and per-episode `EPISODE_WALL_S_V07` wall-clock cuts.
- Stack hold counting no longer fights the parent collar-on-block reset.
- Key yaw sampling + `key_tol` obs packing; latch so completed hard steps stay completed.
