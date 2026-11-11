# NIST-Lite v0.7 "Assembly III": design (profile `v07`, live as of 0.7.0–0.7.4)

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
  explode sim steps. Hard horizon `MAX_HORIZON_V07 = 1600` (v0.7.4; was 1400).
- **Observation:** still 288-d with the v0.6 layout as a compatible prefix. New kinds reuse
  collar/twist/task channels with documented encodings (see HANDOFF / README).
- **Profile clearance:** `Profile.clearance` for `v07` is a design target only — the simulator
  still draws from `conditions.RANGES["clearance"]` (0.25–0.8 mm). It is **not** wired.

## Scoring

Same rubric family as v0.6 (done = 0.70 + 0.30·time; unfinished ≤ 0.30). In v0.7.3, **keys**
finished out of order score **0** (stack/rework keep the out-of-order floor). Weights (code
`STEP_WEIGHT_V07`, v0.7.3):

| kind | weight (v0.7.4) |
|---|---|
| insert | 0.03 |
| collar | 0.03 |
| twist | 0.03 |
| screw | 0.05 |
| rework | 1.60 |
| stack | 3.50 |
| key | 4.50 |

## Release bars (calibration, honest privileged)

Privileged reference must run with `oracle=False` / `tools/compound_solver.py --honest`
(no helpers that set `step_done` / `collar_run` / `_key_hold` directly).

- Privileged (honest) ≥ ~0.45 overall.
- v0.7.1-era / v0.6-style kings ≤ ~0.25 (through built images / same code path).
- Obs-only / zero low (≤ ~0.15).
- 64-ep live eval well under the 1800 s platform timeout (target ≤ ~600 s for slow kings);
  no referee wall-clock score cuts — platform timeout is a non-scoring abort only.

## v0.7.1 fixes (Macrocosmos preview blockers)

- Referee image bakes `NIST_PROFILE=v07` (was still `v06`).
- Removed `round_wall_s` and per-episode `EPISODE_WALL_S_V07` wall-clock cuts.
- Stack hold counting no longer fights the parent collar-on-block reset.
- Key yaw sampling + `key_tol` obs packing; latch so completed hard steps stay completed.

## v0.7.4 branching / probe harden (vs adaptive conditional scripts)

- **Press-gated probe-refined key yaw:** after touch/late reveal, obs `sin/cos` carries a
  large decaying bias that converges only while *pressing* on the peg (`KEY_PROBE_FRAMES`) —
  touch-and-snap scripts keep a wrong absolute yaw; press-and-refine (or true-state) works.
- **Single-active key rotation (`activate`):** with ≥2 unfinished keys, only one may complete;
  inactive keys report `blocked=1` and cannot accumulate hold.
- **`regraph`:** drop soft unfinished edges and redraw 1–3 wave-consistent ones (stronger than
  single-edge `reorder`); often rotates the active key.
- **Post-stack key retarget** (plus denser post-insert retarget); deeper rework seat.
- Interface unchanged (`[B,288]→[B,5]`); no wall-clock score caps; miner-path latch sealed.

## v0.7.3 adaptive harden (vs v0.7.1-trained kings)

- Mid-episode `retarget` / `reorder` / `flip_twist` events + post-insert key retarget.
- Touch-gated key/twist reveals; stack-before-key edges; keys complete only after prereqs.
- Do not manufacture difficulty via unreachable physics; miner-path stack latch stays sealed.

## v0.7.2 fixes (stack action-mate + twist/screw draw)

- Stack rings use a slight interference fit so a deliberate peg-top place can rest in the
  mate zone; `_maybe_mate_stacks` arms on a closed grasp in zone, then pins after
  `STACK_MATE_HOLD` released frames (single-frame sweeps do not free-pin).
- Honest privileged (`oracle=False`) must not call `_pin_stack_on_peg` / stack snaps;
  completion uses the same latch miners use (pose nudges only).
- Twist/screw: per-eligible-peg Bernoulli(`twist_frac` / `screw_frac`) instead of
  `round(frac * n)` (which was almost always 0 on live-sized graphs).
- King gate was calibrated only on v0.6-era kings — insufficient for Round 22 (see v0.7.3).
