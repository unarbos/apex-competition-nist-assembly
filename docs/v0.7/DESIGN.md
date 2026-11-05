# NIST-Lite v0.7 "Assembly III": design (dev; profile `v07`)

## Goal

Far more randomness and compound complexity than v0.6, with an extremely high ceiling.
`c4` should be near-impossible even for a strong privileged controller; partial credit keeps
the gradient. Perception noise stays a minor knob. Eval cost must stay inside the platform
timeout for a ~1 ms/call policy on 128 episodes / 2 CPU.

## What changes vs v0.6

- **New step kinds** (on top of insert / collar / twist):
  - `rework`: peg starts seated in a wrong empty slot; extract, then insert into the true slot.
  - `screw`: twist-while-pressing (multi-regrasp); yaw only counts while the gripper presses down.
  - `stack`: second-level ring that must rest on the *top* of a seated peg (not on the block).
  - `key`: yaw-indexed mating — a seated peg (or its collar/stack) only counts when peg yaw is
    inside a random key window relative to the slot.
- **Longer, denser graphs:** more extra pegs on hard tiers, higher compound-step share, denser
  random precedence, more waves (up to 5 on c4).
- **Late reveals:** some twist/screw/key parameters are zeroed in the observation until a
  prerequisite step finishes (or a wave arrives).
- **Stronger between-step events:** higher fire rate and wider scales; an extra `swap_bias`
  event reshuffles peg-vs-slot residuals.
- **Tighter time slack on c3/c4** (and slightly shorter per-step budgets) so failed episodes
  do not explode wall time.
- **Observation:** still 288-d with the v0.6 layout as a compatible prefix. New kinds reuse
  collar/twist/task channels with documented encodings (see HANDOFF / README).

## Scoring

Same rubric family as v0.6 (done = 0.70 + 0.30·time; out-of-order floored at partial; unfinished
≤ 0.30). Weights favour the new hard steps: insert 0.25, collar 1.2, twist 0.9, rework 1.4,
screw 1.6, stack 1.5, key 1.3.

## Release bars (calibration)

- Privileged reference clearly above a reasonable obs-only scripted controller; c4 ≤ ~0.4.
- Obs-only scripted (full 288-d compound features) ≤ ~0.35 overall.
- Projected live eval ≤ ~600 s for 128 episodes on 2 CPU for a ~1 ms/call policy.
- v06 profile and tests remain unchanged.
