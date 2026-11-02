# NIST-Lite v0.6 "Assembly II": design (released as v0.6.0; patched as v0.6.1)

## Goal

Make tasks compound and randomized, with a very high ceiling. The hardest tier should be
close to unsolvable, and partial progress is scored so there is no cliff. Perception noise
is a minor knob, not the main axis.

## Task: random compound assembly (`referee/nist_sim/compound.py`)

- **Steps:** `insert` (peg into hole), `collar` (a ring that fits only over a peg that is
  already seated), and `twist` (rotate a seated round peg past a random lock angle).
- **Precedence graph:** mandatory edges (insert before collar/twist on the same peg) plus
  random cross-part edges. Steps done out of order still count, at reduced credit.
- **Waves:** parts are parked off-board and arrive in random waves, either when the
  previous wave is done or on a timer.
- **Between-step dynamics:** after a step completes, a random event may fire (friction,
  actuator gain, part mass, action latency, sensor re-bias, a lateral shove).
- **Tiers c1 to c4** (15/30/30/25% of episodes): part count, collar and twist fractions,
  wave count, edge density, event rate, and time slack all scale with the tier. Every
  episode has at least one collar or twist.
- **Lost parts** do not end the episode. The part is removed and its steps score zero.

## Scoring

Each step is weighted: insert 0.3, collar 1.5, twist 1.0. A done step earns
0.70 + 0.30 × (time remaining fraction). A done step whose prerequisites were not met
earns `max(0.30, 0.15 × in-order credit)`. An unfinished step earns up to 0.30 for
partial progress (collar 0.20/0.30 tiers require the peg in its hole). The episode
score is the weighted mean.

## Observation and physics (profile `v06`)

- Observation: 288-d, the v0.5 184-d prefix plus collar, twist, step-status, and wave
  features.
- Physics: 500 Hz (`physics_dt` 0.002), 8-segment collars, 25 Hz control.
- A stateful-policy protocol and action rounding ship only together. Rounding stays off
  until then.

## Release bars (`tools/calib_v06.py`)

- Privileged solver ≥ 0.30 overall and ≤ 0.70 on c4. The ceiling stays open.
- Fixed obs-only scripted controller ≤ 0.5× privileged.
- A learned policy beats scripted at z ≥ 3.
- Reroll determinism ≥ 0.95.
- The projected live round fits the agreed eval budget.

## Status

Released in v0.6.0; v0.6.1 patches out-of-order credit, collar partial gating, and
`nist_lite_record/4` before launch. The referee runs this task, and the live round size is 8
episodes per stratum (64) on 2 CPU. Open after launch: a learned policy beating scripted (naive BC 0.006; DAgger with a
stateful policy and PPO are next), a stateful-policy protocol with action rounding, and more
step kinds (gear/key, screw, rework, stack).
