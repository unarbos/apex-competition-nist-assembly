# NIST Assembly Competition

Docs page text for v0.7.1 (Assembly III). Platform-wide items that Macrocosmos sets
(submission fee, rate limits) are not repeated here.

### NIST Assembly Competition

**Getting a peg to the hole is not the same as seating it.** The last millimetre is the hard part: with 0.25 to 0.8 mm of radial clearance, the contact that completes an insertion is the same contact that jams it. A peg still held in the gripper has not been assembled, and on a full board each seated peg is one more thing the gripper can knock out on the way to the next.

The NIST Assembly competition asks miners to drive a **parallel 2-finger gripper on a Cartesian stage** (x, y, z, yaw) through **multi-peg sequential assembly** on a board modelled on the **NIST Assembly Task Board #1**, simulated in MuJoCo on CPU. Miners submit an ONNX policy that outputs incremental stage and gripper commands at 25 Hz. The setup borrows from AlphaNIST (Zhang et al., arXiv 2610.13354): the ATB-1 board, the 1 mm success criterion, 25 Hz control, diverse start states and free-order multi-part episodes. It is much smaller: pegs only, primitive geometry, a 4-DoF stage instead of a 7-DoF arm, and noisy state estimates instead of point clouds. Since v0.6 every episode is a **compound assembly task**; v0.7 adds rework, press-gated screws, stack rings and obs-indexed keys. Scoring gives weighted partial credit per step so that a ranked competition has a gradient. Cost is bounded by the sim horizon only — same policy and seed score the same under any machine load.

**What changed in v0.7.1**

* **Referee image bakes `NIST_PROFILE=v07`.** The v0.7.0 image still had `v06` in the Dockerfile, so `setdefault` never promoted the live task.
* **No wall-clock scoring caps.** Removed `round_wall_s` and the per-episode 18 s wall. Horizon (≤ 1400 steps) is the only cost bound inside the referee.
* **Stack / key / latch fixes.** Stack hold counting works; key yaw is reachable on non-round pegs and `key_tol` is shown; completed hard steps stay completed.
* **Docs match code.** Weights insert 0.45 / collar 0.60 / twist 0.60 / screw 1.00 / rework 1.50 / stack 1.40 / key 1.80; c4 waves at most 2; profile `clearance` is wired via `conditions.active_ranges()` (v07 draws 0.20–0.68 mm). Result/records log `nist_profile`.

**What changed in v0.7.0 (Assembly III, breaking)**

* New step kinds: `rework`, `screw`, `stack`, `key`. Denser graphs, late reveals, `swap_bias` events.
* Observation stays 288-d; encodings reuse twist/collar channels (see README).
* Live default profile `v07`; `NIST_PROFILE=v06` still selects Assembly II for regression.

**What changed in v0.6.3**

* **Later-wave parts after a mass change.** The v0.6.2 mass fix also cleared the collision masks of parts that were still waiting for their wave, so when they arrived they fell through the board and counted as lost. They now collide normally. Thanks to OddK for the report.

**What changed in v0.6.2**

* **Mass-change fix.** After a hidden part-mass change the simulator now refreshes the derived MuJoCo constants (subtree mass, inverse weights). Before, they stayed at the old mass, so the next grasp could oscillate and end the episode as `unstable`. Thanks to msulab for the report.

**What changed in v0.6.1**

* **Out-of-order credit.** A step done before its prerequisites earns `max(0.30, 0.15 × in-order credit)`, so finishing never pays less than stopping just short of done.
* **Collar partial credit needs the peg.** The 0.20 / 0.30 collar tiers require that collar's peg in its hole.
* **Records `nist_lite_record/4`.** Episode rows carry `conditions.collars`, the full `task` graph, `step_results` and `events`. Trajectory frames add collar poses plus `parked` / `collar_parked` (world frame, same conventions as `/2`).
* Smaller: extra precedence only when wave(a) ≤ wave(b); independent estimate biases on c3/c4 extra pegs; collar rows get the same glitch bursts and dropout as peg rows; release builds ignore `NIST_*` physics/rounding overrides.

**What changed in v0.6.0 (breaking)**

* **Compound tasks.** Each episode is a random assembly graph with three kinds of steps. An *insert* puts a peg into its own hole. A *collar* drops a ring over a peg that is already seated. A *twist* turns a seated round peg by a random 1.2 to 2.0 rad in a given direction, a bayonet-style lock that can take several regrasps.
* **Random ordering.** A peg must be seated before its collar or twist, and random extra edges link steps across parts. Each step's blocked flag is observed. A step done before its prerequisites still counts, at reduced credit (see v0.6.1).
* **Parts arrive in waves.** There are 1 to 4 waves per episode. A later wave arrives when the earlier work is done or at a random deadline. Parts that haven't arrived are off the board and zeroed in the observation.
* **Hidden changes between subtasks.** After a step completes, friction, controller gain, part mass, actuation delay or the pose-estimate bias may change, or a loose part may be shoved. The policy is not told.
* **Four complexity tiers, c1 to c4** (15/30/30/25% of each stratum). They scale part count (up to 2 extra pegs), collar and twist share, wave count, graph density, event rate and time slack. c4 is meant to be close to impossible; even our privileged solver scores about 0.24 there.
* **Per-step partial credit**, weighted toward the new steps (insert 0.3, collar 1.5, twist 1.0). A lost part no longer ends the episode.
* **Observation 288 floats.** The first 184 are the v0.5 layout. A policy whose input is statically 184 wide still runs on that prefix.
* Physics runs at 500 Hz (20 substeps per action) to keep the longer episodes inside the time limit. Estimate glitch bursts are at half the v0.5.1 rate.

**What changed in v0.5.1**

* **Harder observations.** The extra peg-vs-hole estimate offset is now 5 to 11.5 mm (was 1.5 to 5 mm), pose noise is 0.8 to 3.5 mm (was 0.4 to 2.0 mm), estimates drift faster and on every tier, and short "glitch" bursts (8 to 30 steps) add extra noise and a transient offset to every pose estimate.
* **More lying-down starts:** about 22% of pegs on the hard tier and 12% on the medium tier (was 8% and 2%).
* **Fixes** from the v0.5.0 validation: overlapping layouts are redrawn, free pegs never start inside another hole, a pre-grasped hand yaw stays inside the joint range, and records carry both the effective and the drawn start type.
* Same task, interface (`[B, 184]` in, `[B, 5]` out) and scoring. Scores are not directly comparable with v0.5.0.

On v0.5.1, fixed insert-only controllers from earlier rounds score about 0.20 on v0.6, because they never earn collar or twist credit. On v0.5.0 the leading hand-coded controllers scored about 0.92, above our privileged solver. On v0.5.1 the same controllers score about 0.15 to 0.27, while the privileged solver scores about 0.66.

**What changed in v0.5.0**

* **Every episode draws its own board.** Hole positions, hole yaw, peg size and peg cross-section are drawn per episode from continuous ranges. There is no fixed slot table to memorise.
* **More cross-sections, no shape label.** Live families are round, rectangle/square, regular 5/6/8-sided polygons, and D-flat, with inscribed sizes from 4 to 16 mm. The observation gives sizes and an estimated hole yaw but never says which family a peg is.
* **Some families and condition ranges are held out.** They never appear in live rounds and are used to check that leading policies generalise.
* **The interface is unchanged** (`[B, 184]` in, `[B, 5]` out), but observation features 8 to 11 of each peg row changed meaning (see below). Policies tuned to v0.4.0 will score much lower.

The point of these changes is to reward policies that generalise. A controller tuned to a fixed layout scored up to 0.79 on v0.4.0; on v0.5.0 the top hand-coded controllers from round 5 score about 0.01 to 0.02.

**Evaluation Overview**

Each live evaluation runs **64 episodes**: 8 strata (number of pegs K ∈ {3, 4, 5, 6} × clutter low/high) with 8 episodes each. (It was 128 on v0.5. Compound episodes cost about 2.4× more to simulate, so the count was cut to fit the time limit on 2 CPUs.) Inside each stratum, the physical conditions are drawn by Latin hypercube. Each episode gets a sensing difficulty tier (about **25% easy, 35% medium, 40% hard**) and a complexity tier (**15% c1, 30% c2, 30% c3, 25% c4**).

| Condition | Live range |
| --- | --- |
| Pegs per episode (K) | 3 – 6 by stratum (mean 4.5), +1 on c3 and +2 on c4 (at most 8) |
| Collars / twists | c1 ~25% / 30%, c2 50% / 60%, c3 70% / 80%, c4 90% / 100% (collar share of pegs, at most 4 collars, only on pegs up to 7.2 mm outer radius; twist share of round pegs); at least one per episode |
| Waves / extra precedence | c1 1 / none, c2 1–2 / 10%, c3 2–3 / 20%, c4 3–4 / 30% of compatible step pairs |
| Between-step events | after each completed step with probability 0 / 25% / 45% / 65% (c1–c4): friction × 0.55–1.6, gain × 0.6–1.4, part mass × 0.7–2.0, actuation delay 0–2 steps, estimate re-bias 1–4 mm, or a shove |
| Clutter blocks | 0 – 1 (low) or 2 – 3 (high); often placed near a hole approach |
| Board layout | 8 holes per board; positions, yaw, family and size drawn per episode |
| Board pose | ±30 mm in x and y, ±0.15 rad yaw, per episode |
| Radial clearance | 0.25 – 0.8 mm, with a variable entry chamfer |
| Friction | 0.3 – 1.3 |
| Peg mass | 5 – 50 g |
| Controller gain | × 0.6 – 1.4 |
| Pose-estimate noise | 0.8 – 3.5 mm, 0.015 – 0.06 rad (smaller on easy and medium tiers) |
| Observation dropout | 3 – 18% per row per step (the row repeats its last value) |
| Hole-estimate bias | 4 – 10 mm and roughly 7 – 16° per episode, independent per hole (smaller on easy and medium tiers) |
| Extra peg-vs-hole bias | 5 – 11.5 mm, independent per peg, so the reported peg-minus-hole offset does not cancel |
| Estimate glitches | bursts of 8 – 30 steps with 4× noise and a 2 – 6 mm transient offset per object; up to 2% chance per step on the hard tier. Bursts cover about 25% of control steps on the live mix (easy 16%, medium 25%, hard 29%) |
| Other | bias drift (0.01 – 0.06 mm per step random walk), Cartesian backlash, a rare one-step action delay, and on some episodes one mid-episode disturbance (a peg shove, a gripper slip, or a hand shift) |
| Control / physics | 25 Hz control, 500 Hz physics (20 substeps per action). **Fixed.** |
| Horizon | sum of step budgets (insert 260, +140 if the peg starts lying, collar 200, twist 130) × tier slack (1.2 / 1.0 / 0.85 / 0.70) + 40 per extra wave; mean about 1,800 steps (72 s) |

* **The difficulty does not drift between rounds.** Strata, tier mix and ranges are fixed; only the draw inside them changes.
* Within a round, **every submission faces the same episodes**, and the incumbent is re-scored on them.
* **The round seed never enters the player sandbox.** Every episode is derived inside the referee from the platform seed plus private build entropy. Local runs use the same distribution but cannot reproduce a live round.

**Start States**

Each peg gets its own start type. Easy-tier episodes use grasped and near-hole starts; medium and hard tiers mix all five. Pegs in a later wave arrive as `loose` or `lying` at a free spot. Collars start lying flat on the board, away from the holes.

| Start type | Peg begins |
| --- | --- |
| `grasped` | already in the gripper, 10 – 40 mm above and within 10 mm of its hole (at most one per episode) |
| `near_hole` | standing on the board just beside its hole |
| `loose` | standing upright on the board, 40 – 70 mm from its hole |
| `lying` | toppled on its side, 50 – 70 mm away (about 22% of hard-tier and 12% of medium-tier pegs). A lying peg can be reached and grasped (tested), but seating it takes a re-orienting regrasp. Our privileged solver seats none of them, and we have not shown a lying seat on the multi-peg board. Treat these pegs, about 13% of inserts, and the steps that depend on them as unproven headroom (`CALIBRATION.md`) |
| `tray` | standing in a ring with the other tray pegs, in a tray area near one edge of the board |

**Step by Step**

At each control step the policy receives a **288-float observation**:

| Block | Size | Contents |
| --- | --- | --- |
| Proprioception | 16 | hand position relative to the board (×10, m); sin/cos of hand yaw; hand velocity (3); finger gap (×50); wrist force (3); wrist torque (3); finger-pad force (1) |
| Last action | 5 | the previous action |
| Peg table | 8 × 17 | one row per active peg, paired with its target hole; unused rows are zero. Per row: presence (1); estimated peg position relative to the estimated hole (×10, 3); estimated peg quaternion (4); estimated hole yaw as sin, cos (2); peg outer (circumscribed) size (×50, 1); a zero (1); peg inscribed size (×50, 1); estimated hole position relative to the board (×10, 3); seated flag (1) |
| Clutter table | 3 × 9 | presence; estimated position relative to the board (×10, 3); quaternion (4); size (×50) |
| Task table | 8 × 6 | per peg row: insert done, insert blocked, twist direction (±1, 0 = none), twist progress (0 – 1), twist blocked, has-collar |
| Collar table | 4 × 13 | presence (0 until it arrives); estimated collar position relative to the estimated target hole (×10, 3); quaternion (4); inner and outer radius (×100, 2); target peg row / 8; done; blocked |
| Globals | 4 | elapsed fraction of the horizon; fraction of steps done; fraction of waves arrived; number of steps / 16 |

* Row features 8 – 11 (hole yaw sin/cos, outer size, zero) replaced the v0.4.0 shape one-hot.
* **Exact peg and hole poses are never given.** Every pose in the table is an estimate with the biases and noise listed above. A peg is aligned when its yaw matches the hole yaw, up to the cross-section's own symmetry, which the policy has to work out from the sizes and from contact.
* **Clearance, friction, mass, gain and the bias values are not observed.** The wrist force/torque and pad force channels are what a policy can use to find a hole it cannot see precisely.
* **There is no recurrent state.** The policy is called once per step on the current observation; only the previous action is fed back. The task and collar tables tell the policy where it is in the graph.
* Rows for parts that haven't arrived yet are zero.

The policy returns **5 floats in [−1, 1]**: `dx, dy, dz` (±4 mm per step), `dyaw` (±0.08 rad per step), and gripper (+1 open, −1 close). These are incremental position targets tracked by a PD loop. Stage travel is ±250 mm in x and y, −3 to 250 mm in z, and ±3.1 rad in yaw.

**Constraints**

* Terminal reasons, reported per episode:
  * `assembled`: every step is done and the gripper is clear. The episode ends early.
  * `time_limit`: the horizon ran out.
  * `unstable`: non-finite state, or a part moving faster than 4 m/s.
  * A part that leaves the board area (|x| or |y| beyond 300 mm) or falls below the surface is removed; its unfinished steps score their partial credit, and the episode continues.
  * `invalid_action`: that episode's action was not a finite 5-vector. Ends that episode.
  * `player_fault`: the `/act` call errored, missed its deadline, or returned a batch of the wrong shape. Ends every episode still running.
* In every case, steps that are done when the episode stops keep their credit; unfinished steps keep their partial-progress credit.
* One batched `/act` call per control step covers every episode still running, with a **500 ms deadline per call**.
* Timeouts: **1800 s** for the referee and for the player. A full v0.6 live evaluation (64 episodes) is projected at about **500 – 1150 s**, depending on the policy (v0.5.1 took about 585 s for 128 episodes).
* Player and referee each run on **2 CPU / 2 GiB / no GPU**.

**Scoring**

A peg is **inserted** when, for 5 consecutive control steps, it is in its own hole with its bottom within 1 mm of fully inserted, its centre within the hole radius, its tilt within 10° of vertical, and the gripper is not touching it. A **collar** is done when it rests on the slot block, centred on its seated peg within 3 mm. A **twist** is done when the seated peg has turned past its target angle in the given direction. A done step stays done while the part stays in place, and is undone if it is knocked out.

Each step k has a weight `w_k` (insert 0.3, collar 1.5, twist 1.0) and a credit `c_k`:

```
c_k = 0.70 + 0.30 * (1 - t_k / T)              done in order and still done at the end (t_k = step when done, T = horizon)
c_k = max(0.30, 0.15 * in-order credit)       done while one of its prerequisites was still open
c_k = up to 0.30                              not done: partial progress (grasped/lifted, aligned over the target,
                                              partly inserted, collar around a seated peg, fraction twisted)
```

```
S_e       = sum_k w_k c_k / sum_k w_k
raw_score = mean of S_e over all episodes      -> [0.0, 1.0]
```

* **Compound steps dominate.** A collar is worth five inserts, and a twist more than three. An insert-only controller tops out far below a policy that finishes the graph.
* **Order matters.** Doing a step early earns little; the blocked flags say what is open.
* **Partial progress counts**, so even c4 episodes give a gradient.
* The denominator is fixed: every episode counts, including ones ended by a fault.
* To take the lead, a challenger must beat the current top raw score by more than 1%. The incumbent is re-scored at the start of each round on the same episodes, so the comparison is like-for-like. `baseline_raw_score` is 0.0.

**Reference numbers (v0.6.x)**

Measured on the public dev build, live mix, 64 paired episodes (2 seeds); see `CALIBRATION.md` in the repo:

| Policy | Overall | c1 / c2 / c3 / c4 |
| --- | --- | --- |
| zero (no-op) | 0.008 | |
| our hand-coded observation-only controller | 0.018 | |
| strongest insert-only controllers from v0.5 rounds | 0.19 – 0.20 | 0.30 / 0.25 / 0.12 / 0.12 |
| privileged solver (reads the true state; not available to policies) | 0.41 | 0.60 / 0.52 / 0.26 / 0.24 |

The privileged solver is a limited reference. It doesn't adapt to the between-step physics changes, and it seats none of the lying pegs (0 of 64 in a 96-episode sample). Lying pegs are about 13% of inserts, and their seating is not demonstrated on the multi-peg board.

**Miner Submissions**

* A single **ONNX graph** with one input and one output:
  * input `[batch, 288]` float32 (a static `[batch, 184]` input gets the v0.5 prefix)
  * output `[batch, 5]` float32
  * the batch dimension must be **dynamic**.
* Opset **13 – 21**, standard operators only, one self-contained file (no external data).
* Maximum size **50 MB**. The architecture is not constrained.
* Default round length: **1 day**. Submissions are revealed **1 day** after evaluation.
* After each round, every evaluation's records are available: a per-episode record (`nist_lite_record/4`) with the score, the full task graph (`task.steps`, wave times), `conditions.collars`, per-step `step_results`, events, credit by step kind, parts lost, per-peg outcomes, terminal reason and the exact conditions drawn, including the board layout (`conditions.slots`) and peg sizes, plus a full trajectory for every 8th episode (collar poses and parked flags included). The website replay viewer draws each episode's own layout.

**Tips for Miners**

* **The new steps are where the score is.** Collars and twists carry most of the weight. A policy that only inserts pegs, however well, scores about 0.2.
* **Read the graph.** The task table and blocked flags say which steps are open; out-of-order completions earn `max(0.30, 0.15 × in-order credit)`.
* **Expect the physics to change mid-episode.** After a step completes, friction, gain, mass, latency or the estimate bias may shift. Force/torque and the effect of your own actions are the only signals.
* **The headroom is large.** The privileged solver scores about 0.41, and 0.24 on c4, and it is not adaptive. A privileged teacher distilled into an observation-only student, followed by RL, is a reasonable route. Naive behaviour cloning of our solver scores near zero, because the solver relies on true state.
* **Train against the real simulator.** `referee/nist_sim` imports directly: with `NIST_PROFILE=v07` (live), `nist_sim.compound_v07.sample_compound_v07(seed, per_stratum)` draws a round's episodes and `AssemblyIIIEpisode(cond, rng).step(action)` is the exact simulator used for scoring. `NIST_PROFILE=v06` still selects the Assembly II generator. `tools/compound_solver.py --honest` is the privileged reference for calibration bars, and `tools/calib_v06.py --profile v07` scores policies on Assembly III.
* **Use the repo tools.** `tools/local_run.py` scores a policy on any seed, and the README has the exact two-container Docker loop the platform runs.
* **Don't overfit the dev seeds.** Live rounds add private build entropy, and some families and condition ranges are held out entirely.
* The full environment is public at [apex-competition-nist-assembly](https://github.com/macrocosm-os/apex-competition-nist-assembly).
