# NIST Assembly (NIST-Lite): an Apex competition

Train a policy that drives a 2-finger gripper through **multi-peg sequential assembly**
on a full NIST Assembly Task Board #1–style peg board in simulation. Credit is dense
partial credit per peg (not binary all-or-nothing), so ranking stays open under the
platform’s 1% takeover rule. Submit the policy as an ONNX file.

> Version: **v0.6.2** ("Assembly II", compound tasks). Every episode is a random multi-step
> assembly graph rather than a list of independent pegs. Peg inserts are joined by collars, which
> fit only over a seated peg, and by twist-lock steps. Random precedence links the steps, parts
> arrive in waves, and the physics changes without notice between subtasks. Episodes come in four
> complexity tiers (c1–c4), and each step earns partial credit. The observation is `[B, 288]`, and
> its first 184 dims keep the v0.5 layout. A policy with a static 184-wide input still runs: the
> player feeds it that prefix. Records are `nist_lite_record/4`. v0.6.1 fixes out-of-order credit,
> collar partial-credit gating, and the record layout for the replay viewer. v0.6.2 fixes stale
> MuJoCo mass constants after a hidden mass change (spurious `unstable` episodes; reported by
> msulab). Scores are not comparable with v0.5.x.
>
> v0.5.1: harder pose estimates (5–11.5 mm peg-vs-hole offset, drift, glitch bursts) and more
> lying starts. v0.5.0: board layout and peg shapes are drawn per episode.

## Relation to AlphaNIST

AlphaNIST (Zhang et al., NVIDIA + UW, [arXiv 2610.13354](https://arxiv.org/abs/2610.13354))
trains one PPO policy in Isaac Lab over all 20 ATB-1 tasks (pegs, rods, gears, connectors,
nut threading) with a binary success reward (part within 1 mm), a cached reset bank,
success-guided Beta sampling, point-cloud observations, a 7-DoF Franka at 25 Hz, and up
to 256K GPU envs; 98.2% training success, and 84% on 10-pair sequences with no prescribed
order. All of it is in simulation.

Ours is much smaller: **CPU MuJoCo**, peg family only (no gears, connectors, or threads),
primitive geometry, a **4-DoF Cartesian gripper** instead of a Franka arm, **partial /
biased state observations** (not exact poses, not point clouds), at most **6 pegs**, and
**partial-credit scoring** instead of a single terminal reward (needed for a ranked
competition). We borrow the board, the 1 mm success criterion, the 25 Hz control, the
diverse start states, and free-order multi-part episodes. Neither project is real-robot
evidence.

**Meant to reward learning.** v0.6 follows a Macrocosmos request for more randomness and a
very high ceiling. The tasks are compound, so a fixed insert-only controller tops out at about
0.20, and our privileged solver scores 0.41 overall and 0.24 on the hardest tier. On v0.5.0, well-tuned hand-coded controllers reached about
0.92, above our privileged solver. v0.5.1 makes the observation harder to trust:
peg/slot channels carry per-episode pose-estimate offsets of 5–11.5 mm and several degrees
(independent per slot, clutter block and peg), plus drift, noise, glitch bursts and dropout,
and there is no fixed board layout or shape list to hard-code against. Wrist
force/torque is provided so contact-driven search can adapt. Some cross-section
families are held out of live rounds. See [CALIBRATION.md](CALIBRATION.md).

## The task (v0.6.2)

- **Board:** one MuJoCo board per episode with 8 peg slots. Every episode draws its own
  layout: slot positions, slot yaw, and a cross-section per slot from continuous families
  (round, rectangle/square, regular 5/6/8-gon, D-flat; inscribed size 4–16 mm). The board pose
  is jittered, and held-out families appear only on the reserved split.
- **Parts:** 3–6 pegs by stratum, plus up to 2 extra on c3/c4 (at most 8). Pegs start loose
  upright, lying, in a tray, near their hole, or (at most one) pre-grasped. There are 0–3
  clutter blocks.
- **Steps:**
  - **insert** puts a peg into its matching slot.
  - **collar** drops a ring (inner clearance 0.6–1.2 mm) over a peg that is already seated,
    so the ring rests on the slot block.
  - **twist** turns a seated round peg by a random 1.2–2.0 rad in a given direction (a
    bayonet-style lock); this can take several regrasps.
- **Precedence:** a peg must be seated before its collar or twist. On top of that, random
  cross-part edges are added, denser on harder tiers. Each step's blocked flag is in the
  observation.
- **Waves:** parts arrive in 1–4 waves. A wave arrives when the earlier waves are done or
  at a random deadline. Parts that haven't arrived are off the board and zeroed in the
  observation.
- **Between-subtask changes:** after a step completes, a random event may fire: friction
  or controller gain change, part mass changes, an actuation delay appears, pose estimates
  re-bias, or a loose part is shoved. The policy is not told when this happens.
- **Tiers** (15/30/30/25% of each stratum): c1 has one wave and few compound steps. c4 has
  +2 pegs, 3–4 waves, a dense graph, events after most steps and the least time slack.
  Every episode has at least one collar or twist.
- **Horizon:** the sum of per-step budgets (insert 260, +140 for a lying peg, collar 200,
  twist 130 control steps), times the tier slack (1.2 / 1.0 / 0.85 / 0.70), plus 40 per extra
  wave. The mean is about 1,800 steps (72 s). The episode ends early once every step is done
  and the gripper is clear.
- **Hidden dynamics:** carried over from v0.5.1. Friction, mass, gain, backlash,
  pose-estimate bias and drift, noise and dropout, and glitch bursts are all drawn per
  episode; glitch bursts are softened to half rate on v0.6. Glitch bursts cover about 25% of
  control steps on the live mix (easy 16%, medium 25%, hard 29%).
- **Score:** each step `k` has weight `w_k`: insert 0.3, collar 1.5, twist 1.0.
  - A step done in order and still done at the end earns `0.70 + 0.30 · (1 − t_done/T)`.
  - A step done while its prerequisites were still open earns
    `max(0.30, 0.15 × in-order credit)` (never less than unfinished partial).
  - An unfinished step earns up to 0.30 for partial progress (grasp/lift, alignment,
    partial insertion, collar around a seated peg, fraction twisted). Collar 0.20 / 0.30
    tiers require that collar's peg in its hole.
  - The episode score is `Σ w_k c_k / Σ w_k`; the round score is the mean over episodes.
  - A lost part (off the board) no longer ends the episode. Its steps just can't finish.
- **Round size:** 8 strata (K × clutter). The fixture uses 8 per stratum (64 episodes); the
  live rounds' episode count is set by the platform round input (see `HANDOFF.md` §4). Timeout
  1800 s.

## Policy interface

One ONNX file (opset 13–21, standard operators only, a single self-contained file, ≤ 50 MB):

| | shape | contents |
|---|---|---|
| input | `[B, 288]` float32 (or `[B, 184]`, see below) | See layout below |
| output | `[B, 5]` float32 in [−1, 1] | `dx, dy, dz` (±4 mm/step), `dyaw` (±0.08 rad/step), gripper (+1 open, −1 close) |

If the policy's input is statically 184 wide, the player passes only the first 184 dims (the
v0.5 layout); every other policy gets all 288.

### Observation layout (`OBS_DIM = 288`)

1. **Proprio (16):** hand `(x,y,z)` relative to board origin ×10; `sin/cos` yaw; hand
   velocity (3); finger gap ×50; **wrist force (3)**, **wrist torque (3)**, pad force (1).
2. **Last action (5).**
3. **Peg table (8 × 17 = 136):** for each of 8 slot rows (inactive rows and parts that haven't
   arrived are zeros): presence; **biased/noisy** peg pose relative to the **biased** slot
   estimate ×10 (3) + quaternion (4); **biased** slot-yaw estimate as `sin, cos` (2); outer
   (circumscribed) size ×50 (1); a zero (1); inscribed size ×50 (1); biased slot position
   relative to board ×10 (3); seated flag. No shape label, and exact poses are never given.
4. **Clutter table (3 × 9 = 27):** presence; biased pose relative to board ×10 (3) +
   quaternion (4); size ×50.
5. **Task table (8 × 6 = 48), one row per peg slot:** insert done, insert blocked, twist
   direction (±1, 0 = no twist), twist progress (0–1), twist blocked, has-collar flag.
6. **Collar table (4 × 13 = 52):** presence (0 until it arrives); noisy collar position
   relative to the biased target-slot estimate ×10 (3) + quaternion (4); inner and outer
   radius ×100 (2); target peg row / 8; done; blocked.
7. **Globals (4):** elapsed fraction of the horizon, fraction of steps done, fraction of waves
   arrived, number of steps / 16.

The batch dimension must be dynamic. Each control step is one call covering every episode
still running, with a 500 ms deadline per call. A non-finite or wrong-shaped action ends
that episode; done steps keep their credit. A call that errors or misses the deadline ends
every episode still running the same way.

## Iterate locally

```bash
pip install numpy==2.5.4 mujoco==3.16.0 onnx==1.23.2 onnxruntime==1.31.0
python tools/make_policies.py                       # builds baselines/baseline.onnx
python tools/local_run.py --policy baselines/baseline.onnx --seed 1
python tools/read_records.py out/history
python tools/write_calibration.py                   # CALIBRATION.md (tiered live mix)
python tools/compound_solver.py --seed 0           # privileged compound solver (v0.6)
python tools/calib_v06.py --profile v06 --seeds 1 --per 2 --policies reference zero  # v0.6 calibration
python tools/train_ppo.py --steps 100000 --workers 2  # residual PPO learnability
```

Local runs use the public dev build. Live rounds also mix private build entropy into the
episode draw, so a local `--seed` has the same distribution as a live round but does not
reproduce it.

Every live round mixes ~25% easy / 35% medium / 40% hard (`difficulty_tier`).
The reference baseline is a soft-nearest-first insert/release servo on the biased obs.
`ObsOnlyScripted` is the strongest hand-coded obs-only controller (anti-scripting
gate: EMA + F/T spiral + jam retry). `tools/train_ppo.py` is the residual-PPO
learnability harness (scripted base + privileged demos; curve in `CALIBRATION.md`).
The privileged solver in `tools/reference_solver.py` is **not** shipped in the player.
See `CALIBRATION.md` for measured scores and wall times.

### Exact Docker loop (the same contract the platform runs)

```bash
docker build -f player/Dockerfile  -t nist-assembly-player  .
docker build -f referee/Dockerfile -t nist-assembly-referee .
docker network create nist-net
docker run -d --name nist-player --network nist-net --cpus 2 --memory 2g \
  -v "$PWD/baselines/baseline.onnx:/app/policy.onnx:ro" nist-assembly-player
docker run --rm --network nist-net --cpus 2 --memory 2g \
  -e MATCH_ID=local -e SEED=1 -e NUM_PLAYERS=1 -e PLAYER_URLS=http://nist-player:8000 \
  -e CONFIG_JSON="$(cat fixtures/input.json)" -v "$PWD/out:/data" nist-assembly-referee
cat out/result.json; docker rm -f nist-player; docker network rm nist-net
```

## Layout

| Path | What |
|---|---|
| `spec.yaml`, `input.schema.json`, `fixtures/input.json` | Competition spec (`apex.competition.v1`) and round input |
| `player/` | Player image: loads `/app/policy.onnx` over gym_v1 |
| `referee/` | Referee image: `nist_sim/` MuJoCo sim + scoring |
| `baselines/`, `test_policies/` | Reference + junk ONNX policies (`tools/make_policies.py`) |
| `tools/` | `local_run.py`, `read_records.py`, `make_policies.py`, `calibrate.py` |
| `tests/` | Integration tests (`pytest -q tests/`) |
| `CALIBRATION.md` | Junk / baseline / privileged reference numbers across seeds |
| `docs/` | Apex competition description (`description.md`) and docs page text (`competition.md`) |
