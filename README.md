# NIST Assembly (NIST-Lite): an Apex competition

Train a policy that drives a 2-finger gripper through **multi-peg sequential assembly**
on a full NIST Assembly Task Board #1–style peg board in simulation. Credit is dense
partial credit per peg (not binary all-or-nothing), so ranking stays open under the
platform’s 1% takeover rule. Submit the policy as an ONNX file.

> Version: **v0.5.1** (observation hardening). Same task, observation layout and scoring
> as v0.5.0, but harder observations: the extra peg-vs-hole estimate offset is now 5–11.5 mm
> (was 1.5–5 mm), pose noise 0.8–3.5 mm (was 0.4–2.0 mm), estimate drift is faster and
> present on every tier, short estimate "glitch" bursts add extra noise and a transient
> offset to every pose estimate, and lying-down starts are more common (hard 22%, medium 12%).
> Also fixes from the v0.5.0 validation notes (layout and spawn edge cases, pre-grasped hand
> yaw near ±π, records carry both the effective and the drawn start type). Scores are not
> directly comparable with v0.5.0. Input/output shapes (`[B, 184]` -> `[B, 5]`), records
> (`nist_lite_record/2`, peg entries add `drawn_start_type`) and `timeout_s` (1800 s) are unchanged.
>
> v0.5.0: board layout and peg shapes are drawn per episode; peg-table features 8–11 have no
> shape label.

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

**Meant to reward learning.** On v0.5.0, well-tuned hand-coded controllers reached about
0.92, above our privileged solver. v0.5.1 makes the observation harder to trust:
peg/slot channels carry per-episode pose-estimate offsets of 5–11.5 mm and several degrees
(independent per slot, clutter block and peg), plus drift, noise, glitch bursts and dropout,
and there is no fixed board layout or shape list to hard-code against. Wrist
force/torque is provided so contact-driven search can adapt. Some cross-section
families are held out of live rounds. See [CALIBRATION.md](CALIBRATION.md).

## The task (v0.5.1)

- **Board:** one MuJoCo board per episode with 8 peg slots. Every episode draws its own
  layout: slot positions (non-overlapping, anywhere on the board), slot yaw, and the peg
  cross-section of each slot from continuous families (round, rectangle/square, regular
  5/6/8-gon, D-flat) with inscribed size 4–16 mm. Board pose `(x, y, yaw)` is jittered per
  episode. Held-out families appear only on the reserved split.
- **Active pegs:** `K ∈ {3..6}` by stratum (mean 4.5). Pegs start loose upright, lying,
  in a tray/cluster, near-hole, or (at most one) pre-grasped.
- **Free order:** the policy chooses the sequence. Each peg must go into its **matching**
  slot (its own row in the peg table). Wrong slot = 0. A seated peg later knocked out = 0 at episode end.
- **Clutter:** 0–3 free distractor blocks; sometimes blocking a slot approach.
- **Fits:** radial clearance **0.25–0.8 mm** with variable entry chamfer (contact
  `solimp` width 0.02 mm). Tight fits need wiggle / spiral search.
- **Horizon:** 11.2 s (280 control steps @ 25 Hz) **per active peg**. Early end when all
  pegs are seated and the gripper is clear.
- **Hidden dynamics (LHS):** friction, mass, gain, mild backlash, pose-estimate bias and
  drift, noise/dropout, estimate glitch bursts, rare act-delay, mid-episode shove or gripper drop.
- **Score:** for episode `e` with `K` pegs,
  `S_e = (1/K) Σ_k s_k`, where `s_k = 0` if peg `k` is not seated at end, else
  `0.75 + 0.25 · time/effort term`. Partial credit rewards even one or two seated pegs.
- **Round size:** 64 episodes (8 strata × 8) in the fixture; live rounds use 16 per stratum (128 episodes); 2 sim workers; timeout 1800 s.

## Policy interface

One ONNX file (opset 13–21, standard operators only, a single self-contained file, ≤ 50 MB):

| | shape | contents |
|---|---|---|
| input | `[B, 184]` float32 | See layout below |
| output | `[B, 5]` float32 in [−1, 1] | `dx, dy, dz` (±4 mm/step), `dyaw` (±0.08 rad/step), gripper (+1 open, −1 close) |

### Observation layout (`OBS_DIM = 184`)

1. **Proprio (16):** hand `(x,y,z)` relative to board origin ×10; `sin/cos` yaw; hand
   velocity (3); finger gap ×50; **wrist force (3)**, **wrist torque (3)**, pad force (1).
2. **Last action (5).**
3. **Peg table (8 × 17 = 136):** for each of 8 slot rows (inactive rows are zeros):
   presence; **biased/noisy** peg pose relative to **biased** slot estimate ×10 (3) +
   quaternion (4); **biased** slot-yaw estimate as `sin, cos` (2; a peg is aligned when its
   yaw matches the slot yaw, modulo the unknown symmetry of its cross-section); outer
   (circumscribed) size ×50 (1); a zero (1); inscribed size ×50 (1); biased slot position
   relative to board ×10 (3); seated flag. No shape label. Exact hole/peg poses are never
   given. Before v0.5.0, features 8–11 were a shape one-hot.
4. **Clutter table (3 × 9 = 27):** presence; biased pose relative to board ×10 (3) +
   quaternion (4); size ×50.

The batch dimension must be dynamic. Each control step is one call covering every episode
still running, with a 500 ms deadline per call. A non-finite or wrong-shaped action ends
that episode (remaining unseated pegs score 0). A call that errors or misses the deadline
ends every episode still running the same way.

## Iterate locally

```bash
pip install numpy==2.5.4 mujoco==3.16.0 onnx==1.23.2 onnxruntime==1.31.0
python tools/make_policies.py                       # builds baselines/baseline.onnx
python tools/local_run.py --policy baselines/baseline.onnx --seed 1
python tools/read_records.py out/history
python tools/write_calibration.py                   # CALIBRATION.md (tiered live mix)
python tools/reference_solver.py --seed 0           # privileged solvability check
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
