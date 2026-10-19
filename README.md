# NIST Assembly (NIST-Lite): an Apex competition

Train a policy that drives a 2-finger gripper through **multi-peg sequential assembly**
on a full NIST Assembly Task Board #1–style peg board in simulation. Credit is dense
partial credit per peg (not binary all-or-nothing), so ranking stays open under the
platform’s 1% takeover rule. Submit the policy as an ONNX file.

> Version: **v0.2.0** (multi-peg board). Spec digests are left as-is until the maintainer
> release; do not treat pinned digests as shipping this code yet.

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

**v0.2 requires learning.** A hand-coded controller that trusts the observation vector
cannot win: peg/slot channels carry a fixed per-episode pose bias of several mm and
degrees, plus noise and dropout. Wrist force/torque is provided so contact-driven search
can adapt. Hex and D-flat keyed shapes are held out of live rounds.

## The task (v0.2.0)

- **Board:** one MuJoCo board per episode with 8 peg slots (round + square × 4/8/12/16 mm
  live). Board pose `(x, y, yaw)` is jittered per episode. Held-out shape families
  (hex, D-flat keyed) appear only on the reserved split.
- **Active pegs:** `K ∈ {3..6}` by stratum (mean 4.5). Pegs start loose upright, lying,
  in a tray/cluster, near-hole, or (at most one) pre-grasped.
- **Free order:** the policy chooses the sequence. Each peg must go into its **matching**
  slot (shape + size). Wrong slot = 0. A seated peg later knocked out = 0 at episode end.
- **Clutter:** 0–3 free distractor blocks; sometimes blocking a slot approach.
- **Fits:** radial clearance **0.25–0.8 mm** with variable entry chamfer (contact
  `solimp` width 0.02 mm). Tight fits need wiggle / spiral search.
- **Horizon:** 11.2 s (280 control steps @ 25 Hz) **per active peg**. Early end when all
  pegs are seated and the gripper is clear.
- **Hidden dynamics (LHS):** friction, mass, gain, mild backlash, pose-estimate bias,
  noise/dropout, rare act-delay, mid-episode shove or gripper drop.
- **Score:** for episode `e` with `K` pegs,
  `S_e = (1/K) Σ_k s_k`, where `s_k = 0` if peg `k` is not seated at end, else
  `0.75 + 0.25 · time/effort term`. Partial credit rewards even one or two seated pegs.
- **Round size:** 64 episodes (8 strata × 8) by default; 2 sim workers; timeout 900 s.

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
   quaternion (4); shape one-hot (round, square, hex, dflat); size ×50; biased slot
   position relative to board ×10 (3); seated flag. Exact hole/peg poses are never given.
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
python tools/calibrate.py                           # CALIBRATION.md (anti-scripting)
python tools/reference_solver.py --seed 0           # privileged solvability check
python tools/quick_learn.py                         # short hill-climb vs obs scripted
```

The reference baseline is a soft-nearest-first insert/release servo on the biased obs.
`ObsOnlyScripted` is the strongest hand-coded obs-only controller (anti-scripting gate).
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
