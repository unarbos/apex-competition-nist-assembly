# NIST Assembly (NIST-Lite): an Apex competition

Train a policy that lets a 2-finger gripper assemble pegs into the NIST Assembly Task Board #1
pattern in simulation, and get credit **only for a completed assembly**. This is the setup popularized by
AlphaNIST (binary task reward, diverse start states). Train any way you like: RL with
success-guided sampling, imitation, scripted teachers plus distillation. Submit the resulting policy as
an ONNX file.

> Version: v0.1.3, the release Apex Subnet 1 runs. Submit with `apex submit policy.onnx -c <competition id>`.

## The task

- **Parts:** round and square pegs in the four NIST ATB-1 sizes (4, 8, 12, 16 mm), each with its
  matching 20 mm-deep hole. Radial clearance is a fraction of a millimetre.
- **Robot:** a parallel 2-finger gripper on a Cartesian stage (x, y, z, yaw). Control runs at 25 Hz and physics at 1 kHz.
- **Start states** (equal share of each, every round): part already grasped above the hole; grasped
  and far from the hole; grasped and resting on the hole rim; standing loose on the board; lying on
  its side on the board.
- **Success (the only thing scored):** within 10 s the part is seated within 1 mm of the
  fully-inserted pose, inside the hole and upright (within 10° of vertical, either end down), and
  the gripper has let go and is clear of the part (fingertips above its top face), held for 5
  consecutive control steps.
- **Score:** `raw_score` = fraction of the round's episodes that end assembled. Higher is better.
  To take the top spot a submission must beat the current top score by 1%; in a round with no top
  scorer, the first scored submission takes it. Rounds last 1 day.
  Each round has 640 episodes (40 shape × size × start-type strata × 16).
- Each round redraws the physical conditions: clearance, friction, part mass, controller stiffness,
  sensor noise, hole placement, and start offsets. A policy has to be robust to all of them.

## Policy interface

One ONNX file (opset 13–21, standard operators only, a single self-contained file, ≤ 50 MB):

| | shape | contents |
|---|---|---|
| input | `[B, 26]` float32 | `[0:3]` gripper pos − hole (×10, m) · `[3:5]` sin/cos gripper yaw · `[5:8]` gripper velocity · `[8]` finger gap (×50) · `[9:12]` part pos − hole top centre (×10) · `[12:16]` part quaternion (w,x,y,z) · `[16:19]` part velocity · `[19]` part width (×50, m) · `[20]` 1 if square · `[21:26]` previous action |
| output | `[B, 5]` float32 in [−1, 1] | `dx, dy, dz` (±4 mm/step), `dyaw` (±0.08 rad/step), gripper (+1 open, −1 close) |

The batch dimension must be dynamic. Each control step is one call covering every episode still running,
with a 500 ms deadline per call. An action that isn't a finite 5-vector ends that episode
unsuccessfully. A call that errors or misses the deadline ends every episode still running.

## Iterate locally

```bash
pip install numpy==2.5.4 mujoco==3.16.0 onnx==1.23.2 onnxruntime==1.31.0
python tools/make_policies.py                       # builds baselines/baseline.onnx
python tools/local_run.py --policy baselines/baseline.onnx --seed 1
python tools/read_records.py out/history            # per-task / per-start-type breakdown
```

The reference baseline (`baselines/baseline.onnx`, a hand-written servo) scores ~0.33. It never
grasps loose or lying parts. To train, import `referee/nist_sim` directly (`pip install -r referee/requirements.txt`). `Episode(cond).step(action)`
is the exact simulator used for scoring, and `nist_sim.conditions.sample_episodes(seed, k)` draws
start states and conditions.

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
| `spec.yaml`, `input.schema.json`, `fixtures/input.json` | The competition spec (`apex.competition.v1`) and round input |
| `player/` | Player image: loads `/app/policy.onnx` and serves it over gym_v1 (vendored `gym_v1/`) |
| `referee/` | Referee image: `nist_sim/` MuJoCo simulation plus `referee.py` scoring (vendored `gym_v1/`) |
| `baselines/`, `test_policies/` | Reference policy and test-set policies (built by `tools/make_policies.py`, `tools/make_overfit.py`) |
| `tools/` | `local_run.py`, `read_records.py`, `make_policies.py` |
| `tests/` | Integration tests (`pytest -q tests/`) |
