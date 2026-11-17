# NIST-Lite calibration

## v0.8.0 (Assembly IV, profile `v08`)

Spec `0.8.0`. Image bake stays `NIST_PROFILE=v08`. **Do not tag, release, push
images, or change release digest pins from this PR.** The existing pins remain
on v0.7.5 until the owner performs a release bake.

Assembly IV uses equal per-step weights, no key closeness/detent/retarget cues,
anti-aligned key windows, procedural blueprints, dynamics randomization, and a
320-d observation contract. Scores are not comparable with v0.7.x.

### Scoring and no-free-credit rules

- Parts start ungrasped and away from their target holes, including extra pegs
  added by graph generation. Reset-only states and passive end poses score zero.
- Completed steps require a physical latch during an action transition and an
  end-state that survives the perturbation settle. Finalization does not invent
  completion from a passive pose.
- An unfinished, unblocked step can earn at most **0.36 of that step**, not
  additional credit on top of completion: **0.12** for a real grasp/lift,
  **0.06** for held transport toward the real hole, **0.06** for held insertion,
  **0.06** for holding near the target pose, and **0.06** for retaining the pose
  after release through the settle. Transport and insertion have continuous
  best-so-far fractions bounded by **0.60**. Transport requires earned lift,
  closed contacting jaws, and actual part motion toward its target. Insertion
  additionally requires transport, an upright held part, and depth inside the
  actual hole region. Wrist proximity, released/falling parts, oscillation, and
  passive near-hole poses cannot earn these terms. Legacy scalar progress is not
  a scoring input.
- Within the lift bucket, a best-so-far fraction contributes at most **0.024**
  before the binary lift fires. It requires a closed, contacting grasp and a
  measured rise of **0.5–8 mm above the part's settled reset height**. The
  fraction is `0.20 * clip((rise_mm - 0.5) / 7.5, 0, 1)`. Full lift remains
  **0.12**; oscillation, contact alone, wrist motion, absolute spawn height,
  blocked steps, passive poses, and settling frames cannot accumulate credit.
- Blueprint step entries use an unambiguous integer peg id plus separate blocked
  and done flags. The observation remains 320 floats.

### Measured gate table

Measurements were made on **2026-11-17 UTC** on the final follow-up working tree
(lift credit **0.12**, public grasp bias **+8 mm**). Confidence intervals are
episode-bootstrap 95% intervals, 10,000 resamples with seed 20261117. All table
values are **measured** (none projected).

| Gate | Episodes | Measured mean [95% CI] | Status |
|---|---:|---:|---|
| Zero | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Uniform random | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Observation-only scripted (seed 7, public +8 mm) | 64 | **0.011368 [0.008048, 0.014910]** | Pass; in 0.01–0.05 band |
| Observation-only scripted (seed 19) | 64 | **0.010184 [0.007143, 0.013490]** | Pass; held-out seed |
| CEM grasp/place/lift RL (seed 7) | 64 | **0.014695 [0.011276, 0.018250]** | Pass; beats scripted |
| CEM grasp/place/lift RL (seed 19) | 64 | **0.014502 [0.011614, 0.017481]** | Pass; held-out seed |
| Paired RL − scripted (seed 7) | 64 | **Δ +0.003326 [0.000240, 0.006498]** | Pass; CI excludes 0 (30W/22L/12T) |
| Paired RL − scripted (seed 19) | 64 | **Δ +0.004318 [0.001226, 0.007470]** | Pass; CI excludes 0 |
| Pooled paired RL − scripted | 128 | **Δ +0.003822 [0.001511, 0.006082]** | Pass; margin beyond seed noise |
| Honest privileged (seed 7) | 8 | **0.196647 [0.123477, 0.265505]** | Pass; in ~0.2–0.35 band |
| Honest privileged (seed 7) | 64 | **0.135470 [0.112563, 0.159420]** | Below 0.2 on full draw; still ≫ RL and ≪ 1.0 |
| Determinism, scripted | 8 + repeat | identical per-episode scores | Pass |
| Built-image 64-episode wall, cpuset 0,1 | 64 + repeat | **158.787 s**, repeat **230.220 s** | Pass (<600 s), container HTTP contract |
| Built-image determinism | 64 + repeat | identical `result.json` / `episodes.jsonl` | Pass |

Learnable gradient: the public observation-only scripted controller ships with a
fixed **+8 mm** mid-shaft grasp-height bias (`GRASP_Z_BIAS_M = 0.008`). A
black-box CEM search over metre-space knobs
`θ = [grasp_bias_m, place_z_m, lift_z_m]` (train seeds 101–106; never trained on
eval seeds 7/19) recovers **θ = [-0.002, 0.028, 0.105]**. Search + held-out
evaluation wall for that block was **4114.2 s** (~1.14 CPU-hours). Evaluation
uses only observations and actions. This is parameter search around the public
FSM, not a from-scratch deep RL miner; residual-PPO tooling remains in-tree.

Honest privileged uses true-state actions and pose-nudging helpers but does not
set completion latches, probe counters, or key-yaw teleports. The 8-episode slice
lands in the ~0.2–0.35 target band; the full 64-episode draw is lower because
harder strata finish fewer in-order steps, while remaining far above the learned
observation-only policies.

Built-image wall: the **built referee image** completed real 64-episode rounds
over the host-player/container-referee HTTP contract in **158.787 s** and
**230.220 s** on cpuset **0,1**. Both containers exited **0**, completed all
**64** episodes, reported `terminal_reason=scored`, and had no player fault.
Their `result.json` and `history/episodes.jsonl` were byte-identical across the
two runs; measured wall times differed, as expected. CI artifacts
`referee-wall-64`, ids **14366028188** and **14367165832**, from runs
**44167711787** and **44167702029** at commit `87466bf`.

### Red team (final code, seed 7, 8 episodes per attack)

All scores below are measured means over eight episodes unless noted.

| Attack | Score / result | Terminal behavior |
|---|---:|---|
| Jitter | 0.000000 | 8 time limits |
| Vibrate parts | 0.000000 | 8 time limits |
| Slam down | 0.000000 | 8 time limits |
| Yaw spin | 0.000000 | 8 time limits |
| Extreme actions | 0.000000 | 8 time limits |
| NaN actions | 0.000000 | 8 time limits |
| Sensor spoofing | **0.000580** [0.000017, 0.001347] | 8 time limits |
| Early termination | 0.000000 | 8 player faults |
| Solver blowup | 0.000000 | 7 unstable, 1 time limit |
| Cue hill-climb sweep | **0.000179** [0, 0.000538] | 8 time limits |
| Open-hand transport/wrist spoof | 0.000000 | 8 time limits |
| Repeated close/release near hole | 0.000000 | 8 time limits |
| Natural-seat key freebie | **0 / keys** | no natural-seat key completed |
| Single-kind cap | analytic cap **0.443478** across 8 blueprints | analytic completed-step bound, not a physical rollout |

Jitter and vibrate attacks against the densified partials score exactly zero.
The tiny sensor-spoofing and cue-sweep values are bounded sub-lift fractions, far
below the scripted baseline, not key-window exposure or completed steps. The
single-kind row is an analytic scoring construction, not a rollout score.
General attacks are deliberately not described as proving that all possible
exploits are closed.

### Reproduce

```bash
export NIST_PROFILE=v08 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/make_policies.py
pytest -q tests/
python tools/calib_v08.py --seeds 7 --per 8 --episodes 8 \
  --skip-privileged --out /tmp/v08-calib.json
python tools/compound_solver.py --profile v08 --honest --seed 7 --per 8 --episodes 8
python tools/train_rl_v08.py --train-seeds 101,102,103,104,105,106 \
  --train-per 4 --train-episodes 4 --eval-seed 7 --eval-per 8 --eval-episodes 64 \
  --candidates 12 --polish-gens 4 --polish-pop 8 --seed 2 --out /tmp/v08-rl.json
```

Optional residual-PPO / evolution tooling remains available under `tools/`; those
packages are not referee-image dependencies. Model weights, generated
demonstrations, and measurement artifacts are not committed. No miner ONNX,
hotkeys, or secrets are included.

For the real built-image wall check:

```bash
docker build -f referee/Dockerfile -t nist-assembly-referee:ci .
mkdir -p /tmp/v08-wall && chmod 0777 /tmp/v08-wall
python tools/docker_wall_v08.py --image nist-assembly-referee:ci \
  --policy baselines/baseline.onnx --seed 7 --cpus 0,1 --out /tmp/v08-wall
```

### Open risks

- Honest privileged on the full 64-episode draw measures **0.135**, below the
  ~0.2–0.35 8-episode band, because harder strata finish fewer in-order steps.
  The observation-only ladder (zero < scripted < RL) remains intact on 64.
- The CEM result shows a learnable local gradient around the public FSM; it is
  not a claim that from-scratch deep RL has been solved for Assembly IV.
- Long-horizon generalization, unseen blueprints, and stronger adaptive miners
  remain unmeasured.

## v0.7.5 (historical)

See git history. The v0.8.0 profile and scores are not comparable with v0.7.x.
