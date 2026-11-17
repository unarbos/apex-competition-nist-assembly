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

### Reported 0.36-cap CEM measurements

The following results were reported on **2026-11-17 UTC** for the 0.36-cap
scorer and public +8 mm grasp-bias controller. Confidence intervals are
episode-bootstrap 95% intervals, 10,000 resamples with seed 20261117. They
compare CEM against the simpler `ObsScriptedPolicy`, not the strong script used
by the matched three-round gate below.

| Gate | Episodes | Measured mean [95% CI] | Status |
|---|---:|---:|---|
| Zero | 8 | **0.000000 [0, 0]** | Pass; exactly zero |
| Uniform random | 8 | **0.000000 [0, 0]** | Pass; exactly zero |
| Observation-only scripted (seed 7, public +8 mm) | 64 | **0.011368 [0.008048, 0.014910]** | In 0.01–0.05 band |
| Observation-only scripted (seed 19) | 64 | **0.010184 [0.007143, 0.013490]** | Second referee round |
| CEM grasp/place/lift RL (seed 7) | 64 | **0.014695 [0.011276, 0.018250]** | Paired gain reported |
| CEM grasp/place/lift RL (seed 19) | 64 | **0.014502 [0.011614, 0.017481]** | Paired gain reported |
| Paired RL − scripted (seed 7) | 64 | **Δ +0.003326 [0.000240, 0.006498]** | CI excludes 0 |
| Paired RL − scripted (seed 19) | 64 | **Δ +0.004318 [0.001226, 0.007470]** | CI excludes 0 |
| Pooled paired RL − scripted | 128 | **Δ +0.003822 [0.001511, 0.006082]** | Paired CI excludes 0 |
| Honest privileged (seed 7) | 8 | **0.196647 [0.123477, 0.265505]** | Approximately 0.2 |
| Honest privileged (seed 7) | 64 | **0.135470 [0.112563, 0.159420]** | Below 0.2 on full draw |

The CEM search recovers `θ = [-0.002, 0.028, 0.105]` from the public +8 mm
bias in **4114.2 s** of search/evaluation on train seeds 101–106. Evaluation
uses only observations and actions. This is parameter search around the public
FSM, not from-scratch deep RL. The reported two-round pooled policy intervals
overlap, so the full three-round matched comparison below is the relevant gate
measurement.

### Current 0.36-cap matched evaluation

The strong observation-only script and residual-PPO evaluation use
`tools/obs_boundary_v08.py`, which permits only observation/action/terminal
operations and rejects simulator-state reads. Standalone/zero-residual exact
action-sequence equivalence is covered by tests. Privileged demonstrations are
restricted to training seeds and never available at evaluation.

On the current 0.36-cap scorer, all observation-only policies were measured
through the same fail-closed boundary on **64 episodes per referee seed**, seeds
**7, 19, 31** (192 episodes per policy). The simpler `ObsScriptedPolicy` is the
strongest measured script: **0.009242 [0.007419, 0.011152]**, round means
**0.011770, 0.009212, 0.006744**, sample round standard deviation **0.002513**.
The strong controller scored **0.008735 [0.006988, 0.010604]**, round means
**0.008362, 0.008299, 0.009543**, sample round standard deviation **0.000701**.

The selected CEM policy scored **0.013466 [0.011604, 0.015390]**, round means
**0.013930, 0.015595, 0.010872**, sample round standard deviation **0.002396**.
Against the simpler script, its paired improvement was **+0.004224
[0.002338, 0.006142]**; against the strong controller, **+0.004731
[0.002671, 0.006758]**. Both pooled comparisons are nonoverlapping and both
paired intervals exclude zero. The CEM round standard deviation is smaller than
both policy gaps. **The learned-over-strongest-script gate passes on these
rounds.** CEM is a local learned FSM parameterization, not a from-scratch PPO
result.

The honest privileged reference scored **0.145420 [0.130896, 0.160576]** over
the same 192 episodes, with round means **0.135470, 0.147758, 0.153033** and
sample round standard deviation **0.009012**. It remains above the learned
policy but below the requested approximately **0.2–0.35** calibration range.
The earlier eight-episode value **0.196647 [0.123477, 0.265505]** is not a
substitute for this full-round result.

The following PPO experiment used the earlier 0.30-cap scorer and is retained as
a completed negative result. It is not compared numerically to the 0.36-cap CEM
rows above.

| Policy / check | Episodes | Mean [95% CI] | Round std | Status |
|---|---:|---:|---:|---|
| Strong observation-only script | 24 | **0.008439 [0.004525, 0.012991]** | **0.001968** | Earlier-scorer baseline |
| Learned BC + residual PPO | 24 | **0.004448 [0.002289, 0.006769]** | **0.001359** | Below script; overlapping intervals |
| Honest privileged | 24 | **0.168414 [0.123025, 0.217372]** | **0.024460** | Earlier-scorer calibration |

The PPO run used 29,440 public observation/residual pairs from 32 training
episodes on seeds 101–104. After the observation-derived lift check, an honest
privileged insertion teacher supplied local residual targets; no teacher state
was available at evaluation. BC ran for 15 epochs, followed by **131,072**
residual-PPO timesteps with a c1→full-distribution curriculum. The saved model
was evaluated through the fail-closed public-observation boundary. Training
took **1518.7 s**. Its seed-201 development mean was **0.004791
[0.001580, 0.008423]**, versus the strong script's **0.003530
[0.001200, 0.006135]**; those intervals overlap.

The held-out learned-minus-script paired mean was **-0.003991**, with 95%
interval **[-0.008735, 0.000013]**. The PPO run does not establish improvement.
The collector now matches the script's 160-step insertion limit; this measured
run used an earlier mismatched limit. BC residuals are normalized exactly once
by the collector and scaled once at inference; the former double normalization
is fixed with a regression test.

The earlier whole-trajectory matched PPO run reached **851,968 timesteps**
before its two-hour execution limit without a final model or held-out score.
No claim is made from that incomplete run. A separate lying-grasp script
experiment scored **0.004306 [0.001873, 0.007038]** on 24 earlier-scorer
episodes and was reverted; it is not the current baseline.

Built-image wall measurements are from real host-player/container-referee HTTP
rounds, not local Docker runs. At the CEM commit `1af7d01`, CI runs
**44190044229** and **44190032713** completed 64 episodes on cpuset 0,1 in
**315.467 s** and **316.582 s**, respectively. Both exited 0 with no player
fault. Their `result.json` and `history/episodes.jsonl` were byte-identical.
Earlier runs at `87466bf` and `d05b6e6` measured **158.787 s**, **230.220 s**,
and **228.450 s**. All are under 600 seconds; wall time varies with CI load.
At `bea4f7d` and `1af7d01`, both GitHub pytest runs and both referee-image
runs passed. Current-head checks are tracked on the PR; prior wall results are
not represented as fresh validation of later commits.

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
python tools/eval_ladder_v08.py --kind cem --model /tmp/v08-rl.json \
  --seeds 7,19,31 --per 8 --episodes 64 --out /tmp/v08-cem-matched.json
python tools/eval_ladder_v08.py --kind script --seeds 7,19,31 --per 8 \
  --episodes 64 --out /tmp/v08-script-matched.json
python tools/eval_ladder_v08.py --kind strong_script --seeds 7,19,31 --per 8 \
  --episodes 64 --out /tmp/v08-strong-matched.json
python tools/eval_ladder_v08.py --kind privileged --seeds 7,19,31 --per 8 \
  --episodes 64 --out /tmp/v08-privileged-matched.json
python tools/collect_privileged_demos_v08.py --train-seeds 101,102,103,104 \
  --per 8 --episodes 8 --workers 1 --teacher-mode insert \
  --out /tmp/v08-insert-demos.npz
python tools/train_ppo_v08.py --timesteps 131072 --train-seeds 101,102,103,104 \
  --per 8 --eval-seed 201 --eval-per 8 --eval-episodes 8 --dense --residual \
  --residual-policy strong --curriculum --bc-demos /tmp/v08-insert-demos.npz \
  --bc-teacher privileged --bc-epochs 15 --out /tmp/v08-insert-ppo
python tools/eval_ladder_v08.py --kind ppo --residual-policy strong \
  --model /tmp/v08-insert-ppo/model.zip --seeds 7,19,31 --per 8 --episodes 8 \
  --out /tmp/v08-insert-heldout.json
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

- The CEM result is a local gradient around the public FSM, not from-scratch
  deep RL. The independent 192-episode-per-policy comparison against the strong
  script has nonoverlapping pooled intervals and a paired interval excluding
  zero. The earlier small 24-episode comparison did not establish separation and
  is not used as the final result.
- The completed 131,072-step PPO model is below the strong script under the
  earlier scorer and does not establish learned improvement.
- Honest privileged on the full three-round, 192-episode evaluation measures
  **0.145420 [0.130896, 0.160576]**, below approximately **0.2–0.35**. This is
  an open calibration issue, not a reason to loosen scoring.
- Policy/red-team samples remain small for the matched three-round evaluation.
  The built-image wall measurement is a real 64-episode round but does not
  enlarge the learned-policy sample.
- Long-horizon generalization, unseen blueprints, and stronger adaptive miners
  remain unmeasured. Do not interpret the low scripted baseline as proof of
  frontier-lab hardness.
- Release cosign and digest verification remain owner/CI release tasks. This PR
  does not push images, merge, tag, release, or change digest pins.

### Test verification

The earlier transport/insertion focused suite completed with **22 passed in
229.74 s**. The complete suite before the CEM changes completed with **185
passed, 5 skipped, 1 warning in 3348.35 s**. The current-scorer focused
scorer/audit/wall-harness suite with the CEM-boundary and BC tests completed
with **32 passed in 250.22 s**. Current-head checks are tracked on the PR; no
prior suite is represented as validation of later scorer changes.

## v0.7.5 (historical)

See git history. The v0.8.0 profile and scores are not comparable with v0.7.x.
