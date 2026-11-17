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
- An unfinished, unblocked step can earn at most **0.30 of that step**, not
  additional credit on top of completion: **0.06** for a real grasp/lift,
  **0.06** for held transport toward the real hole, **0.06** for held insertion,
  **0.06** for holding near the target pose, and **0.06** for retaining the pose
  after release through the settle. Transport and insertion have continuous
  best-so-far fractions bounded by **0.60**. Transport requires earned lift,
  closed contacting jaws, and actual part motion toward its target. Insertion
  additionally requires transport, an upright held part, and depth inside the
  actual hole region. Wrist proximity, released/falling parts, oscillation, and
  passive near-hole poses cannot earn these terms. Legacy scalar progress is not
  a scoring input.
- Within the lift bucket, a best-so-far fraction contributes at most **0.012**
  before the binary lift fires. It requires a closed, contacting grasp and a
  measured rise of **0.5–8 mm above the part's settled reset height**. The
  fraction is `0.20 * clip((rise_mm - 0.5) / 7.5, 0, 1)`. Full lift remains
  **0.06**; oscillation, contact alone, wrist motion, absolute spawn height,
  blocked steps, passive poses, and settling frames cannot accumulate credit.
- Blueprint step entries use an unambiguous integer peg id plus separate blocked
  and done flags. The observation remains 320 floats.

### Current transport/insertion measurements

The current code adds the verified transport/insertion terms described above.
The observation-only controller is evaluated through
`tools/obs_boundary_v08.py`, which permits only observation/action/terminal
operations and rejects other episode reads. The standalone strong script and the
zero-residual wrapper are covered by full-episode, exact-action-sequence
equivalence tests. Privileged demonstrations, if used, are restricted to
training seeds and are never available at evaluation.

The following results use referee seeds **7, 19, 31**, eight episodes per seed
(`per=8`), and the current transport/insertion scoring. Confidence intervals
are episode-bootstrap 95% intervals with 10,000 resamples and seed 20261117.
`round std` is the sample standard deviation of the three per-referee-seed
means, not the per-episode standard deviation.

| Policy / check | Episodes | Mean [95% CI] | Round std | Status |
|---|---:|---:|---:|---|
| Zero | 8 (seed 7) | **0.000000 [0, 0]** | not measured | Pass; exactly zero |
| Uniform random | 8 (seed 7) | **0.000000 [0, 0]** | not measured | Pass; exactly zero |
| Strong observation-only script | 24 | **0.008439 [0.004525, 0.012991]** | **0.001968** | Valid observation-only matched baseline |
| Learned BC + residual PPO | — | no completed evaluation | — | **Gate open; training stopped at limit, no final model** |
| Honest privileged | 24 | **0.168414 [0.123025, 0.217372]** | **0.024460** | Above script; below requested 0.2–0.35 range |

The learned-policy gate is not established until a completed learned model
beats the strong script on these held-out seeds with nonoverlapping confidence
intervals. The final comparison also uses a paired episode bootstrap of the
learned-minus-script score differences. The privileged result is measured under
the same current scorer, but remains a true-state calibration reference rather
than an observation-only policy.

The matched BC + residual PPO experiment used 29,440 public observation/action
pairs from 32 privileged-teacher episodes on training seeds 101–104, a
c1→full-distribution curriculum, and the verified observation-only residual
controller. It reached **851,968 timesteps** before its two-hour execution limit
and was stopped before saving a final model or running held-out evaluation. Its
last logged training-round mean reward was **0.00674**, below the script mean;
that is a training metric, not a held-out score. **No learned-policy score or
separation claim is made from this incomplete run.** The training command and
observation-boundary implementation remain reproducible, but a completed model
is still required for the learned-policy gate.

The current red-team run at seed 7 measured **0.000000** for the new open-hand
transport/wrist-spoof and repeated close/release-near-hole attacks. The complete
attack table below includes those results. The complete-suite result is recorded
below. The **built referee image** completed real 64-episode rounds over the
host-player/container-referee HTTP contract in **158.787 s** and **230.220 s**
on cpuset **0,1**. Both containers exited **0**, completed all **64** episodes,
reported `terminal_reason=scored`, and had no player fault. Their `result.json`
and `history/episodes.jsonl` were byte-identical across the two runs; measured
wall times differed, as expected. These pass the **<600 s** built-image and
deterministic-repeat gates and are not in-process timings. The CI artifacts are
`referee-wall-64`, ids **14366028188** and **14367165832**, from runs
**44167711787** and **44167702029** at commit `87466bf`.
The wall harness itself has **2 passing local tests** using the real ONNX player
and a Docker stub: complete-result parsing and timeout/process cleanup. These
are orchestration tests, not built-image wall measurements.

### Historical dense-lift gate table

Measurements below were made on **2026-11-17 UTC** using the earlier 0.18-cap
implementation. They do not include the later transport/insertion evidence and
must not be compared as if they used the current scorer. In-process policy runs
used seed **7**, `per=8`, and the episode counts shown below. Confidence
intervals are episode-bootstrap 95% intervals, 10,000 resamples with seed
20261117. Small samples are not evidence of generalization to unseen seeds. All
table values are measured; none is a projection.

| Gate | Episodes | Measured mean [95% CI] | Status |
|---|---:|---:|---|
| Zero | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Uniform random | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Observation-only scripted | 8 | **0.005876 [0, 0.013878]** | Pass; small physical lift/alignment credit |
| Residual PPO, zero residual initialization | 8, matched to scripted | **0.001588 [0, 0.004597]** | **Ladder not established**: mean is below scripted and intervals overlap |
| Evolutionary residual, accept-only selection | 8, matched to scripted | **0.005876 [0, 0.013878]** | **No improvement**: 0/48 candidates accepted; unchanged scripted incumbent |
| Honest privileged | 8 | **0.196625 [0.123375, 0.265503]** | Pass; upper reference remains well below ceiling |
| Determinism, scripted | 8 + repeat | identical per-episode scores | Pass |
| Determinism, real 64-episode run | 64 + repeat | identical `result.json` and episode records | Pass |
| Real 64-episode wall, 2 pinned CPUs | 64 | **251.9 s**, repeat **256.2 s** | Pass (<600 s), in-process contract |

The denser residual-PPO scores were
`[0, 0.0006716231440830144, 0.012033705297570443, 0, 0, 0, 0, 0]`, on exactly
the same eight held-out episodes as scripted. Training ran for **262,144
timesteps** on seeds 101, 102, and 103 in **1613.3 s**, with **49.5 s** of
held-out evaluation. It uses a zero-initialized residual action mean around the
public scripted controller, bounded residual scale 0.35, and PPO fine-tuning;
evaluation uses only observations and actions. It is not a from-scratch PPO
result. **PPO beating scripted beyond seed noise remains an open gate.** The
pre-densification 262,144-timestep run scored 0.003614 [0, 0.008591] on these
episodes in 1803.0 s of training; densification did not improve this PPO result.
An earlier 32,768-timestep residual run scored 0.007500 on only the first four
held-out episodes and is not used for the matched comparison. A separate
non-residual PPO run of 32,768 timesteps measured 0.0000 on four held-out
episodes, so unaided exploration is not presented as a learnability demonstration.

The accept-only evolutionary residual experiment used 48 candidates (12
generations × population 4), with initial sigma 0.02 decayed by 0.92 and residual
scale 0.35. It used a 3,200-parameter linear residual over the current 320-d
observation and a three-frame observation mean. Selection used eight episodes
from training seeds 101 and 102 plus eight from disjoint validation seeds 201
and 202; reported seed 7 was never used for selection. It retained the exact
zero-residual incumbent unless a candidate strictly improved both training and
validation means. **0/48 candidates were accepted** in **1087.1 s**, across
**408 rollouts**. Its final test result is therefore identical to scripted:
0.005875981718568924 [0, 0.013877945155706771]. The zero-residual path was
verified to reproduce the calibration scores exactly. This method avoids a
regression but does not demonstrate learned improvement.

The honest privileged scores were
`[0.325, 0.307, 0.166, 0.091, 0.005, 0.201, 0.300, 0.178]`.
The privileged reference uses true-state actions and pose-nudging helpers but
does not set completion latches, probe counters, or key-yaw teleports. It is not
an observation-only policy and is not an upper bound on learned performance.

The wall runs used the real player/referee process contract,
`episodes_per_stratum=8`, `workers=2`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `taskset -c 0,1`, with the zero-action ONNX policy.
`result.json` was byte-identical between runs; `history/episodes.jsonl` had the
same SHA-256 on both runs (`fb5355cb8cb21eb9b02fa74681184ccabfbf161fc04b09c4920e1b52736cfa8a`).
The records identify `nist_profile=v08` and log `assembly=IV`; the generic
`assembly=II` field is retained for compound-record compatibility.
**Docker was unavailable**, so these are host in-process measurements, not
built-image measurements. Docker/image validation remains a release-bake task.

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
| Sensor spoofing | **0.000290** [0.000009, 0.000674] | 8 time limits |
| Early termination | 0.000000 | 8 player faults |
| Solver blowup | 0.000000 | 7 unstable, 1 time limit |
| Cue hill-climb sweep | **0.000090** [0, 0.000269] | 8 time limits |
| Open-hand transport/wrist spoof | 0.000000 | 8 time limits |
| Repeated close/release near hole | 0.000000 | 8 time limits |
| Natural-seat key freebie | **0 / 20 keys** | no natural-seat key completed |
| Single-kind cap | analytic cap **0.443478** across 8 blueprints | analytic completed-step bound, not a physical rollout |

The sensor-spoofing and cue-sweep results are below the scripted baseline and
come from small verified sub-lift evidence, not key-window exposure or completed
steps. The single-kind row is an analytic scoring construction, not a rollout
score. General attacks are deliberately not described as proving that all
possible exploits are closed.

### Reproduce

```bash
export NIST_PROFILE=v08 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/make_policies.py
pytest -q tests/
python tools/calib_v08.py --seeds 7 --per 8 --episodes 8 \
  --skip-privileged --out /tmp/v08-calib.json
python tools/compound_solver.py --profile v08 --honest --seed 7 --per 8 --episodes 8
python tools/train_ppo_v08.py --timesteps 262144 --train-seeds 101,102,103 --per 8 \
  --eval-episodes 8 --eval-per 8 --dense --residual --zero-residual \
  --residual-log-std -2.0 --out /tmp/v08-residual
python tools/evolve_residual_v08.py --generations 12 --population 4 \
  --train-seeds 101,102 --validation-seeds 201,202 \
  --train-per 4 --validation-per 4 --train-episodes 4 --validation-episodes 4 \
  --eval-seed 7 --eval-per 8 --eval-episodes 8 --workers 4 --out /tmp/v08-es
```

The optional PPO tooling used `torch 2.15.0+cpu`, `stable-baselines3 2.9.0`,
`gymnasium 1.3.0`, and `scipy 1.17.1`; these are not referee-image dependencies.
Model weights, generated demonstrations, and measurement artifacts are not
committed. No miner ONNX, hotkeys, or secrets are included.

For the real built-image wall check, build the referee image and run the host
player/container harness. The output directory must be writable by the image's
non-root UID:

```bash
docker build -f referee/Dockerfile -t nist-assembly-referee:ci .
mkdir -p /tmp/v08-wall && chmod 0777 /tmp/v08-wall
python tools/docker_wall_v08.py --image nist-assembly-referee:ci \
  --policy baselines/baseline.onnx --seed 7 --cpus 0,1 --out /tmp/v08-wall
```

The older host-only timing command is not a substitute for this built-image
measurement.

### Open risks

- The current learned-policy ladder is not demonstrated beyond seed noise. The
  strong script and privileged reference have three independent referee rounds,
  but a completed learned model must still beat the strong script with
  nonoverlapping held-out intervals. Earlier residual-PPO/evolution results are
  retained below as historical negative results, not evidence of current success.
- The honest privileged pooled mean is **0.168414**, below the requested
  approximately **0.2–0.35** reference range. This is an open calibration issue,
  not a pass by projection or a reason to loosen scoring.
- Policy/red-team samples remain small: eight episodes per referee seed for the
  current matched table and eight for each attack. The separate built-image wall
  measurement is a real 64-episode round and does pass, but it does not enlarge
  the learned-policy sample.
- Release cosign and digest verification remain owner/CI release tasks; this PR
  does not push images, merge, tag, release, or change digest pins.
- Long-horizon generalization, unseen blueprints, and stronger adaptive miners
  remain unmeasured. Do not interpret the low scripted baseline as a proof of
  frontier-lab hardness.

### Test verification

The current transport/insertion focused suite completed with **22 passed in
229.74 s**. It covers physical predecessor gates and evidence bounds,
oscillation/released-part rejection, full-episode fail-closed observation
access, and exact strong-script/zero-residual action equivalence. The complete
repository suite on the same current code completed with **185 passed, 5
skipped, 1 warning in 3348.35 s**. The warning is the pre-existing unregistered
`pytest.mark.slow` mark.

The historical full local suite on the earlier dense-lift implementation
completed with **181 passed, 5 skipped** in **1402.60 s**. The four added tests
cover dense-lift contact/reference gating, prerequisite gating and caps, and
evolutionary zero-residual/seed separation. The pre-existing `pytest.mark.slow`
warning remains.



See git history. The v0.8.0 profile and scores are not comparable with v0.7.x.
