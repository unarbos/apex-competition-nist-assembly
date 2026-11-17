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
- An unfinished, unblocked step can earn **0.06** for a real grasp/lift, **0.06**
  for holding that part near its target pose, and **0.06** for retaining the pose
  after release through the settle: at most **0.18 of that step**, not additional
  credit on top of completion. Legacy scalar progress is not a scoring input.
- Blueprint step entries use an unambiguous integer peg id plus separate blocked
  and done flags. The observation remains 320 floats.

### Measured gate table

Measurements were made on **2026-11-17 UTC**, using the final follow-up working
tree. In-process policy runs used seed **7**, `per=8`, and the episode counts
shown below. Confidence intervals are episode-bootstrap 95% intervals, 10,000
resamples with seed 20261117. Small samples are not evidence of generalization
to unseen seeds. All table values are measured; none is a projection.

| Gate | Episodes | Measured mean [95% CI] | Status |
|---|---:|---:|---|
| Zero | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Uniform random | 8 | **0.000000 [0, 0]** | Pass; every episode exactly zero |
| Observation-only scripted | 8 | **0.005795 [0, 0.013636]** | Pass; small physical lift/alignment credit |
| Residual PPO, zero residual initialization | 8, matched to scripted | **0.003614 [0, 0.008591]** | **Ladder not established**: mean is below scripted and intervals overlap |
| Honest privileged | 8 | **0.199250 [0.124000, 0.270878]** | Pass; upper reference remains well below ceiling |
| Determinism, scripted | 8 + repeat | identical per-episode scores | Pass |
| Determinism, real 64-episode run | 64 + repeat | identical `result.json` and episode records | Pass |
| Real 64-episode wall, 2 pinned CPUs | 64 | **416.3 s**, repeat **404.2 s** | Pass (<600 s), in-process contract |

The longer residual-PPO scores were
`[0, 0.010909090909090908, 0.018, 0, 0, 0, 0, 0]`, on exactly the same eight
held-out episodes as scripted. Training ran for **262,144 timesteps** on seeds
101, 102, and 103 in **1803.0 s**, with **48.2 s** of held-out evaluation. It
uses a zero-initialized residual action mean around the public scripted
controller, bounded residual scale 0.35, and PPO fine-tuning; evaluation uses
only observations and actions. It is not a from-scratch PPO result. **PPO beating
scripted beyond seed noise remains an open gate.** An earlier 32,768-timestep
residual run on seeds 101 and 102 scored 0.007500 [0, 0.022500] on only the
first four held-out episodes in 863.3 s of training, so it is not used for the
matched comparison. A separate dense-shaped, non-residual PPO run of 32,768
timesteps measured 0.0000 on four held-out episodes (609.4 s training), so
unaided exploration is not presented as a learnability demonstration.

The honest privileged scores were
`[0.335, 0.318, 0.166, 0.091, 0.005, 0.201, 0.300, 0.178]`.
The privileged reference uses true-state actions and pose-nudging helpers but
does not set completion latches, probe counters, or key-yaw teleports. It is not
an observation-only policy and is not an upper bound on learned performance.

The wall runs used the real player/referee process contract,
`episodes_per_stratum=8`, `workers=2`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `taskset -c 0,1`, with the zero-action ONNX policy.
`result.json` was byte-identical between runs; `history/episodes.jsonl` had the
same SHA-256 on both runs (`c23fcb4f62a8743f8f50f5a0dddab48d1a105888f9b1fb6c1b897b215edc80e1`).
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
| Sensor spoofing | **0.000682** [0, 0.002045] | 8 time limits |
| Early termination | 0.000000 | 8 player faults |
| Solver blowup | 0.000000 | 7 unstable, 1 time limit |
| Cue hill-climb sweep | 0.000000 | 8 time limits |
| Natural-seat key freebie | **0 / 20 keys** | no natural-seat key completed |
| Single-kind cap | analytic cap **0.443478** across 8 blueprints | analytic completed-step bound, not a physical rollout |

The sensor-spoofing result is below the scripted baseline and does not expose a
key window. The single-kind row is an analytic scoring construction, not a
rollout score. General attacks are deliberately not described as proving that
all possible exploits are closed.

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
```

The optional PPO tooling used `torch 2.15.0+cpu`, `stable-baselines3 2.9.0`,
`gymnasium 1.3.0`, and `scipy 1.17.1`; these are not referee-image dependencies.
Model weights, generated demonstrations, and measurement artifacts are not
committed. No miner ONNX, hotkeys, or secrets are included.

For the real wall check, run the following twice on the same two CPUs and
compare `result.json` and `history/episodes.jsonl`:

```bash
taskset -c 0,1 python tools/local_run.py --policy baselines/baseline.onnx \
  --seed 7 --input fixtures/input.json --out /tmp/v08-wall \
  --override '{"episodes_per_stratum":8,"workers":2}'
```

### Open risks

- The RL ladder is not demonstrated beyond seed noise. The matched residual-PPO
  mean is below scripted on eight held-out episodes, with overlapping intervals.
- The policy/red-team tables use eight-episode slices, not the complete 64-episode
  policy ladder or additional held-out seeds. The wall gate itself was measured
  on a real 64-episode run and repeated.
- No Docker daemon was available locally. Image-level wall, cosign, and release
  digest verification must be performed by the owner/CI without changing pins in
  this PR.
- Long-horizon generalization, unseen blueprints, and stronger adaptive miners
  remain unmeasured. Do not interpret the low scripted baseline as a proof of
  frontier-lab hardness.

### Test verification

The full local suite on implementation commit
`357727675e0a2b9c69af3a7d154b24996065cc0a` completed with **178 passed,
5 skipped** in **2077.94 s**. The subsequent calibration-only edit does not
change simulation, policy, or test code.



See git history. The v0.8.0 profile and scores are not comparable with v0.7.x.
