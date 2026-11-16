# NIST-Lite calibration

## v0.8.0 (Assembly IV, profile `v08`)

Spec `0.8.0`. Image bake stays `NIST_PROFILE=v08` (do not regress). **Do not tag /
release / push images / change digest pins from this PR** — digests remain on the last
released v0.7.5 pins until a release bake. Owner bot handles release and Macrocosmos rollout.

Hard reset vs v0.7.x. Closes the v0.7.5 R32 hole classes documented in audit draft PR #40
by design (not as a tweak):

1. **No hill-climbable per-step cues** — key closeness, wrist detent, and retarget pulse removed.
2. **No natural-seat freebie** — key windows anti-aligned from slot symmetries / wrist-zero; no D-flat auto-arm.
3. **No single-subtask dominance** — equal per-step weights; score is a mean over long blueprints.

Reproduce (in-process; same code the image bakes):

```bash
NIST_PROFILE=v08 python tools/calib_v08.py --seeds 7,19 --per 8 --episodes 64
NIST_PROFILE=v08 python tools/compound_solver.py --profile v08 --honest --seed 7 --per 8 --episodes 64
NIST_PROFILE=v08 python tools/obs_scripted_v08.py --seed 7 --per 8 --episodes 64
NIST_PROFILE=v08 python tools/train_rl_v08.py --steps 40 --eval-episodes 8
pytest -q tests/test_v08_assembly.py tests/test_v08_audit_seals.py
```

### Gate table (measured)

All numbers below are **measured** on public seeds 7 and 19 unless noted (UTC).
In-process `NIST_PROFILE=v08` code path. Single-process calib walls; 2-CPU referee wall
measured via `tools/local_run.py`.

| Gate | Target | seed 7 | seed 19 | mean | notes |
|---|---|---:|---:|---:|---|
| honest privileged (actions only) | ≤ ~0.35, > 0 | **0.230** | **0.212** | **0.221** | wall ~306 / 307 s (1 proc) |
| obs-only scripted | ≤ ~0.03 | **0.0060** | **0.0076** | **0.0068** | wall ~336 / 333 s |
| zero | ~0 | **0.0070** | **0.0076** | **0.0073** | wall ~331 / 339 s |
| random (16 eps) | ~0 | **0.0110** | **0.0080** | **0.0095** | short slice |
| short RL (linear REINFORCE, 40 updates) | preferably > scripted | eval **0.0066** (8 eps) | — | ≈ scripted | 237 s train+eval |
| v0.7.x leader ONNX | ≤ ~0.02 or rejected | **rejected** (width 288 ≠ 320) | — | — | `PolicyRejected` |
| determinism | bit-identical | **match** (zero, 8-ep repeat) | — | — | |
| wall 64 eps / 2 CPU | < 600 s | **~224 s projected** from 8 eps @ 28.2 s (`local_run`, workers=2, zero) | — | — | measured 8-ep wall |

### Red-team (seed 7, measured)

General attacks (16 episodes):

| Strategy | mean |
|---|---:|
| jitter | 0.021 |
| vibrate_parts | 0.004 |
| slam_down | 0.008 |
| yaw_spin | 0.008 |
| extreme_actions | 0.010 |

Audit hole-class attacks (PR #40):

| Strategy | result | target |
|---|---|---|
| `cue_hillclimb_sweep` (press + continuous yaw search) | mean **0.009** (16 eps) | near 0 (v0.7.5 cue sweep completed ~58% of keys after seat) |
| `natural_seat_key_freebie` (slot-symmetry yaw + probe/hold) | **0 / 36** keys | 0 |
| `single_kind_dominance_cap` (finish only one kind, analytic) | max **0.454** | ≪ 1, ideally < ~0.55 |

### Notes

- Privileged reference uses `solve_compound(..., oracle=False)`: true-state actions, pose
  nudges for seat/stack/collar/rework, reads true `key_yaw`. Does **not** write `step_done` /
  holds / probe frames or teleport key yaw.
- Obs scripted uses only published observations (including the 32-d blueprint encoding).
- Leader ONNX files were evaluated privately outside the repo and are not committed.
- RL baseline did not clearly beat scripted in the short CPU budget; both sit near zero.
  Sparse end-state credit still provides a small partial-credit gradient (frontier ≤ 0.18).
- Scores are not comparable with v0.7.x.

### Frontier-lab outlook (**estimate**, not measured)

See the PR / agent final report. Rough expectation after weeks of large-scale sim RL +
privileged imitation: gradual climb into roughly the **0.15–0.30** band for top policies,
with 1.0 still far away. This is an **estimate**, not a measured gate.

## v0.7.5 (historical)

See git history. Live as of round 32; audit draft PR #40; superseded by v0.8.0 for difficulty.
