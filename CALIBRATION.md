# NIST-Lite calibration

## v0.7.5 (observation-completable keys, profile `v07`)

Spec `0.7.5`. Image bake stays `NIST_PROFILE=v07` (do not regress). Digests on main stay at
the last published pins until a later ship decision — this PR does **not** pin digests or tag.

Addresses Macrocosmos hold on v0.7.4 (keys unobservable / overweight) and the pre-merge
calibration miss at `ef63d2b` (leaders ~0.42–0.45 after key weight was cut to 2.80; wall
~850–990 s at horizon 1600):

- **Noisy closeness + wrist detent + retarget pulse** so keys are searchable from obs alone
  (absolute yaw still hidden; cues publish only after press-probe).
- **D-flat / n=1 auto-arm** after press-probe.
- **Hand-yaw key index** (wrist vs hidden window); `KEY_HOLD=4`.
- **Weights:** insert/collar/twist 0.04, screw 0.10, rework 1.10, stack 1.20, key **6.00**
  (missing keys cap non-key finishers ≤ ~0.30). In-order key finishes keep a 0.95 credit floor
  so short horizons do not erase the time bonus.
- Horizon hard cap **870** control steps (leaders run to the horizon at ~18–20 ms/step; 1600
  projected ~850–990 s on 2 CPU). No wall-clock score caps. ONNX session loaded once per worker.
- Honest privileged key path is **actions only**; oracle snaps stay behind `oracle=True`.
- Insert latch (credit survives pull-out) documented; insert weight stays low.

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold` / `_stack_mated` / `_peg_probe_frames` / key
teleports). Pose nudges for seat/stack/screw only. Public seeds. Leader ONNX evals load the
session once per worker (same as the player image).

Reproduce (in-process; same code the image bakes):

```bash
NIST_PROFILE=v07 python tools/obs_key_searcher.py --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
```

Public-seed evidence (mean over seeds 7 and 19, 64 eps / 2 CPU workers):

| Policy | seed 7 | seed 19 | mean | wall 7 / 19 |
|---|---:|---:|---:|---|
| honest privileged | 0.534 | 0.566 | **0.550** | 94 / 85 s |
| zero | 0.000 | 0.000 | **0.000** | 87 / 84 s |
| leader_1 | 0.275 | 0.291 | **0.283** | 484 / 480 s |
| leader_2 | 0.226 | 0.303 | **0.265** | 596 / 579 s |
| leader_3 | 0.241 | 0.294 | **0.267** | 479 / 485 s |
| leader_4 | 0.226 | 0.303 | **0.265** | 494 / 476 s |
| leader_5 | 0.220 | 0.269 | **0.245** | 481 / 477 s |
| obs-only key searcher | 26/64 keys | 18/64 keys | **pass** | — |

Strong scripted controllers A/B/C are covered by the leader_N band (same gate: mean ≤ ~0.30,
wall ≤ ~600 s). Determinism: repeat leader_2 seed 7 matched exactly.

Obs searcher: fixture seats + press-probes; policy uses observations/actions only (no
true-state reads, no probe writes / teleports).

## v0.7.4 (branching / probe adaptive harden, profile `v07`)

Spec `0.7.4`. Held for rollout after Macrocosmos review (keys unobservable / overweight).
Superseded by v0.7.5 for the observation-completable key path. Historical table:

| Policy | mean (seeds 7+19) | notes |
|---|---|---|
| privileged reference (honest) | **0.478** | used probe writes + key teleports |
| v0.6-style scripted | **0.068** | ≤~0.25 |
| strong scripted controllers (3) | **0.217–0.239** | ≤~0.25 |

Weights (v0.7.4): insert/collar/twist 0.02, screw 0.04, rework 1.10, stack 1.25, key 7.00

## Earlier releases

See git history for v0.7.3 / v0.7.2 / v0.7.1 / v0.7.0 / v0.6.x calibration tables.
