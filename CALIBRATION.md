# NIST-Lite calibration

## v0.7.5 (observation-completable keys, profile `v07`)

Spec `0.7.5`. Image bake stays `NIST_PROFILE=v07` (do not regress). Digests on main stay at
the last published pins until a later ship decision — this PR does **not** pin digests or tag.

Addresses Macrocosmos hold on v0.7.4 (keys ~73% weight but unobservable; D-flat arming;
retarget invisible; privileged probe writes / teleports; dead `_obs_key_yaw` RNG):

- **Noisy closeness + wrist detent + retarget pulse** so keys are searchable from obs alone
  (absolute yaw still hidden).
- **D-flat / n=1 auto-arm** after press-probe.
- **Hand-yaw key index** (wrist vs hidden window) with noisy closeness + detent + retarget pulse.
- **Weights:** insert/collar/twist 0.12, screw 0.45, rework 1.50, stack 2.00, key 2.80.
- Honest privileged key path is **actions only**; oracle snaps stay behind `oracle=True`.
- Horizon hard cap **1600** control steps. No wall-clock score caps.
- Insert latch (credit survives pull-out) documented; insert weight stays low.

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold` / `_stack_mated` / `_peg_probe_frames` / key
teleports). Pose nudges for seat/stack/screw only. Public seeds.

Reproduce (in-process; same code the image bakes):

```bash
NIST_PROFILE=v07 python tools/obs_key_searcher.py --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/compound_obs_scripted.py --seed 7 --per 8 --episodes 32
```

Public-seed evidence (this RC):

| Bar | Setup | Result |
|---|---|---|
| (a) obs-only key searcher | seed 7 / 64 eps / budget 500 | **22/80 keys (27.5%)**, mean 183 frames |
| (b) honest privileged | `--honest` seed 7 / 64 | **0.565** |
| (b) oracle privileged | seed 7 / 16 | **0.722** |
| (c) trivial zero | seed 7 / 16 | **~0.001** |
| (c) strong obs-scripted | seed 7 / 32 | **0.005** |
| wall | honest 64 | **~311 s** (<400 s) |

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
