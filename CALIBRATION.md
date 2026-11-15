# NIST-Lite calibration

## v0.7.6 (actions-only key completability, profile `v07`)

Spec `0.7.6`. Image bake stays `NIST_PROFILE=v07` (do not regress). **PR only — do not
tag / pin digests / publish images from this branch.** Digests in `spec.yaml` remain the
v0.7.5 pins until a release bake.

Fixes the Macrocosmos hold that honest **actions-only** key completion was near zero on
v0.7.5 (1/64 seed 7, 0/64 seed 19) while fixture-assisted yaw search looked fine.

### Root cause (verified)

1. **Searcher gated yaw search on closeness.** Far from the window, closeness is ~0 by
   design (soft cue only rises inside ~`KEY_CLOSE_SOFT × tol`). Waiting for `t[3] > 0.08`
   before sweeping meant the policy never started the search.
2. **Grasped starts often spawned outside `r_hole`.** Legacy grasped offset was a 10 mm
   disk; tight holes are ~3–4 mm. Under centimetre-scale observation bias that is not
   seatable by honest press-and-release — keys looked unreachable from actions alone.
3. Secondary searcher bugs (treating insert-done as key-done, slot-chase yank on biased
   peg estimates, stacking feat-5 overwriting `key_tol`) are sealed/fixed.

### Physically sound env changes

- `KEY_TOL = (0.090, 0.150)`; `KEY_CLOSE_SOFT = 4.5`; `KEY_CLOSE_NOISE = 0.035`.
- `KEY_PRESENT_MARK = 0.5` on feat 2 after reveal+touch+active (pre-probe); final packing
  pass so stack/collar cannot clobber `key_tol` on feat 5.
- Grasped spawn centre inside `r_hole` (`≤ 0.55 × outer_radius`); keyed yaw err ±0.08.
- Key assignment prefers grasped / early-wave near-hole pegs (yaw window is the designed
  difficulty, not a millimetre XY lottery after a far loose pick).

Keys stay hard: completion still needs press-probe, leave/reenter arming (D-flat auto-arms),
and open-hold inside `KEY_TOL`. Absolute yaw is never published.

### Privileged reference inputs (`solve_compound(..., oracle=False)`)

**Uses (privileged):**

- Pose nudges: `_nudge_seat_peg`, `_nudge_stack_pose`, `_nudge_collar_seat`,
  `_nudge_rework_extract` (true-state `_set_free` teleports after engagement evidence).
- Wrist aim: `_aim_key_hand` writes hand `qpos[3]` + `target[3]` to the true `key_yaw`.
- Reads true `key_yaw` / seat geometry for planning.

**Does not use on the honest path:** `step_done` / `collar_run` / `_key_hold` /
`_stack_mated` / `_peg_probe_frames` writes, `_nudge_key_yaw` peg teleports, or
`oracle=True` completion snaps (`_privileged_*_snap`). Those stay behind `oracle=True`.

Leader ONNX fixtures from live v0.7.3/v0.7.5 were **not** in this repo checkout; scripted
obs controller stands in for the ≤~0.30 king gate.

Reproduce:

```bash
NIST_PROFILE=v07 python tools/obs_key_searcher.py --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/obs_key_searcher.py --fixture --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
```

Public-seed evidence (64 eps / single process on this agent host; wall times scale with
CPU — all well under the 600 s / 2-CPU bar and hard 1800 s timeout):

| Policy | seed 7 | seed 19 | mean | wall 7 / 19 |
|---|---:|---:|---:|---|
| honest privileged | 0.705 | 0.760 | **0.732** | 179 / 166 s |
| compound obs scripted | 0.031 | 0.002 | **0.016** | 195 / 183 s |
| actions-only key searcher (score) | 0.183 | 0.169 | **0.176** | 180 / 173 s |
| actions-only key searcher (key steps) | **18/64** | **17/64** | miner-path bar | — |
| fixture-assisted yaw search (key steps) | 58/64 | 58/64 | observability only | 51 / 51 s |

Before (v0.7.5 actions-only key steps): **1/64** (seed 7), **0/64** (seed 19).
Fixture-assisted on v0.7.5 was 26/64 and 18/64 (privileged seat+probe — not miner-path).

Gate check: scripted ≤ ~0.30 ✓; honest privileged in ~0.5–0.75 band ✓ (seed 19 slightly
above 0.75 at 0.760); walls ≪ 600 s ✓; no wall-clock score caps ✓.

## v0.7.5 (observation-completable keys, profile `v07`)

Spec `0.7.5`. **Released and pinned:** tag `v0.7.5`, release run
[43596917775](https://github.com/unarbos/apex-competition-nist-assembly/actions/runs/43596917775).
Player `sha256:80fad6b865c9af2a5731c7d2c3e516a9e465ff9551da053e4e3f740c10ecf324`, referee
`sha256:a55568d56eea8e69af0bfaa9c458a414f587aba231793783cb1842a99625c5b9`.

Historical public-seed table (actions-only key steps near zero — the hold that v0.7.6 fixes):

| Policy | seed 7 | seed 19 | mean | wall 7 / 19 |
|---|---:|---:|---:|---|
| honest privileged | 0.534 | 0.566 | **0.550** | 94 / 85 s |
| leader_1..5 | 0.220–0.275 | 0.269–0.303 | **0.245–0.283** | ~480–600 s |
| actions-only key searcher (key steps) | 1/64 | 0/64 | **near-zero** | — |
| fixture-assisted yaw search (key steps) | 26/64 | 18/64 | observability bar | — |

## Earlier releases

See git history for v0.7.4 / v0.7.3 / v0.7.2 / v0.7.1 / v0.7.0 / v0.6.x calibration tables.
