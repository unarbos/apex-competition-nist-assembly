# NIST-Lite calibration

## v0.7.3 (adaptive compound harden, profile `v07`)

Spec `0.7.3`. Image bake stays `NIST_PROFILE=v07` (do not regress). Digests on main stay at
v0.7.2 until a later ship decision — this release does **not** pin digests.

Hardens vs v0.7.1-trained Round 22 kings (which lost almost nothing on v0.7.2). Difficulty is
in **feedback / adaptation structure**, not perception noise or broken physics:

- Mid-episode adaptive events: `retarget` (re-sample unfinished key yaw), `reorder` (add a
  wave-consistent precedence edge), `flip_twist`, plus denser `misseat` / `swap_bias`.
- Post-insert key retarget (per-tier probability) so early open-loop yaw plans fail.
- Touch-gated reveals (`touch_reveal`) for keys/twists — params stay zeroed until the gripper
  has touched that peg.
- Stack-before-key precedence on the same peg; keys only complete after prerequisites.
- Out-of-order **keys** score 0 (stack/rework keep the normal floor). Weights shift onto stacks.
- Honest privileged solver prefers stack before key; miner-path stack latch tests extended.

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold` / `_stack_mated`). Pose nudges only. Public seeds.
R22 king ONNX files are **not** in the repo (calibration only).

Reproduce (in-process; same code the image bakes):

```bash
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
# ONNX kings (paths local only — never commit):
NIST_PROFILE=v07 python tools/calib_v06.py --profile v07 --seeds 7 19 --per 8 --workers 4 \
  --policies reference v06_style /path/to/5DU65b_v1.onnx /path/to/5FH3YY_v2.onnx \
  --ref reference --live-eps 64 --cpus 2 --out out/calib_v073
```

Image-through (after `docker build` with `NIST_PROFILE=v07` baked):

```bash
python tools/image_eval_onnx.py --seed 7 --full --referee-image nist-assembly-referee:v073 \
  --player-image nist-assembly-player:v073 /path/to/5DU65b_v1.onnx
```

### Calibration table (64 episodes = 8/stratum, 2 CPU workers, `deadline_ms` 80)

| Policy | seed 7 mean | seed 7 wall_s | seed 19 mean | seed 19 wall_s | notes |
|---|---|---|---|---|---|
| privileged reference (honest) | **0.471** | 138 | **0.426** | 74 | mean ≈ **0.449**; ≥~0.45 bar |
| R22 king `5DU65b_v1` | **0.236** | 459 | **0.210** | 445 | ≤~0.25; was ~0.43–0.45 on v0.7.2 |
| R22 king `5FH3YY_v2` | **0.124** | 462 | **0.100** | 460 | ≤~0.25; was ~0.33–0.34 on v0.7.2 |
| v0.6-style scripted | **0.171** | 103 | — | — | seed 7; stays low |
| same seed ⇒ same scores | — | — | — | — | `5DU65b` 8-ep twice: **identical** |
| image smoke `5DU65b_v1` (built `v073`, seed 7, 16 ep) | **0.218** | 85 | — | — | asserts `nist_profile=v07` |

v0.7.2 published-image baseline (for comparison, measured externally on digests
`player@sha256:4716bc83…` / `referee@sha256:121f08d6…`): `5DU65b_v1` ≈0.43–0.45,
`5FH3YY_v2` ≈0.33–0.34; honest privileged ≈0.50.

Weights (v0.7.3): insert/collar/twist 0.04, screw 0.06, rework 1.60, stack 5.00, key 2.60.
`STACK_MATE_HOLD=5`, `STACK_MATE_XY=0.0034`. No wall-clock score caps. Horizon ≤ 1400.

## v0.7.2 (stack action-mate + Bernoulli twist/screw, profile `v07`)

Spec `0.7.2`. Image bake stays `NIST_PROFILE=v07` (do not regress). Digests on main stay at
v0.7.1 until a later ship decision — this release does **not** pin digests.

Fixes vs v0.7.1:
- Stack completion goes through `_maybe_mate_stacks` only for honest privileged (no
  `_pin_stack_on_peg` teleport). Stack rings are interference-fit; latch arms on a closed
  grasp in the mate zone, then counts released frames (anti-king single-frame sweep).
- Twist/screw draws use per-peg Bernoulli(frac) instead of `round(frac * n)` (which was 0
  on live-sized graphs).

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold` / `_stack_mated`). Pose nudges only. Public seeds.

Reproduce:

```bash
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/calib_v06.py --profile v07 --seeds 7 --per 8 --workers 4 \
  --policies reference v06_style compound_scripted zero --ref reference \
  --live-eps 64 --cpus 2 --out out/calib_v072
```

| Policy | mean (seed 7, 8/stratum, 64 eps) | notes |
|---|---|---|
| privileged reference (honest) | **0.527** | stack completed/drawn **44/74 (59%)**; latch only |
| v0.6-style scripted | **0.201** | insert/collar/twist port; ≤~0.25 king bar |
| zero | **0.003** | in-process idle |
| obs-only compound_scripted | low (≤~0.15 target) | in-process |

Twist/screw on this draw: 1 twist + 4 screws across 64 eps (were 0/0 under `round`). Weights
were later hardened in-tree (key/stack/rework dominate). `NIST_PROFILE=v07` baked in referee Dockerfile.

**Mistake for Round 22:** the ≤0.25 king gate was calibrated only on **v0.6-era** kings. v0.7.1-trained
policies still scored ~0.43 on published v0.7.2 — see v0.7.3.

## v0.7.1 (superseded for stack teleport / twist draw)

Live digests may still point here until ship. Privileged honest was **0.533** but stack
credit relied on `_nudge_stack_mate` → `_pin_stack_on_peg` (miners cannot call). With nudge
patched out: stack **0/75**. Twist/screw: **0** on seed 7 / 64 eps via `round(frac * n)`.

## v0.7.0 (superseded — oracle-inflated privileged bar)

The v0.7.0 privileged row used oracle helpers in `tools/compound_solver.py` and is **not** a
valid solvability proof. Image also baked `NIST_PROFILE=v06` and cut rounds on wall clocks.
See v0.7.1.

## Earlier releases

See git history for v0.6.x / v0.5.x calibration tables (unchanged meaning).
