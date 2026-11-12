# NIST-Lite calibration

## v0.7.4 (branching / probe adaptive harden, profile `v07`)

Spec `0.7.4`. Image bake stays `NIST_PROFILE=v07` (do not regress). Digests on main stay at
v0.7.3 until a later ship decision — this release does **not** pin digests.

Hardens vs strong conditional ONNX scripts that still scored ~0.50–0.60 on live v0.7.3
(Rounds 25–27). Difficulty is in **feedback / adaptation structure**, not perception noise
or broken physics:

- **Blind key yaw:** obs never publishes absolute key yaw or a closeness scalar. After
  reveal + press-probe, only `key_tol` may pack into a free collar-flag slot.
- **Press-probe + leave/reenter:** key completion needs `KEY_PROBE_FRAMES` of press, then
  the peg must leave the key window once (`_key_index_armed`) before hold can latch —
  blocks natural-seat freebies after an accidental press.
- **Single-active key rotation (`activate`)** on dual-key hard tiers; inactive keys report
  `blocked=1` and cannot accumulate hold.
- **`regraph`:** drop soft unfinished edges and redraw 1–3 wave-consistent ones.
- **Post-stack / post-insert key retarget**; keys only wait on their own peg's insert/rework.
- Weights: insert/collar/twist 0.02, screw 0.04, rework 1.10, stack 1.25, key 7.00
- Horizon hard cap **1600** control steps. No wall-clock score caps.

**Privileged numbers below are honest** (`solve_compound(..., oracle=False)` / no helpers that
set `step_done` / `collar_run` / `_key_hold` / `_stack_mated`). Pose nudges only. Public seeds.
Strong scripted ONNX files are **not** in the repo (calibration only); refer to them as
"strong scripted controllers" with aggregate scores only.

Reproduce (in-process; same code the image bakes):

```bash
NIST_PROFILE=v07 python tools/compound_solver.py --profile v07 --honest --seed 7 --per 8 --episodes 64
NIST_PROFILE=v07 python tools/calib_v06.py --profile v07 --seeds 7 19 --per 8 --workers 4 \
  --policies reference v06_style compound_scripted /path/to/strong_scripted_*.onnx \
  --ref reference --live-eps 64 --cpus 2 --out out/calib_v074
```

### Calibration table (128 paired episodes = seeds 7+19 × 8/stratum; project 64 live eps on 2 CPU)

| Policy | mean (seeds 7+19) | by tier (c1/c2/c3/c4) | projected live s | notes |
|---|---|---|---|---|
| privileged reference (honest) | **0.478** | 0.54 / 0.52 / 0.29 / 0.37 | 84 | ~0.48 (target ~0.5–0.75) |
| v0.6-style scripted | **0.068** | 0.08 / 0.07 / 0.04 / 0.05 | 86 | ≤~0.25 |
| strong scripted controllers (3) | **0.217–0.239** | — | 485–506† | ≤~0.25; drop vs v0.7.3 |
| projected live (slowest clean) | — | — | **485 s** | <600 s target; hard 1800 s |

† Clean single-policy remeasure for the previously contention-inflated king was **485 s**.
Paired 5-policy run under CPU contention briefly projected ~900 s — not used.

Interface: **unchanged** (`gym_v1`, `[B,288]→[B,5]`). Miner-path stack latch + snap seals kept;
new `tests/test_v074_adaptive.py`, `tests/test_v074_miner_path.py`.

### v0.7.3 → v0.7.4 strong-script drop (same seeds 7+19, 8/stratum, 128 eps)

| Policy | v0.7.3 mean | v0.7.4 mean | Δ |
|---|---|---|---|
| privileged reference (honest) | 0.449 | **0.478** | +0.03 |
| v0.6-style scripted | 0.182 | **0.068** | −0.11 |
| strong scripted controllers (best of 3) | **0.588** | **0.239** | −0.35 |
| strong scripted controllers (range of 3) | 0.549–0.588 | 0.217–0.239 | — |

Live board on published v0.7.3: R25/R26/R27 kings ~0.52–0.60 (matches local ~0.55–0.59).

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
  --policies reference v06_style /path/to/strong_scripted_*.onnx \
  --ref reference --live-eps 64 --cpus 2 --out out/calib_v073
```

Live board (Rounds 25–27 on published v0.7.3): top submissions ~0.50–0.60 — same family of
strong scripted controllers; see v0.7.4 gate.

Weights (v0.7.3): insert/collar/twist 0.02, screw 0.04, rework 0.9, stack 1.1, key 7.5

## Earlier releases

See git history for v0.7.2 / v0.7.1 / v0.7.0 / v0.6.x calibration tables. v0.7.1-trained
policies still scored ~0.43 on published v0.7.2 — see v0.7.3.
