# Competition Onboarding Manifest: `nist_lite_assembly` v0.7.0 (Assembly III)

## 1. Goal statement & alignment plan
**What success looks like:** a single learned policy that completes random compound
assembly graphs (peg inserts, collars, twist-locks, and in v0.7-dev also rework / screw /
stack / key steps, precedence, parts in waves, hidden physics changes between subtasks) on a
full NIST ATB-1-style board under clutter, tight fits, and sensor/actuation perturbations. It
generalises across sizes and start states. Partial-credit scoring keeps headroom above plain
success so the platform 1% takeover rule stays usable. v0.7-dev aims for a much higher ceiling
than v0.6 (c4 near-impossible) without blowing the eval budget.

- Check 1: success is not concentrated in one tier, step kind or size. Per-tier scores, step-done fraction and per-size seat rates are reported.
- Check 2: performance holds on reserved conditions (held-out LHS sub-ranges).
- Check 3: actions stay physically sensible (mean squared action recorded; not ranked).
- Review cadence: each round, replay top trajectories (`tools/read_records.py --episode`).

## 2. Deliverables
| Item | Where | Done |
|---|---|---|
| Competition repo (public) + release tag | `https://github.com/unarbos/apex-competition-nist-assembly` @ `v0.7.0` (tag pending owner cut) | ☑ |
| `spec.yaml` version `0.6.3` with v0.6.3 digests; `apex-dev preflight` passes (stage + prod) | `spec.yaml` | ☑ |
| Player image | `ghcr.io/unarbos/apex-competition-nist-assembly-player@sha256:362ea25b5f7fdc7b6567e8f919fb8bd2e7521cdf048adeec6771d6fc161e2c7a` | ☑ |
| Referee image (**private package**) | `ghcr.io/unarbos/apex-competition-nist-assembly-referee@sha256:385e39bffb066440a2583e713ec981be9cfc7c1eca31483ee516bbb0ac650717` | ☑ |
| Held-out `reserved.json` from build secret | referee Dockerfile `reserve.py` | ☑ mechanism |
| Reference baseline ONNX | `baselines/baseline.onnx` | ☑ |
| Junk test-set policies | `test_policies/` | ☑ |
| Calibration across ≥10 seeds | `CALIBRATION.md` | ☑ |
| Miner README + AlphaNIST section | `README.md` | ☑ |

Pins unchanged: python:3.12-slim@sha256:a6e34c…, numpy 2.5.4, mujoco 3.16.0, onnx 1.23.2,
onnxruntime 1.31.0. Image digests from release run 41474262290 (tag `v0.6.1`) until the v0.6.2 release.

## 3. Ops parameters
| Parameter | Proposal | Reason |
|---|---|---|
| process_type | cpu | MuJoCo on CPU |
| kind | solo | absolute score |
| round_length_in_days | 1 | platform norm |
| submission_reveal_days | 1 | |
| lower_is_better | false | |
| baseline_raw_score / baseline_score | 0.0 / 0.0 | platform convention |
| resources | 2 CPU / 2Gi / 0 GPU | 2 sim workers |
| evaluate / referee timeout_s | 1800 / 1800 (since 0.2.1) | v0.5 128-episode live rounds took ~400–660 s on Macrocosmos staging; v0.6 at 64 episodes is projected at ~500–1150 s (§4) |
| episodes_per_stratum (round input) | **8** on 2 CPU (64 episodes); 12 (96) only with 4 referee CPUs and `workers` 4 | §4 |
| deadline_ms | 500 per batched step | |

## 4. Round variation & evaluation sizing
- **Strata:** `(K ∈ {3,4,5,6}) × (clutter low|high)` = 8 strata. The fixture uses 8 per stratum (64
  episodes). Live rounds ran at 16 (128) on v0.5; **v0.6 live: 8 per stratum = 64 episodes** on
  2 CPU (see Budget). Mean K = 4.5 before the c3/c4 extra pegs.
- **Tiers:** sensing difficulty 25% easy / 35% medium / 40% hard (unchanged). Complexity tiers
  c1–c4 are assigned 15/30/30/25% within each stratum.
- **Axes (LHS inside strata, v0.5.1 ranges):** clearance 0.25–0.8 mm with variable chamfer,
  friction 0.3–1.3, mass 5–50 g, gain ×0.6–1.4, pose noise 0.8–3.5 mm and 0.015–0.06 rad,
  dropout 3–18%, hole-estimate bias 4–10 mm, peg-vs-hole residual 5–11.5 mm, drift 0.01–0.06 mm/step,
  backlash, rare act delay, disturbance, plus board pose, layout, families, start types, clutter.
- **Compound layer (v0.6):** steps (insert / collar / twist), precedence edges, 1–4 waves, and
  between-step events (friction, gain, mass, delay, re-bias, shove). Glitch bursts run at half the
  v0.5.1 rate and cover ~25% of control steps (easy 16 / medium 25 / hard 29%).
- **Horizon:** Σ step budgets (insert 260, +140 lying, collar 200, twist 130) × tier slack
  (1.2 / 1.0 / 0.85 / 0.70) + 40 per extra wave; mean ~1,810 control steps. (v0.5.x: 280 × K, mean
  ~1,260; the "200 × K" previously listed here was stale.)
- **Scoring:** weighted per-step credit (insert 0.3 / collar 1.5 / twist 1.0). A done step earns
  0.70 + 0.30·time; out of order earns `max(0.30, 0.15 × in-order)`; unfinished ≤ 0.30 partial;
  episode = weighted mean, round = mean.
- **Lying pegs:** reachable and graspable, but the privileged solver seats 0/64. They are ~13% of
  inserts, also delay later waves, and are stated as unproven headroom / ceiling in `CALIBRATION.md`.
- **Reserved region:** `reserve.py` draws from extended `RANGES` at referee build time.
- **Budget (measured on the v0.6.0 images, podman, 2 CPU / 2 GiB each, seed 1):** baseline policy,
  96 episodes: 426 s. Revealed v0.5 scripted king, 64 episodes: 410 s. The same setup ran v0.5.1 at
  233 s for 128 episodes (baseline), and Macrocosmos staging took ~10 min for that round, a ~2.6× factor
  (1.7–2.8× over the v0.5 live range). A v0.6 episode costs ~2.4× a v0.5.1 episode here, not the 1.4× the
  idle-physics estimate gave: there are ~1.7× more lockstep player calls (the longest horizon is ~2,850
  steps), plus the compound bookkeeping. Projected staging time: 64 episodes ≈ 500–800 s (baseline)
  and up to ~700–1150 s (heavy policy); 96 episodes ≈ 720–1200 s (baseline) and up to ~1700 s (heavy
  policy), which is too close to the 1800 s timeout. **Live: 8 per stratum (64) on 2 CPU.** With 4
  referee CPUs and `workers` 4, 12 per stratum (96) should fit.

## 5. Threat-model questionnaire
Unchanged process: answers withheld from the public repo; available to Macrocosmos privately.

## 6. GPU justification: n/a (CPU)

## 7. v0.3.0 changes (observation hardening)
- Same obs/action contract (`[B,184]` -> `[B,5]`), task, strata and scoring as v0.2.
- Breaking for tuned policies: the observation noise model changed. Peg, slot and clutter
  pose-estimate offsets are drawn independently per slot, per clutter block and per peg.
- Published per-episode records (`nist_lite_record/3`) carry only coarse labels (K, clutter
  bucket and count, tier, clearance class, disturbance flag, peg slot / start type / shape).
  No bias values, board pose, true start poses, or exact continuous axis values.
- The release referee mixes a salt derived from the `NIST_HELDOUT_SEED` build secret into the
  round and episode RNGs, so a round cannot be reproduced offline from the platform SEED and
  the public simulator. Dev builds (no secret) are unchanged and still deterministic per seed.
- `baselines/baseline.onnx` is now a memoryless rule-based twin of the Python baseline
  (was a linear distillation).

## 8. v0.4.0 changes (replay-compatible records)
- No change to the simulator, task, difficulty, scoring, obs/action contract or images'
  runtime behaviour; only the published record layout.
- Records (episode rows in `episodes.jsonl` and `traj_*.json`) use `format: "nist_lite_record/2"`
  again, the layout the website replay viewer renders.
- Exact: `conditions.board_xy`, `conditions.board_yaw`, and per clutter block `offset_xy`,
  `yaw`, `size_xy`, `height` (not held-out axes; re-drawn every round; records publish after
  the round).
- Coarse: `conditions.clearance` is the midpoint of the episode's clearance class
  (tight 0.325 mm, mid 0.5 mm, loose 0.7 mm); `conditions.chamfer` is a fixed nominal 0.5.
  The simulator still uses the exact per-episode values.
- Unchanged from v0.3.0: `n_pegs`, `k`, `clutter_bucket`, `n_clutter`, `difficulty_tier`,
  `clearance_class`, `disturbance`, `max_steps`, pegs (`slot`, `shape`, `start_type`),
  `pegs_seated`, trajectory steps (`t`, `hand`, `action`, peg `pos`/`quat`, `seated`).
- Still not published: estimate-bias values, exact values of any continuous condition axis,
  reserved sub-ranges, the build salt, true peg start poses in episode rows.

## 9. v0.5.0 changes (generalization hardening)
- **Breaking for tuned policies.** Same `[B,184] -> [B,5]` contract, strata, tiers,
  horizon and scoring, but the board is no longer a fixed public table:
  - every episode draws its own 8-slot layout (non-overlapping positions anywhere on the
    board, slot yaw in (-pi, pi]) and a cross-section per slot from continuous families
    (round, rectangle/square, regular 5/6/8-gon, D-flat; inscribed size 4-16 mm);
  - held-out families (reserved split, hard tier only) are new and never appear in live rounds;
  - peg-table features 8-11 are now `sin, cos` of a biased slot-yaw estimate, the outer
    (circumscribed) size and a zero. There is no shape label. The slot-yaw channel carries
    its own per-peg residual against the peg-yaw channel.
- The v0.3.0 estimate-offset structure is unchanged (independent per slot, clutter block
  and peg); `tests/test_obs_leak.py` checks it against each episode's own layout.
- Records stay `nist_lite_record/2`. Added: `conditions.slots` (shape, size, xy, yaw,
  params for all 8 slots of the episode) and peg `size`. The layout is re-drawn every
  episode and only published after the round. Result summary seat keys are now
  `seat_<family>_<small|medium|large>`.
- Calibration tools: `tools/eval_policies.py` (fast local eval), `tools/obs_refport.py`
  (obs-only port of the reference solver, the strong scripted gate),
  `tests/test_scripted_regression.py` (scripted controllers must stay <= 0.30; extra
  private ONNX fixtures via `NIST_SCRIPTED_FIXTURES`). The reference solver presses
  gently on a blocked insertion (target no longer runs away), which lifts the privileged
  reference to ~0.70.

## 10. v0.5.1 changes (observation hardening)
- Same `[B,184] -> [B,5]` contract, task, strata, tiers, horizon, scoring and record format
  (`nist_lite_record/2`). Scores are not directly comparable with v0.5.0.
- Harder observations: peg-vs-slot estimate residual 5–11.5 mm (was 1.5–5; tier scale easy 0.9,
  medium 1.0), pose noise 0.8–3.5 mm (was 0.4–2.0), estimate drift 0.01–0.06 mm/step on every
  tier, estimate glitch bursts (8–30 steps, 4× noise plus a 2–6 mm transient offset per object;
  4% per step on hard, scaled 0.8 medium / 0.5 easy), slot-yaw residual 0.05–0.14 rad.
- More lying-down starts: hard 22% (was 8%), medium 12% (was 2%).
- Fixes from the v0.5.0 validation (issue #13): layout redraw instead of an overlap fallback;
  free-peg spawns, tray spots, `_force_safe_start` and the teleport grid clear every slot;
  pre-grasped hand yaw wrapped into [-pi/2, pi/2]; records add peg `drawn_start_type` next to
  the effective `start_type`; docstrings.
- Tools: `tools/fingerprint.py` (action fingerprints on a probe set, for near-duplicate
  detection), `tools/train_ppo.py --base-onnx`.
- Calibration: see `CALIBRATION.md`. Revealed v0.5.0 scripted top submissions drop from
  ~0.9 to ≤ ~0.27; privileged reference 0.66.

## 11. v0.6.0 changes (compound tasks, breaking)
- Every episode is a random compound assembly graph (`referee/nist_sim/compound.py`):
  - steps: peg `insert`, `collar` (a ring that fits only over a seated peg) and `twist` (turn a
    seated round peg 1.2–2.0 rad);
  - mandatory and random precedence edges; parts in 1–4 waves;
  - hidden events between subtasks;
  - complexity tiers c1–c4;
  - at least one collar or twist per episode;
  - lost parts are removed rather than ending the episode.
- Scoring: weighted per-step partial credit (`compound_score`), replacing the per-peg score.
- Observation `[B, 288]`: the v0.5 184-d prefix plus a task table (8×6), a collar table (4×13) and
  globals (4). The player passes a statically 184-wide policy the prefix, so v0.5 submissions
  still load (and score ~0.2).
- Physics at 500 Hz (20 substeps), 8-segment collars. Action rounding stays **off**: it ships only
  together with a stateful-policy protocol.
- Records `nist_lite_record/3`. Episode rows add `task` (tier, n_steps, kinds, n_edges, n_waves,
  n_collars), `tier`, `steps_done`, `steps_in_order`, `n_steps`, `credit_by_kind`, `events_fired`
  and `parts_lost`. Result summary keys add `step_done_fraction` and `score_c1..c4`; the per-K
  `score_k*` keys are gone.
- Round input: `episodes_per_stratum` 8 (64 episodes) on 2 CPU (§4).
- Folded in from the v0.5.2 notes: lying-peg ceiling stated (0/64 privileged, unproven
  headroom), glitch-burst step share documented, §4 numbers corrected (horizon, strata, tiers).
- Calibration: `CALIBRATION.md` (privileged 0.41 overall / 0.24 c4, revealed v0.5 scripted ~0.20,
  zero 0.008). Learned-beats-scripted is open; work continues after launch.

## 12. v0.6.1 changes (pre-launch Macrocosmos fixes)
- Out-of-order step credit is `max(PARTIAL_CAP, 0.15 × in-order credit)` (≥ 0.30); docs match.
- Collar partial tiers 0.20 / 0.30 require that collar's peg in its hole.
- Records `nist_lite_record/4`: `conditions.collars`, full `task` (steps, wave_time),
  `step_results`, `events`; trajectories add `collars`, `parked`, `collar_parked` (world frame).
- Extra precedence a→b only when wave(a) ≤ wave(b); c3/c4 extra pegs get independent estimate
  biases; collar obs share peg glitch bursts / dropout.
- Release builds ignore `NIST_PHYSICS_DT` / `NIST_ACTION_QUANT`. CI runs with `NIST_PROFILE=v06`.
- Lying-peg ceiling restated in `CALIBRATION.md` (0/64 privileged; delays later waves).
- Digests / tag: pin after merge (do not tag from this PR).

## 12a. v0.6.2 changes (mass-event fix)
- A `mass` event rescales peg `body_mass` / `body_inertia` and now calls `mj_setConst` on a
  scratch `MjData`, so `body_subtreemass`, `dof_M0`, `body_invweight0` and `dof_invweight0`
  match the new masses. Stale constants made the next grasp oscillate under `implicitfast` and
  could end the episode `unstable` (reported by msulab, Discord, Nov 2 2026). The live state and
  the other event / parking edits are untouched (regression test in `tests/test_v06_compound.py`).
- Friction and gain events have no derived constants. (Parking does: see 12b.)

## 12b. v0.6.3 changes (parked parts after a mass event)
- `mj_setConst` also re-derives `body_contype` / `body_conaffinity` as the OR of the body's
  *current* geom masks. Parked (later-wave) parts have their geom masks set to 0, so a mass event
  while they were parked zeroed their body masks; when their wave arrived the geoms came back but
  the body masks stayed 0, the broad phase skipped them, and they fell through the board and were
  counted lost (reported by OddK, Discord, Nov 2 2026). The mass branch now restores the
  compile-time body masks after `mj_setConst`; parked geoms are still filtered at the geom level.
- A full-model diff of `mj_setConst` on a model with parked parts shows `body_contype` /
  `body_conaffinity` are the only fields it changes besides the mass-derived constants.
  Regression test `test_mass_event_keeps_parked_parts_collidable`; the v0.6.2 mass test still passes.

## 13. v0.7.0 (Assembly III, live)
- Profile `v07` is the live referee default (`referee.py` sets `NIST_PROFILE=v07`).
  Spec version `0.7.0`. Digests stay on the last v0.6.3 pins until the owner tags/releases.
- Module `referee/nist_sim/compound_v07.py`: denser tiers, new steps (`rework`, `screw`,
  `stack`, `key`), press-gated screws, yaw-index keys, late reveals, `swap_bias` events.
  Stack generation is deferred (contact flaky); hardness is screw/key weighted.
- Observation remains 288-d; v0.7 encodings reuse twist/collar channels (see README / DESIGN).
- Privileged solver (`tools/compound_solver.py`), v0.6-style port (`tools/v06_style_scripted.py`),
  and `tools/compound_obs_scripted.py`; `tools/calib_v06.py --profile v07`.
- Calibration (seed 7, 8/stratum, 64 eps): privileged **0.583** (c4 **0.334**), v0.6-style
  **0.313**, obs-only **0.071**, zero **0.008**. Measured 64-ep privileged wall on 2 CPU:
  **206 s**. Details in `CALIBRATION.md`.
- Design notes: `docs/v0.7/DESIGN.md`.

## 14. Supersedes
PR #6 (`v0.1.6` efficiency-on-single-peg) is superseded by this multi-peg redesign. The
efficiency idea is reused inside the per-peg `0.75 + 0.25·time` term.
