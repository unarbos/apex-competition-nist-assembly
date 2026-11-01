# Competition Onboarding Manifest: `nist_lite_assembly` v0.6.0

## 1. Goal statement & alignment plan
**What success looks like:** a single learned policy that completes random compound
assembly graphs (peg inserts, collars, twist-locks, precedence, parts in waves, hidden physics
changes between subtasks) on a full NIST ATB-1-style board under clutter, tight fits, and
sensor/actuation perturbations. It generalises across sizes and start states. Partial-credit scoring keeps
headroom above plain success so the platform 1% takeover rule stays usable.

- Check 1: success is not concentrated in one tier, step kind or size. Per-tier scores, step-done fraction and per-size seat rates are reported.
- Check 2: performance holds on reserved conditions (held-out LHS sub-ranges).
- Check 3: actions stay physically sensible (mean squared action recorded; not ranked).
- Review cadence: each round, replay top trajectories (`tools/read_records.py --episode`).

## 2. Deliverables
| Item | Where | Done |
|---|---|---|
| Competition repo (public) + release tag | `https://github.com/unarbos/apex-competition-nist-assembly` @ `v0.6.0` | ☑ |
| `spec.yaml` version `0.6.0`, v0.6.0 digests pinned | `spec.yaml` | ☐ after the release run |
| Player image | `ghcr.io/unarbos/apex-competition-nist-assembly-player@sha256:b1724099a7a39c5e6cbf69107e6e30a571a305af017e38d7b7f20126ed561966` | ☑ |
| Referee image (**private package**) | `ghcr.io/unarbos/apex-competition-nist-assembly-referee@sha256:7d83da82b0bc90d88a163ab6c93da9e637d1ea6967c47d991ea92e1d5fa40064` | ☑ |
| Held-out `reserved.json` from build secret | referee Dockerfile `reserve.py` | ☑ mechanism |
| Reference baseline ONNX | `baselines/baseline.onnx` | ☑ |
| Junk test-set policies | `test_policies/` | ☑ |
| Calibration across ≥10 seeds | `CALIBRATION.md` | ☑ |
| Miner README + AlphaNIST section | `README.md` | ☑ |

Pins unchanged: python:3.12-slim@sha256:a6e34c…, numpy 2.5.4, mujoco 3.16.0, onnx 1.23.2,
onnxruntime 1.31.0. The digests above are the v0.5.1 ones (release run 41125321583) until the v0.6.0
release run pins new ones.

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
| evaluate / referee timeout_s | 1800 / 1800 (since 0.2.1) | v0.5 128-episode live rounds took ~400–660 s on Macrocosmos staging; v0.6 at 96 episodes is projected at ~420–700 s (§4) |
| episodes_per_stratum (round input) | **12** on 2 CPU (96 episodes); 16 (128) only with 4 referee CPUs and `workers` 4 | §4 |
| deadline_ms | 500 per batched step | |

## 4. Round variation & evaluation sizing
- **Strata:** `(K ∈ {3,4,5,6}) × (clutter low|high)` = 8 strata. The fixture uses 8 per stratum (64
  episodes). Live rounds ran at 16 (128) on v0.5; **v0.6 live: 12 per stratum = 96 episodes** on
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
  0.70 + 0.30·time, or ×0.15 if done out of order; an unfinished step earns ≤ 0.30 partial progress;
  episode = weighted mean, round = mean.
- **Lying pegs:** reachable and graspable, but the privileged solver seats 0/64. They are ~13% of
  inserts and are stated as unproven headroom in `CALIBRATION.md`.
- **Reserved region:** `reserve.py` draws from extended `RANGES` at referee build time.
- **Budget:** a v0.6 episode costs ~1.4× a v0.5.1 episode: 1.44× the control steps, and per-step
  cost at 500 Hz with 8-segment collars ≈ v0.5.1 at 1 kHz. Scaling the observed v0.5 live
  400–660 s for 128 episodes gives ~560–930 s for 128 v0.6 episodes on 2 CPU, or **~420–700 s for
  96**. With 4 referee CPUs and `workers` 4, 128 episodes would take ~300–500 s. The 1800 s timeout
  holds either way. Local podman smoke on the released digests: see the release notes.

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
- Round input: `episodes_per_stratum` 12 recommended on 2 CPU (§4).
- Folded in from the v0.5.2 notes: lying-peg ceiling stated (0/64 privileged, unproven
  headroom), glitch-burst step share documented, §4 numbers corrected (horizon, strata, tiers).
- Calibration: `CALIBRATION.md` (privileged 0.41 overall / 0.24 c4, revealed v0.5 scripted ~0.20,
  zero 0.008). Learned-beats-scripted is open; work continues after launch.

## 12. Supersedes
PR #6 (`v0.1.6` efficiency-on-single-peg) is superseded by this multi-peg redesign. The
efficiency idea is reused inside the per-peg `0.75 + 0.25·time` term.
