# Competition Onboarding Manifest: `nist_lite_assembly` v0.4.0

## 1. Goal statement & alignment plan
**What success looks like:** a single learned policy that sequentially assembles several
NIST ATB-1 pegs on a full board under free order, clutter, tight fits, and sensor/actuation
perturbations. It generalises across sizes and start states. Partial-credit scoring keeps
headroom above plain success so the platform 1% takeover rule stays usable.

- Check 1: success is not concentrated in one K or size. Per-K and per-size seat rates are reported.
- Check 2: performance holds on reserved conditions (held-out LHS sub-ranges).
- Check 3: actions stay physically sensible (mean squared action recorded; not ranked).
- Review cadence: each round, replay top trajectories (`tools/read_records.py --episode`).

## 2. Deliverables
| Item | Where | Done |
|---|---|---|
| Competition repo (public) + release tag | `https://github.com/unarbos/apex-competition-nist-assembly` @ `v0.4.0` | ☑ |
| `spec.yaml` version `0.4.0`, v0.4.0 digests pinned | `spec.yaml` | ☑ |
| Player image | `ghcr.io/unarbos/apex-competition-nist-assembly-player@sha256:8c80ce89f575826b515c9a3173eccc54b0d8f9748fc6613d90d8a64dea71d37f` | ☑ |
| Referee image (**private package**) | `ghcr.io/unarbos/apex-competition-nist-assembly-referee@sha256:391bfcaa1250feb26e923e79919ecd7bc4786795564bb839f0a5f79b931fbc69` | ☑ |
| Held-out `reserved.json` from build secret | referee Dockerfile `reserve.py` | ☑ mechanism |
| Reference baseline ONNX | `baselines/baseline.onnx` | ☑ |
| Junk test-set policies | `test_policies/` | ☑ |
| Calibration across ≥10 seeds | `CALIBRATION.md` | ☑ |
| Miner README + AlphaNIST section | `README.md` | ☑ |

Pins unchanged: python:3.12-slim@sha256:a6e34c…, numpy 2.5.4, mujoco 3.16.0, onnx 1.23.2,
onnxruntime 1.31.0. Image digests from release run 39681439574 (tag `v0.4.0`), both cosign-verified.

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
| evaluate / referee timeout_s | 1800 / 1800 (since 0.2.1) | 128-episode live rounds took ~400–660 s on Macrocosmos staging |
| deadline_ms | 500 per batched step | |

## 4. Round variation & evaluation sizing
- **Strata:** `(K ∈ {3,4,5,6}) × (clutter low|high)` = 8 strata × 8 episodes = **64**/round. Mean K = 4.5.
- **Axes (LHS inside strata):** clearance 0.05–0.6 mm, friction 0.3–1.2, mass 5–50 g, gain 0.6–1.4,
  obs noise 0–1 mm, angular noise 0–0.03 rad, dropout, act delay 0–3, disturbance 0/1, plus board pose,
  peg subset, start types, clutter poses.
- **Horizon:** `200 × K` control steps (8 s/peg @ 25 Hz).
- **Scoring:** per-peg `0.75 + 0.25·(time/effort)`; episode mean; round mean. Wrong slot / knocked out = 0.
- **Reserved region:** `reserve.py` draws from extended `RANGES` at referee build time.
- **Budget:** full board ~15k physics steps/s on the box; 64 × 1200 × 40 ≈ 3.1M substeps → ~100–120 s
  with 2 workers locally; prod ~3–4× slower; live 128-episode rounds ~400–660 s, inside the 1800 s timeout. See `CALIBRATION.md`.

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

## 10. Supersedes
PR #6 (`v0.1.6` efficiency-on-single-peg) is superseded by this multi-peg redesign. The
efficiency idea is reused inside the per-peg `0.75 + 0.25·time` term.
