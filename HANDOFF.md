# Competition Onboarding Manifest: `nist_lite_assembly` v0.1.1

## 1. Goal statement & alignment plan
**What success looks like:** a single learned policy that assembles NIST ATB-1 parts as a real cell would need. It is robust
across part sizes, clearances, friction, mass, controller stiffness and sensor noise. It recovers from bad starts (loose or toppled
parts) and finishes by releasing a correctly seated part. It reaches this through general manipulation skill (grasp, reorient, align, insert), not by
fitting one round's numbers. This is the AlphaNIST result (binary task reward → emergent assembly), opened up as a competition.

- Check 1: success is not concentrated in one start type. Per-start-type success is reported, and leaders are inspected for loose/lying gains.
- Check 2: performance holds on reserved conditions. Part of the condition space is never drawn in live rounds. The leader is scored on it out of band each round.
- Check 3: actions stay physically sensible (mean squared action and unstable/lost counts are recorded; they are not ranked).
- Review cadence: each round, replay the top-3 trajectories (`tools/read_records.py --episode`) and compare per-stratum tables.

## 2. Deliverables
| Item | Where | Done |
|---|---|---|
| Competition repo (public) + released tag | `https://github.com/unarbos/apex-competition-nist-assembly` @ `v0.1.1` | ☑ |
| `spec.yaml`; `apex-dev preflight` passes (stage + prod) | `spec.yaml` | ☑ |
| Player image | `ghcr.io/unarbos/apex-competition-nist-assembly-player@sha256:<PLAYER_DIGEST>` | ☐ pending release |
| Referee image (**private package**) | `ghcr.io/unarbos/apex-competition-nist-assembly-referee@sha256:<REFEREE_DIGEST>` | ☐ pending release |
| Layer-2 screen image | n/a: ONNX artifact, with structural checks in the player loader (§5 Q15) | ☑ |
| `generate_round` | n/a: the platform SEED is enough; every episode is derived inside the referee | ☑ |
| Over-fitted reference | `tools/make_overfit.py` + `tests/test_rotation.py` (results in §4) | ☑ |
| Cosign identity + issuer | `https://github.com/unarbos/apex-competition-nist-assembly/.github/workflows/release.yml`, `https://token.actions.githubusercontent.com` | ☐ first run pending |
| `input_schema` + fixtures | `input.schema.json`, `fixtures/input.json` | ☑ |
| Reference/baseline submission | `baselines/baseline.onnx` (built by `tools/make_policies.py`); 0.337 mean over 20 seeds | ☑ |
| Test-set submissions | `test_policies/` (zero, constant, random, huge, NaN, wrong-shape, slow, over-fitted): all ≤ floor on new rounds | ☑ |
| Per-task records + reader | `/data/history/episodes.jsonl` + `traj_*.json`; `tools/read_records.py`; ~1.6 MB per evaluation | ☑ |
| Miner README | `README.md` | ☑ |
| End-to-end run | podman, both images, `--cpus 2 --memory 2g`, seed 11: raw 0.325 in 38 s; bit-identical to the non-container run with the same reserved file | ☑ |

Pins: python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1, numpy 2.5.4, mujoco 3.16.0, onnx 1.23.2, onnxruntime 1.31.0. There are no datasets or model weights. Part geometry is procedural, using NIST ATB-1 nominal sizes.

## 3. Ops parameters
| Parameter | Proposal | Reason |
|---|---|---|
| process_type | cpu | MuJoCo on CPU fits the envelope |
| kind | solo | assembly success is absolute |
| round_length_in_days | 2 | the eval is cheap, and longer rounds give more submissions on an identical set given the noise in §4 |
| submission_reveal_days | 4 | trained policies are real IP (production range 4–7) |
| lower_is_better | false | |
| baseline_raw_score / baseline_score | 0.36 / 0.0 | above the public baseline's 20-seed max (0.356), so resubmitting `baselines/baseline.onnx` cannot win round one. Do-nothing, random and constant policies measure 0–1.3% |
| resources | 2 CPU / 2Gi / 0 GPU | the referee uses 2 sim workers. Measured peak RSS: referee ≤ 561 MiB (sum of 3 processes, an upper bound), player 89 MiB |
| evaluate.timeout_s / referee.timeout_s | 900 / 900 | see §4 |
| deadline_ms | 500 per batched step | |
| Submission fee | ≈ $1 | norm |
| Incentive weight | 0.02–0.03 | new competition |

## 4. Round variation & evaluation sizing
- **Axes**, and the §1 property each tests:
  - clearance 0.15–0.6 mm (precision)
  - friction 0.5–1.2, part mass 5–50 g, controller gain ×0.7–1.3 (robustness to dynamics)
  - position-sensor noise 0–0.5 mm (perception robustness)
  - hole placement ±80 mm, start offsets/heights/yaw (generality of approach)
- **Derivation:** the platform SEED goes through `SeedSequence([SEED, const])`. Within each of the 40 strata, every axis is drawn by a Latin hypercube over its live range, so each round covers every range evenly.
- **Held constant:** 40 strata (2 shapes × 4 NIST sizes × 5 start types) × 16 episodes = 640. The episode budget (250 steps at 25 Hz) is fixed. This keeps difficulty stationary.
- **Difficulty stationarity:** baseline across 20 seeds: mean 0.337, sd 0.0087, range 0.327–0.356. A first version with per-round condition centres drifted (sd 0.067, range 0.18–0.44) and was replaced.
- **Reserved region:** part of the condition space is excluded from live rounds. It is derived at referee build time from a CI secret and ships only in the private referee image, and it is scored out of band.
- **Over-fitted reference:** `tools/make_overfit.py` memorises seed 11. Over 9 controller variants it keeps the first success for each episode (378/640) and serves the result as an ONNX nearest-neighbour table:

  | | seed 11 (own) | 12 | 13 | 14 |
  |---|---|---|---|---|
  | overfit | **0.591** | 0.105 | 0.114 | 0.116 |
  | baseline | 0.322 | 0.347 | 0.336 | 0.350 |

  It falls far behind the baseline off its own round, so the rotation bites. This is asserted in `tests/test_rotation.py`.
- **Sizing:** N = 640. σ_round = 0.0087 (20 seeds). Typical top score is unknown; at 0.34 the 1% margin is 0.0034. **σ ≤ ¼ × margin? NO.** σ is 2.6× the full margin and about 10× the target; meeting it would need ~68k episodes. This is flagged in the onboarding issue as the main open question.
- Reference solutions rank consistently across seeds: the baseline beats every test-set policy on all seeds tested.
- **Wall time** at N=640: 36–41 s on 2 CPUs (box, local); 38 s in containers.

**Timeout budget.**
- Worst case = startup ≤ 60 s + simulation ≤ ~90 s + 250 calls × (0.5 s deadline + 1.0 s client grace) = 375 s, plus records < 5 s. That is ≈ 530 s.
- `referee.timeout_s` = 900; `evaluate.timeout_s` = 900. Headroom 1.7×.
- **Fits.** Any late or failed call ends the evaluation immediately, so stalling shortens it.

## 5. Threat-model questionnaire

All 18 answers are complete. They are withheld from this public repository and the public issue, because written-down defenses are a map for miners.
They are available to Macrocosmos reviewers on request through a private channel (hello@macrocosmos.ai).

## 6. GPU justification: n/a (CPU)
