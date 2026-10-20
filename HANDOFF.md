# Competition Onboarding Manifest: `nist_lite_assembly` v0.2.1

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
| Competition repo (public) + release tag | `https://github.com/unarbos/apex-competition-nist-assembly` @ `v0.2.1` (spec-only; images from `v0.2.0`) | ☑ |
| `spec.yaml` version `0.2.1`, v0.2.0 digests pinned | `spec.yaml` | ☑ |
| Player image | `ghcr.io/unarbos/apex-competition-nist-assembly-player@sha256:3c5c1d048b8e2d4ff541a14a37a9c777872727f82456ec71865897f24b6b8652` | ☑ |
| Referee image (**private package**) | `ghcr.io/unarbos/apex-competition-nist-assembly-referee@sha256:ff3fb58c1e5d8e9154545e79c6b98022fecccdbc3e3c07fd9613a5a2360f2943` | ☑ |
| Held-out `reserved.json` from build secret | referee Dockerfile `reserve.py` | ☑ mechanism |
| Reference baseline ONNX | `baselines/baseline.onnx` | ☑ |
| Junk test-set policies | `test_policies/` | ☑ |
| Calibration across ≥10 seeds | `CALIBRATION.md` | ☑ |
| Miner README + AlphaNIST section | `README.md` | ☑ |

Pins unchanged: python:3.12-slim@sha256:a6e34c…, numpy 2.5.4, mujoco 3.16.0, onnx 1.23.2,
onnxruntime 1.31.0. Image digests from release run 39363449941 (tag `v0.2.0`), both cosign-verified.

## 3. Ops parameters
| Parameter | Proposal | Reason |
|---|---|---|
| process_type | cpu | MuJoCo on CPU |
| kind | solo | absolute score |
| round_length_in_days | 1 | platform norm |
| submission_reveal_days | 4 | |
| lower_is_better | false | |
| baseline_raw_score / baseline_score | 0.0 / 0.0 | platform convention |
| resources | 2 CPU / 2Gi / 0 GPU | 2 sim workers |
| evaluate / referee timeout_s | 1800 / 1800 | 128-episode live rounds took ~400–660 s on Macrocosmos staging |
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

## 7. Supersedes
PR #6 (`v0.1.6` efficiency-on-single-peg) is superseded by this multi-peg redesign. The
efficiency idea is reused inside the per-peg `0.75 + 0.25·time` term.
