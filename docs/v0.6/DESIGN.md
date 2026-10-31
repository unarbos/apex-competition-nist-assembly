# NIST-Lite v0.6 "Assembly II" — design draft (work in progress, branch `v0.6-dev`)

Not released. The live task stays v0.5.x until a v0.6 release is reviewed with Macrocosmos.

## Goal
Make the task much harder and bigger so that winning needs learned perception and control, not a
hand-written controller compiled into ONNX, while keeping a high privileged ceiling and a clear gap a
learned policy can close.

## Planned changes (profile `v06`, `referee/nist_sim/profile.py`)
1. **Rawer perception.** Coarse part/hole regions instead of per-object pose estimates, plus a noisy
   16x16 wrist depth patch and the existing wrist force/torque.
2. **Stateful policies.** ONNX `(obs, state) -> (action, state')`, state zeroed at reset by the referee.
3. **Bigger assemblies.** 12-slot board, K in {4, 6, 8}, collars that fit only over a seated peg.
4. **Tighter tolerances.** Clearance 0.10–0.50 mm, some holes without chamfer.
5. **Held-out families per round.** The round seed holds out shape families from the live mix.
6. **Robust scoring.** Action quantization in the referee (policies that behave the same score the
   same), held-out subscores weighted in the round score.

## Eval budget
Target: projected live eval time at or under ~400 s on 2 CPU. `tools/calib_v06.py` measures per-episode
cost and projects live wall time; anything over budget is cut (episodes, horizon, depth-patch rate) or
raised with Macrocosmos.

## Status
- [x] Profile switch (`NIST_PROFILE=v05|v06`, default v05 = live task unchanged)
- [x] Referee action quantization (v06 profile only)
- [x] Calibration harness `tools/calib_v06.py`: paired deltas with SE, epsilon-copy reroll test,
      per-episode wall time, projected live eval time, release bars
- [ ] Depth patch + coarse prior observation
- [ ] Stateful ONNX protocol
- [ ] 12-slot board, collars, precedence, per-step scoring
- [ ] Tighter clearance, per-round held-out families
- [ ] v0.6 privileged solver, scripted adversary, PPO/BC baseline, calibration vs bars
