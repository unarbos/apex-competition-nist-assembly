# Competition description (v0.8.0)

Short text for the competition description field on Apex.

---

Train an ONNX policy that drives a simulated 2-finger gripper on a 4-DoF (x, y, z, yaw) stage through random compound assembly blueprints on a NIST Assembly Task Board #1-style board (MuJoCo, CPU, 25 Hz). Every episode is a freshly generated Assembly IV graph of many steps (typically about 8 to 28): peg inserts, collars, twist-locks, rework (extract a mis-seated peg), press-gated screws, stack rings on peg tops, and yaw-index keys. Random ordering constraints link the steps. Parts arrive in waves. Mass, friction, gain and latency are stretched per episode and can change between subtasks, so policies must identify the system online. Geometry, tolerances and sensor bias are redrawn every episode. The observation is 320 floats (288-d Assembly III layout plus a 32-d blueprint encoding). Absolute key yaw is never published; there is no closeness, detent or retarget cue. Score is the mean over blueprint steps of physically verified end-state credit that survives a perturbation test, times precision — equal weight per step, with a small frontier partial. A perfect 1.0 is effectively unreachable. Cost is bounded by the sim horizon only. The round score is the mean over 64 episodes.
