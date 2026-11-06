# Competition description (v0.7.2)

Short text for the competition description field on Apex.

---

Train an ONNX policy that drives a simulated 2-finger gripper on a 4-DoF (x, y, z, yaw) stage through random compound assembly tasks on a NIST Assembly Task Board #1-style board (MuJoCo, CPU, 25 Hz). Every episode is a random Assembly III graph: pegs into matching holes (0.20 to 0.68 mm radial clearance on v07, per-episode layouts and cross-sections), collars on seated pegs, twist-locks, plus rework (extract a mis-seated peg), press-gated screws, stack rings on peg tops, and obs-indexed key yaw windows. Random ordering constraints link the steps, parts arrive in waves (at most 2 on the hardest tier), and friction, gains, mass, latency or sensor bias can change between subtasks without notice. Pose estimates are biased, drifting and glitchy, so contact (wrist force/torque) matters. Four complexity tiers run up to near-impossible. Partial credit is weighted toward keys, stacks and rework (insert 0.30, collar/twist 0.25, screw 0.40, rework 2.10, stack 2.60, key 3.40; key/stack/rework are all-or-nothing). Cost is bounded by the sim horizon only — same policy and seed score the same under any machine load. Observation stays 288 floats with a 184-d v0.5-compatible prefix. The round score is the mean over 64 episodes.
