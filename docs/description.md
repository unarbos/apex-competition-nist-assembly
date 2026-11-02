# Competition description (v0.6.1)

Short text for the competition description field on Apex.

---

Train an ONNX policy that drives a simulated 2-finger gripper on a 4-DoF (x, y, z, yaw) stage through random compound assembly tasks on a NIST Assembly Task Board #1-style board (MuJoCo, CPU, 25 Hz). Every episode is a random assembly graph: pegs go into matching holes (0.25 to 0.8 mm radial clearance, per-episode layouts and cross-sections), collars must be dropped over seated pegs, and twist-locks need seated round pegs turned past a random angle. Random ordering constraints link the steps, parts arrive in waves, and friction, gains, mass, latency or sensor bias can change between subtasks without notice. Pose estimates are biased, drifting and glitchy, so contact (wrist force/torque) matters. Four complexity tiers run up to near-impossible, and every step earns partial credit, weighted toward collars and twists. Insert-only controllers from earlier rounds score about 0.2; a privileged solver that reads the true state scores about 0.41 overall and 0.24 on the hardest tier. The round score is the mean over 64 episodes.
