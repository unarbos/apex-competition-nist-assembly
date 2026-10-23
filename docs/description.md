# Competition description (v0.5.0)

Short text for the competition description field on Apex.

---

Train an ONNX policy that drives a simulated 2-finger gripper on a 4-DoF (x, y, z, yaw) stage through multi-peg assembly on a NIST Assembly Task Board #1-style peg board (MuJoCo, CPU, 25 Hz). Each episode has 3 to 6 pegs to insert, in any order, each into its own matching hole, with radial clearances of 0.25 to 0.8 mm, loose and lying starts, clutter blocks, and randomized physics. Since v0.5.0 every episode draws its own board layout: hole positions, orientations, sizes (4 to 16 mm) and cross-sections (round, rectangle/square, 5/6/8-sided polygons, D-flat). There is no fixed template and no shape label. Peg and hole poses in the observation carry per-episode estimate biases of several millimetres, so policies have to locate holes through contact (wrist force/torque is observed) rather than trust the reported pose. The best hand-coded controller we have scores about 0.21, while a privileged solver that reads the true state scores about 0.70; the gap between them is the room left for learned policies. Each peg scores 0 unless it is seated and released at episode end; a seated peg scores 0.75 plus up to 0.25 for speed and smooth actions. The round score is the mean over 128 episodes.
