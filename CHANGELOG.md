# Changelog

## 0.8.2 — 2026-11-21

- Retain credit for a stack ring that was mated in play and remains physically
  seated and undisturbed after final settling. Never-mated, displaced,
  knocked-off, and gripped rings score zero.
- Skip the carrier peg's finalization impulse only for a claimed, physically
  valid mated stack. Continue drawing its kick angle so unrelated random streams
  and impulses remain unchanged. Unmated rings and unrelated pegs retain their
  impulses; finalization still measures the settled pose and never re-pins.
- Orient mated rings to support each carrier peg at its measured slot yaw.
- Restore non-penetrating reset validation around intentional rework contacts,
  including clutter and active rings. Validate active-ring starts on rework-free
  boards too. Validate fallback starts before an episode is returned, preventing
  embedded parts from becoming settling projectiles.
- Add coverage for all 69 unparked stacks across four kick seeds, explicit
  support cases and board 42 without a kick, invalid claims, and fallback resets.
- Keep the `v08` profile, `[B,320]` observation, five-dimensional action,
  0.36 partial-credit cap, milestone weights, 920-action horizon, physical
  scoring tolerances, and released image digest pins unchanged.

## 0.8.1 — Scoring fixes

- Preserve completed-key credit from its physical completion latch and precision.
- Reject dependent-step re-lift farming and enforce the result metadata envelope.
- Verify final collar and stack poses without re-pinning during final settling.
- Require sustained bilateral grasp evidence for fractional lift credit.
