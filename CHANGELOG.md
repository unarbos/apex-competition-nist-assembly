# Changelog

## 0.8.5 — 2026-11-27

- Read complete pending worker frames before applying the transport stall timeout,
  preventing parent decode time from being counted against a mid-send worker.
- Start intentionally mis-seated rework pegs above incompatible wrong-hole walls
  instead of penetrating them; extraction, seating, and fallback validation remain
  unchanged.
- Add a 384-board reset-penetration regression and exact-zero/deterministic-record
  checks for zero and random policies. Scoring, observation/action contracts,
  milestone weights, task rules, the 920-action horizon, resources, and released
  image digest pins remain unchanged.

## 0.8.4 — 2026-11-24

- Accept the legitimate `done_at: -1` sentinel after a physical misseat, tied to
  the event's public peg identity. Preserve valid summary rows byte-for-byte and
  contain an invalid episode row to its own
  deterministic `worker_failure` zero instead of discarding its worker siblings.
  Missing, duplicate, or misordered row identities and invalid summary envelopes
  still fail the whole worker slice.
- Drain a worker's complete final frame through EOF after child exit, retaining
  malformed/truncated-frame failure handling. Restore send-to-all-then-receive
  observation/action scheduling while preserving deterministic batch ordering,
  pending-action delivery, and worker-death lockstep.
- Remove v0.8's synthetic key-detent path without reverting genuine wrist-yaw
  torque. Preserve collar and stack flags in key pegs' shared observation channel.
- Document the generator's 20-step structural maximum, the 0.312 reachable
  partial-credit maximum under the unchanged nominal 0.36 clamp, the first-12-step
  detailed blueprint encoding, and the 920-action horizon. Observation/action
  contracts, milestone weights, task rules, scoring, resources, and released
  image digest pins remain unchanged.

## 0.8.3 — 2026-11-23

- Contain episode-construction failures in referee workers. A failed reset
  produces a deterministic zero with `reset_failure` / `failure_reason` in its
  public record and stays in the fixed 64-episode scoring denominator; other
  episodes continue normally.
- Close child pipe endpoints in the parent after forking and close inherited
  parent endpoints in each child before the next fork. Poll pipe readability
  and child liveness instead of blocking indefinitely, recover from EOF or
  malformed frames, and produce complete deterministic `worker_failure` zero
  records for a failed worker's slice while preserving already-contained reset
  failures. Resynchronize surviving workers after a partial action delivery.
  Reject invalid or non-JSON summary payloads before aggregation. Live episode
  construction and simulation have no additional wall-clock scoring cutoff;
  only a partial live frame or a backpressured send with no transport progress
  for five seconds is contained as a worker failure. Workers that die before
  the next fork can be collected while peers are still constructing episodes;
  a failed readiness launch explicitly terminates and reaps every started worker.
- Exempt a claimed stack's carrier peg from its finalization impulse only when
  `seated_now` is true; a merely in-hole carrier still receives its kick. Keep
  every kick-angle draw unchanged, including ordinary collars' 0.63 m/s kicks.
- Add forced-reset, repeated-record, dead-worker, malformed-frame,
  action-delivery recovery, and non-upright-carrier regressions. Keep
  observation/action contracts, task rules, partial-credit
  scoring, horizon, resources, and released image digest pins unchanged.

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
