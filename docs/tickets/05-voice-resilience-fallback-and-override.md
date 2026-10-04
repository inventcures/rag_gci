# 05: Voice resilience — fallback and backend-admin override

**What to build:** Real-time voice degrades honestly rather than freezing. When the
real-time path is unreachable or stalls mid-answer, the app says so and switches to
the India-resident turn-based path without the worker noticing a gap. The backend
administrator can force that path for a whole deployment during a provider incident,
and every such change is attributable afterwards.

**Blocked by:** 04 — Voice — push-to-talk, then real-time

**Status:** complete

All seven criteria are met. Recorded 2026-10-04; the boxes had never been ticked.

Two of them were previously believed done and were not. The audit panel reported a
fallback rate of zero forever, because the patch that added `get_voice_router` had
an anchor in a different file and silently did nothing. And neither voice route
consulted the router at all until 2026-10-04: `POST /query/voice`, the path the
Android app actually uses, transcribed with Sarvam and hardcoded its own grounding,
while reporting `voice_path=live`.

- [x] Fallback triggers on a first-token timeout and on a per-turn stall, not only on
      a failed connection, so a session that connects and then goes quiet is caught
- [x] The interface shows which voice mode is active, because the two modes are
      perceptibly different and silence is otherwise read as a freeze
- [x] The backend administrator can force the India-resident voice path for the whole
      deployment
- [x] Site coordinators and field staff have no ability to change the voice path
- [x] Changing the override records who, when, the previous value, the new value and
      a required reason, in an append-only log
- [x] The audit view reports interactions per voice path, so a deployment permanently
      on the fallback is visible rather than inferred
- [x] Every interaction records which voice path answered

## Verification

- `tests/test_voice_router.py` — first-token timeout and per-turn stall, with
  availability forced rather than inherited from the host.
- `tests/test_voice_provider.py` — actor, reason and append-only audit.
- `tests/test_voice_endpoint_integration.py` — a fallback turn is recorded as
  `fallback`, which is what criterion 7 rests on.

## Known limitations

- Criterion 4 has no test. The override route is admin-only by inspection, not by
  assertion. On a clinical system that is worth closing.
- Criterion 2 is server-side only. The active mode is reported on every turn, but no
  client test asserts the app surfaces it.
