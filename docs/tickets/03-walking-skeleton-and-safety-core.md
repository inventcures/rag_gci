# 03: Walking skeleton and client safety core

**What to build:** An ASHA Worker opens the app, types a question, and gets a
grounded answer back — and the four safety invariants hold on that first answer. The
dose rule, the Redacted Answer rendering, emergency severity and study
instrumentation are built in from the first slice rather than added later, because a
regression in any of them is a protocol breach rather than a bug.

**Blocked by:** 02 — Project and contract seam

**Status:** complete

All eight criteria are met. Recorded 2026-10-04; the boxes had never been ticked.

Criterion 7 took the longest and had two wrong diagnoses on the way. Six Robolectric
tests failed and the cause was not the network. Room's suspend DAO methods dispatch to
its transaction executor, and under Robolectric that executor never resumes the
continuation, so the repository hung inside `insert()`. The tests were reporting an
incomplete repository as a safety failure. Blame went first to Moshi and then to
OkHttp, both wrong, before the real cause was found.

- [x] A typed question returns a grounded answer with its evidence level shown
- [x] A response containing a specific dose has the dose-bearing sentences removed
      and the remainder kept, and is never rendered as a plain answer
- [x] A response that is entirely dosing becomes a Deferral that names concrete next
      steps and does not state or imply that a clinician has been contacted
- [x] A Deferral or Redacted Answer can never be mistaken for clinical advice in the
      rendered result
- [x] Emergency severity survives to the client; a high-severity phrase such as
      "severe pain" is never escalated to critical, so it cannot override the Dose
      Boundary
- [x] Every interaction is recorded with care role, locale, connectivity, whether the
      answer came from the cached bundle, and the release identifier
- [x] The suite exercises the real database, repositories and view models through the
      user interface, mocking only network, clock, randomness and hardware
- [x] Each safety invariant has its own named test that fails with the invariant named

## Verification

- Kotlin safety core: 35 tests, named per invariant. `SI-1` dose detection, `SI-2`
  redaction and deferral, `SI-4` emergency severity and the boundary override.
- `DoseBoundaryTest` covers the cases that matter clinically rather than the easy
  ones: Indic numerals, one sentence carrying the dose, and over-firing treated as a
  defect.
- Six Robolectric tests exercise the real database, repository and view model,
  mocking only the network.
- `InteractionEntity` records care role, site, release identifier and approval,
  language, channel, offline and cache flags, voice path, answer kind, emergency
  level and whether the dose boundary fired.
