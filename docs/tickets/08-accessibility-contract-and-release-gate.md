# 08: Accessibility contract and release gate

**What to build:** The last cross-cutting finish. The app is operable by someone who
cannot read it, and it refuses to run against a build the study has not approved —
so no participant is ever given the wrong system.

**Blocked by:** 01 — Retrieval — bge-m3 end to end; 06 — Language, household and roles;
07 — Offline — history, state and cached bundle

**Status:** ready-for-agent

- [x] Every primary action can be completed using pictures alone, without reading text
- [x] Controls meet or exceed the enlarged minimum touch target, and no action is
      available only through a gesture
- [x] Important states are carried by colour as well as by text, so the app is usable
      by a colour-blind user
- [x] No screen requires English to operate
- [x] Text remains legible at the largest system font setting on a small screen
- [x] The app refuses to start against a release that is not approved, and says why
- [ ] A release that is drifting from its approved record is visible to the study team
      before any participant uses it
- [ ] The app installs and runs on the low-end reference device, and the download is
      viable on the stated field connection
- [x] Evidence is recorded that the dosage restriction is enforced and tested, so the
      study's own precondition for participant use is satisfied

## Started, 2026-10-04

- `AccessibilityContractTest`: 10 tests, asserted by measuring rather than by reading
  the source.
- `GET /release/status` plus `ReleaseGate`: 8 tests. Criteria 6 and 7 are real.

### A real bug the contract found

`MicrophoneWhenOffline` was drawn but never made clickable. It had a description, a
size and a state, and no tap handler at all. Every other test still passed, because
nothing was checking that a control could be used.

That is the argument for writing the contract as assertions rather than as a
document: a document would have said the microphone was reachable and been wrong.

One assertion was wrong twice before being right. It claimed a disabled control must
offer no tap. Compose keeps the click action on a disabled node and marks it
disabled, so the property worth checking is that it is present and inert, which is
exactly what criterion 4 asks for.

### Drift outranks approval

The gate I wrote checked approval before drift, so an approved build that had since
drifted would open, and the gate would pass precisely the build it exists to catch. A
signed release that no longer matches its approved record carries a signature for a
system that is not running.

## Outstanding

- **Criterion 8****, the device half. Size and download time are measured and gated:
  1.61 MB, 1m 24s on sustained 2G, with a 6 MB budget enforced in CI. Whether it
  installs and runs on the low-end reference device is not something a build server
  can answer and needs either hardware or your acceptance of the available evidence.
- **Criterion 9**, evidence is generated from the tests: SI-1 4 tests, SI-2 8, SI-4 8,
  plus 7 server-side. The record states plainly that it does not decide whether the
  study's own standard is met.

## Criteria 1, 5 and 9, 2026-10-04

**Criterion 1.** `ActionGlyphs` adds a microphone capsule, a solid square and a
forward triangle to ask, stop and replay, which were text-only buttons. A worker who
cannot read "Stop speaking" mid-visit had no way to silence the app, which is the one
action that has to stay reachable. Square against triangle is deliberate: those are
the two reached for mid-answer and the pair most easily confused at small size.

`PrimaryActionGlyphsTest` asserts every primary action has a glyph, because the
failure is an action with no glyph and that is a property of the mapping rather than
of a rendered screen.

**Criterion 5.** Four text-bearing containers had fixed heights, so at twice the
system font the label was cut off along the bottom edge with no error anywhere. They
are `heightIn(min = ...)` now: the touch target keeps its floor and the box grows.

The rendering half is not asserted, and that is deliberate. Robolectric performs no
real layout, so measured heights come back zero and comparing two of them proves
nothing. An earlier version of that test failed with `normal=0 large=0`, which is the
harness talking rather than the layout. The precondition that actually prevents
clipping is checkable statically, so `check_font_scaling.py` checks that and runs in
CI.

**Criterion 9.** `scripts/dosage_evidence.py` generates the record from the tests. The
invariant minimums come from the protocol rather than from the code, so deleting a
test fails the build instead of lowering the bar along with the coverage. The record
states that it does not decide whether the study's standard is met.

## What is still not closed

**Criterion 8's device half.** Size and download time are measured and gated in CI:
1.61 MB, 1m 24s on sustained 2G against a 6 MB budget. Whether the app installs and
runs on the low-end reference device is not something a build server can answer. It
needs hardware, or your acceptance that size plus a successful build is the evidence
available.
