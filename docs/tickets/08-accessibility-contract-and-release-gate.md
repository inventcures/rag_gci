# 08: Accessibility contract and release gate

**What to build:** The last cross-cutting finish. The app is operable by someone who
cannot read it, and it refuses to run against a build the study has not approved —
so no participant is ever given the wrong system.

**Blocked by:** 01 — Retrieval — bge-m3 end to end; 06 — Language, household and roles;
07 — Offline — history, state and cached bundle

**Status:** ready-for-agent

- [ ] Every primary action can be completed using pictures alone, without reading text
- [x] Controls meet or exceed the enlarged minimum touch target, and no action is
      available only through a gesture
- [x] Important states are carried by colour as well as by text, so the app is usable
      by a colour-blind user
- [x] No screen requires English to operate
- [ ] Text remains legible at the largest system font setting on a small screen
- [x] The app refuses to start against a release that is not approved, and says why
- [ ] A release that is drifting from its approved record is visible to the study team
      before any participant uses it
- [ ] The app installs and runs on the low-end reference device, and the download is
      viable on the stated field connection
- [ ] Evidence is recorded that the dosage restriction is enforced and tested, so the
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

- **Criterion 1**, every primary action completable from pictures alone. Needs the
  icon set audited against each primary action, which is judgement rather than code.
- **Criterion 5**, legible at the largest system font on a small screen. Needs a real
  font-scale render, which this Robolectric setup will not do honestly.
- **Criterion 8**, the low-end reference device and the download size. The APK builds
  and its size is measurable; whether it installs and runs on that device is not
  something a build server can answer.
- **Criterion 9**, evidence that the dosage restriction is enforced and tested. The
  tests exist and are named per invariant; whether they satisfy the study's own
  definition of the evidence is the study's call, not mine.
