# 08: Accessibility contract and release gate

**What to build:** The last cross-cutting finish. The app is operable by someone who
cannot read it, and it refuses to run against a build the study has not approved —
so no participant is ever given the wrong system.

**Blocked by:** 01 — Retrieval — bge-m3 end to end; 06 — Language, household and roles;
07 — Offline — history, state and cached bundle

**Status:** ready-for-agent

- [ ] Every primary action can be completed using pictures alone, without reading text
- [ ] Controls meet or exceed the enlarged minimum touch target, and no action is
      available only through a gesture
- [ ] Important states are carried by colour as well as by text, so the app is usable
      by a colour-blind user
- [ ] No screen requires English to operate
- [ ] Text remains legible at the largest system font setting on a small screen
- [ ] The app refuses to start against a release that is not approved, and says why
- [ ] A release that is drifting from its approved record is visible to the study team
      before any participant uses it
- [ ] The app installs and runs on the low-end reference device, and the download is
      viable on the stated field connection
- [ ] Evidence is recorded that the dosage restriction is enforced and tested, so the
      study's own precondition for participant use is satisfied
