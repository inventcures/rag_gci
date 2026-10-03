# One app, three Care Roles, enforced server-side

The Android app serves all three user groups, ASHA Workers, Family Caregivers and Patients, from a single application. The audiences differ in literacy and in clinical authority, so they are separated by a Care Role chosen at Registration, and the backend enforces that role per endpoint rather than trusting the client.

This is not currently true of the codebase. `role` is stored as free text, embedded in the JWT, and read by nothing. `require_auth` only validates the token, and endpoints such as `/patient/{patient_id}` and `/fhir/export/{patient_id}` accept an arbitrary identifier with no ownership check. A single app cannot ship until those checks exist.

## Considered Options

**One app, one shared experience.** Cheapest, but it cannot safely serve a Patient alongside a clinician. The EVAH study compares what each group can do with the tool, so collapsing them removes the thing being measured.

**Separate apps per group.** Rejected. Three codebases for one API, and §3.1 of the study protocol assumes a shared install across groups at the same sites.

## Consequences

- Authorization moves to the backend. Every patient-scoped endpoint needs an ownership check, not just a valid token.
- `digital_literacy_score` is written and never read. If per-role UI tailoring is intended, it needs a consumer.
- PIN lockout in `auth/pin_auth.py` is permanent at 5 failed attempts with no reset path. For a low-literacy ASHA worker on a shared family phone this ends their account permanently, so it needs an unlock mechanism.
