# Care Role is an analysis label, not an access boundary

Everyone in a household reads the same Household Record. Switching Care Role changes what the app surfaces first, how it frames answers, and which voice it speaks in. It does not partition the record, and the app never claims that it does.

The roles were chosen to fit how these households actually work. A patient recovering at home and the relative caring for them share one phone, because a second device is not affordable and two people who share a home already share a trust boundary. Forcing per-person devices would exclude the exact population the study targets.

## Considered options

**Scope data by role.** The patient role would not read the caregiver's notes. Rejected because it teaches a model that is false on a shared device: a caregiver who can see the household record anyway gains nothing from the rule, while the patient is told something untrue about their own privacy. A switch that appears to protect and does not is worse than no switch at all, because the user stops checking.

**One household account, no roles.** Honest and cheapest, but it discards the three-group comparison that is an EVAH outcome. Role survives as an attribute of each interaction rather than as a login.

## What this preserves

Role still attributes every interaction, which is what the protocol needs. It is the stratification variable for comparing ASHA Workers, Family Caregivers and Patients, and it is logged on every row. Nothing is lost analytically; only the fiction of per-role record privacy is dropped.

## Consequences

- The app must never present the role switch as a privacy control. It is labelled as framing ("Speaking as"), switching costs one tap with no PIN, and no lock or shield glyph appears anywhere near it. A PIN on the switch would imply a boundary that does not exist.
- **Active Role** is shown persistently as a large icon, so whoever is holding the phone can see whose view they are seeing. Users may be semi-literate, so this is carried by icon and colour as well as by any label.
- `role` in the JWT is now an analysis attribute. The ownership checks called for in ADR 0001 are still needed, but for a different reason: not to separate household members from each other, but so that a participant cannot read *another household's* record by guessing an identifier. That is a cross-household boundary and it is not negotiable.
- The three role icons are differentiated by context object and posture rather than by person figures, because three similar human silhouettes are indistinguishable at small sizes to a semi-literate user: `medical_services` for the ASHA Worker, `diversity_2` for the Family Caregiver, `bed` for the Patient.
- If the protocol later requires within-household separation — for example if a young patient and an older relative share a device in a way that makes household trust untenable — this decision is the one to revisit, and it should be revisited as a consent design question rather than as a permissions change.