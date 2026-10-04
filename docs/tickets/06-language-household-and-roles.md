# 06: Language, household and roles

**What to build:** A worker sets their language once and the app answers in it. A
family sharing one phone can hand it back and forth, and whoever is holding it can
see at a glance whose view they are seeing — without the app pretending to offer
privacy it cannot deliver.

**Blocked by:** 03 — Walking skeleton and client safety core

**Status:** ready-for-agent

- [ ] Language can be chosen once and is remembered
- [ ] All eleven supported languages are selectable, and every screen and spoken
      prompt exists in each
- [ ] An unsupported language is refused plainly rather than answered in another
      language; the app never silently substitutes one the user did not choose
- [ ] A household can be registered, and an observation recorded against the right
      member
- [ ] Everyone in a household reads the same household record
- [ ] Switching role takes one tap and requires no PIN, and is never presented with a
      lock or shield, because it is not a privacy control
- [ ] The active role is shown as a persistent icon large enough to be recognisable
      without reading
- [ ] The three role icons differ in silhouette and posture, not only in detail, and
      are distinguishable at small size by someone who cannot read the label
- [ ] A participant cannot read another household's record by guessing an identifier

## Progress, 2026-10-04

Started. The data layer is done and tested:

- `HouseholdEntity`, `HouseholdMemberEntity`, `HouseholdDao`.
- Cross-household isolation is enforced in the SQL, not by the caller. Every read
  joins on `owner_id`, and there is deliberately no `getHousehold(id)` overload that
  could be used to bypass it. ADR 0006 leaves this as the one boundary that is real.
- Schema 1 to 2 migration written out in full. `fallbackToDestructiveMigration`
  stays forbidden, so an upgrade keeps the recorded interactions the study depends
  on.
- `HouseholdDaoTest`: 6 tests against a real Room database, including a guessed
  member identifier from another household returning null.

### One bug found on the way

`OnConflictStrategy.REPLACE` is implemented as DELETE then INSERT, and the foreign
key on `household_members` is `ON DELETE CASCADE`. Registering or re-syncing a
household therefore deleted every member of it, silently, and the household came
back empty. `@Upsert` updates in place and the members survive. Caught by a test
that registered two members and found one.

## Still to do

Language selection and persistence, the eleven-language string coverage, plain
refusal of an unsupported language, the household UI, the role switch, and the three
role icons differentiated by silhouette.
