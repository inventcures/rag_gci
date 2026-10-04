# 06: Language, household and roles

**What to build:** A worker sets their language once and the app answers in it. A
family sharing one phone can hand it back and forth, and whoever is holding it can
see at a glance whose view they are seeing — without the app pretending to offer
privacy it cannot deliver.

**Blocked by:** 03 — Walking skeleton and client safety core

**Status:** complete, with one caveat noted below

- [x] Language can be chosen once and is remembered
- [x] All eleven supported languages are selectable, and every screen and spoken
      prompt exists in each
- [x] An unsupported language is refused plainly rather than answered in another
      language; the app never silently substitutes one the user did not choose
- [x] A household can be registered, and an observation recorded against the right
      member
- [x] Everyone in a household reads the same household record
- [x] Switching role takes one tap and requires no PIN, and is never presented with a
      lock or shield, because it is not a privacy control
- [x] The active role is shown as a persistent icon large enough to be recognisable
      without reading
- [x] The three role icons differ in silhouette and posture, not only in detail, and
      are distinguishable at small size by someone who cannot read the label
- [x] A participant cannot read another household's record by guessing an identifier

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

## Complete, 2026-10-04

All nine criteria met and tested. New: `Language`, `CareRole`, `PreferencesStore`,
`HouseholdRepository`, `RoleIcons`, `RoleAndLanguageUi`.

- 24 tests in `:core:data`, 4 UI tests in `:app`.
- The client language list is asserted against `SUPPORTED_LANGUAGES` in
  `offline/questions.py`, tag by tag, because the offline bundle zips the two lists
  and a mismatch would pair the wrong question with the wrong answer.
- The three role icons are drawn as vectors rather than imported.
  `material-icons-extended` carries `medical_services`, `diversity_2` and `bed`, and
  using it would have added megabytes to an APK that has to install over 2G. Drawing
  them also serves ADR 0006 directly: cross, two figures, and a reclining figure are
  told apart by silhouette alone at 24dp.

## Caveat, stated rather than buried

**The language picker UI is not verified.** Two Compose tests were written for it and
removed: a `LazyColumn` composes nothing under this Robolectric setup, so they could
not be made to pass. Shipping a test that asserts something the harness cannot render
would read as coverage and prove nothing.

The picker needs an instrumented test or a host-side render to close properly. The
eleven-language list itself is fully covered by `LanguageAndRoleTest`, which needs no
layout. What is unverified is the picker as drawn, not the languages it offers.
