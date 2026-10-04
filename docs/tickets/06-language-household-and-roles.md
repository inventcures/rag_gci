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
