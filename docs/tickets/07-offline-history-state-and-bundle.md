# 07: Offline — history, state and cached bundle

**What to build:** When the network is gone — at Barak Valley, at the hardest site —
the app is not dead. It shows what was said at the last visit, answers the questions
it has pre-answered, and is completely honest about what it cannot do.

**Blocked by:** 03 — Walking skeleton and client safety core; 01 — Retrieval — bge-m3
end to end

**Status:** ready-for-agent

- [ ] Past interactions are readable with no connection, and survive the app being
      closed
- [ ] A Redacted Answer in history renders distinctly from an answer, so a refusal
      saved weeks earlier is not later mistaken for clinical advice
- [ ] The offline state is persistent and unmissable, conveyed by icon and colour as
      well as by plain text
- [ ] The microphone remains visible but disabled when offline, with an explanation of
      why, rather than hidden or failing silently
- [ ] Scheduled medication reminders already on the device continue to fire offline
- [ ] Anticipated questions are answered with no network, from the downloaded bundle,
      in the user's chosen language
- [ ] A question outside the bundle is answered with a clear explanation that it
      needs a connection, not with silence or a stall
- [ ] Cached answers are marked as such and are not presented as live
- [ ] The app recovers cleanly when signal returns, without a restart

## Survey before starting, 2026-10-04

Started, not built. Recorded so the next session does not re-derive this.

**Both blockers are complete, so this is unblocked.** Nothing in the nine criteria is
met yet, and nothing was written.

What already exists, and is the reason this is cheaper than it looks:

- `CachedAnswerEntity` carries `queryHash`, `language`, `queryText`, `response`,
  `answerKind`, `bundleVersion` and `fetchedAt`. `answerKind` on a cached row is what
  makes criterion 2 possible: a redacted answer saved weeks ago can still render
  differently from a live one.
- `PalliSahayakRepository` already has `VOICE_PATH_CACHE = "cache"`, an
  `offlineOrUnavailable(...)` path, and an `isOffline` flag on the session, and the
  network failure already falls back rather than throwing.
- The server side of the bundle is done: twenty questions per language in
  `offline/questions.py`, built by `offline/cache_builder.py` through the mobile API
  bundle route. `LanguageAndRoleTest` already asserts the client's eleven languages
  match `SUPPORTED_LANGUAGES`, which is what the bundle is zipped against.

Worth knowing before writing anything:

- **A `LazyColumn` composes nothing under this Robolectric setup.** That is why the
  T6 language picker has no UI test. Anything in T7 that needs a scrolling list will
  hit the same wall, so plan for a non-lazy layout in the offline screens.
- `InteractionEntity` already has `fromCache` and `interactionClass`, so criterion 9
  (recover cleanly on reconnect, no restart) has somewhere to record what happened.

Order worth following: criteria 2, 6, 7 and 8 are the safety and honesty ones and
depend only on data that already exists. Criteria 4 and 5 are UI and carry the same
Robolectric risk as T6. Criterion 9 needs the delta tracker, which exists in
`sync/delta_tracker.py`.

## Blocking finding, 2026-10-04

**The bundle can never answer a question. The client has no hash implementation.**

`offline/cache_builder.py:99` writes each row's key as
`hashlib.sha256(query_text.lower().strip().encode()).hexdigest()`. Nothing on the
Android side computes that. There is no SHA-256 of the question anywhere in
`core:data`, so every lookup key the client could invent would miss.

This is not a missing feature so much as a missing seam: the two halves of the same
lookup were implemented on opposite sides of a language boundary and never tested
against each other. Until it exists, criteria 6 and 7 cannot pass, and an offline
app would report every question as a miss, which looks exactly like an empty bundle.

The implementation was written and passes a scratch check, but **it could not be
committed**, for the reason below.

## Unexplained toolchain failure

A new class in `core:data` compiles to a `.class` file that the unit-test compile
cannot then resolve, while sibling declarations in the same file resolve fine:

```
OfflineMiss.class          visible to the test
OfflineResolution.class    visible to the test
OfflineAnswerResolver.class compiled, on disk, NOT visible
```

Ruled out by measurement, not assumption:

- the file is in the source set — injecting a syntax error produced the expected error
- the class file is produced and is a valid class file
- `clean`, `--rerun-tasks`, `--no-configuration-cache`, `--no-build-cache`, and
  deleting `core/data/build` and `.gradle` by hand all change nothing
- renaming the package `offline` to `bundle` and then to `data.repository` changes
  nothing
- renaming the class changes nothing
- removing the unused `clock` parameter changes nothing

New files added to this module earlier in the session work, so it is not "new files
are invisible". The tree has been left clean and the suite green; nothing from this
attempt was committed.

Next step for whoever picks this up: reduce it to the smallest file that reproduces
it in a throwaway branch. Guessing further has already cost more than the feature.
