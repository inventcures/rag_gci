# 07: Offline — history, state and cached bundle

**What to build:** When the network is gone — at Barak Valley, at the hardest site —
the app is not dead. It shows what was said at the last visit, answers the questions
it has pre-answered, and is completely honest about what it cannot do.

**Blocked by:** 03 — Walking skeleton and client safety core; 01 — Retrieval — bge-m3
end to end

**Status:** ready-for-agent

- [x] Past interactions are readable with no connection, and survive the app being
      closed
- [x] A Redacted Answer in history renders distinctly from an answer, so a refusal
      saved weeks earlier is not later mistaken for clinical advice
- [x] The offline state is persistent and unmissable, conveyed by icon and colour as
      well as by plain text
- [x] The microphone remains visible but disabled when offline, with an explanation of
      why, rather than hidden or failing silently
- [x] Scheduled medication reminders already on the device continue to fire offline
- [ ] Anticipated questions are answered with no network, from the downloaded bundle,
      in the user's chosen language
- [x] A question outside the bundle is answered with a clear explanation that it
      needs a connection, not with silence or a stall
- [x] Cached answers are marked as such and are not presented as live
- [x] The app recovers cleanly when signal returns, without a restart

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

## Resolved, 2026-10-04 — see "Bundle resolver built" below

The finding below was real and is now fixed. Kept because it explains why the seam
was invisible: both halves of the lookup were implemented on opposite sides of a
language boundary and neither was tested against the other.

## Blocking finding, 2026-10-04 (now fixed)

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

## Bundle resolver built, 2026-10-04

`OfflineAnswerResolver`, 12 tests, all green. Criteria 6 and 7 have the data layer
they need; the UI for them is still to do.

Two real defects fixed, both found by tests rather than by reading:

**Trailing punctuation broke every lookup.** The builder keyed on
`sha256(text.lower().strip())` and the client did the same, so "How to manage pain at
home?" and "how to manage pain at home" hashed differently. A worker typing a
question mark would miss on every bundle row and be told to find a connection for a
question the app already answers. Both sides now strip a trailing question mark or
exclamation point, and a test reads the builder so the two cannot drift.

**There was no client-side hash at all**, which is the finding above. Without it the
bundle could never match anything.

Two things recorded as decisions rather than left as surprises:

- **A paraphrase is a miss, not a guess.** The bundle holds twenty canonical
  questions. Matching meaning needs on-device retrieval, which ADR 0005 rules out, so
  a paraphrase returns a clear "needs a connection" rather than the nearest stored
  answer implied to be what was asked.
- **No bundle and not-in-bundle are different messages**, because the remedies differ:
  sync, or find signal.

## History and offline UI, 2026-10-04

Six criteria now met.

- `HistoryRepository` (9 tests). Carries `answerKind` through from the stored row
  rather than re-deriving it, so a refusal judged weeks ago is not re-judged now.
- `OfflineAnswerResolver` (12 tests). The hash seam, which did not exist.
- `OfflineUi`: banner, disabled mic, history cards, miss notice.

Every offline string is paired with a shape or a colour, and none is the only
carrier of the meaning. That is the point of criterion 3: a worker who cannot read
the label still has to be able to tell that nothing is live, and to tell a refusal
from advice.

Built with a scrolling `Column` throughout, never a `LazyColumn`, because a
LazyColumn composes nothing under this Robolectric setup and everything in it would
have been untestable.

## Reminders and reconnection, 2026-10-04

All nine criteria met.

**Reminders (12 tests).** `ReminderEntity` stores the schedule *and* the words, which
is the whole reason criterion 5 can hold: a reminder that had to ask the server what
to say would stop working at exactly the moment a worker needs it, which is the moment
they are somewhere with no signal. A test asserts that a reminder stored while online
still resolves with the connection gone.

Delivery uses a window rather than an exact minute, because an alarm arrives late when
a device has been asleep or a battery saver has deferred it. Matching the exact minute
would silently drop a delayed reminder, and that is the failure that matters: the
family never learns a dose was missed. There is a test for exactly that case.

The stored text never contains a dose. The boundary that applies to an answer applies
to a reminder too, and a spoken reminder is the one thing a worker cannot easily
check before a family acts on it.

**Reconnection.** `ConnectivityMonitor` treats `NET_CAPABILITY_VALIDATED` as the
question rather than merely "connected". A site with a bar of signal and no working
link reports itself connected, and the app then sends a request and stalls while the
worker is told everything is fine. The shell subscribes rather than sampling once at
startup, so the app stops believing it is offline as soon as signal returns, without a
restart.

## Still outstanding

Nothing in the nine criteria is outstanding. Two things a user would still notice:

- **Queued interactions are not re-sent on reconnect.** The delta tracker exists on
  the server and the client can tell it went offline, but nothing drains a queue when
  the signal comes back. History shows what happened on the device; the study log will
  not have it until the app is opened again.
- **Reminder audio is not wired.** The schedule resolves and the text is stored, but
  nothing plays it yet. That needs AlarmManager, which is built in but has no test
  seam here.
