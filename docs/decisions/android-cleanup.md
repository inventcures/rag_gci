# Android app cleanup: decision log

A review of `android-app/` on main at `a40a8b6`, asked for on 2026-11-19: find what is broken, inconsistent, inefficient or no longer needed, fix what is certain, and list the rest for the project lead. The rest of the repository comes later.

Every Kotlin source and build file was read, and every Android call was checked against the server it talks to (`mobile_api/router.py`, `voice_session.py`). The committed API contract matches the server schema (`scripts/export_openapi_contract.py --check`), and the generated client matches the contract.

The cloud session cannot download the Android SDK, so the GitHub Android job is the only build and test run for these changes. The Python gates (generated docs, dose evidence, English-literal lint, fixed-height lint) were run locally.

## Decisions

| # | Decision | Why |
|---|---|---|
| A1 | Fix only what is certain and does not change clinical copy. Everything else is listed under "For review" with a recommendation. | tp53's instruction for this round. The 76 Hindi strings still await clinical review, so no string resource was added or changed. |
| A2 | `android-app/app/build/` needs no change. It is already gitignored and has never been committed; it exists only in the local folder. The one committed build artifact was `.kotlin/sessions/*.salive`, Kotlin daemon state, which is removed and `.kotlin/` ignored. | `git ls-files android-app` lists no `build/` path. |
| A3 | Delete `RealtimeVoiceTransport`. | Nothing constructed it. It could not have worked: each turn made a new listener that was never attached to the socket, so a turn never resolved, and it connected to `/ws/voice`, the web Live socket, not the mobile `/api/mobile/v1/ws/voice`. It is in git history if a socket transport is wanted later. |
| A4 | Leave the rest of the voice path in place and fix only its two plain defects: a released mic sent nothing unless a test had injected a coroutine scope, and the hold-to-speak gesture took the first finger movement as the release. | Voice is not wired into the app at all (see R1). Removing it is a product choice; these two fixes are needed whichever way that goes, and they change nothing a participant sees today. |
| A5 | The offline banner counts the Cached Answer Bundle for the language in force, not past interactions. | The string says "%d questions are available without a connection", and the app passed the number of history rows. |
| A6 | Offline, the Ask screen shows the existing disabled-microphone control with its reason ("Voice needs a connection"). | `AskScreen` took an `offline` flag documented as criterion 4 and ignored it; the control and its translated strings existed but were only reachable from a test. |
| A7 | Give the History tab a real icon (a clock). | Its glyph was two zero-width lines, which fill nothing, so the tab drew blank. Navigation by shape is a stated requirement. |
| A8 | Add an OkHttp disk cache (10 MB) and skip storing a bundle the server answers with 304. | The server sends the bundle with an ETag and answers a matching `If-None-Match` with 304, but the app had no cache, so it never sent one and downloaded the whole bundle on every reconnection over 2G. Responses without a validator (the release status) are not served from the cache, so the release gate is unaffected. |
| A9 | Request logging in debug builds only. | A release phone gained nothing from a logcat line per request. |
| A10 | The connectivity monitor follows the default network and reports online only once it is validated. | It reported online the moment any network appeared, before validation, which is the stall its own documentation says it prevents, and a second unvalidated network (captive Wi-Fi beside working data) could flip the app offline. From API 26 (the app's minimum) a capabilities callback always follows `onAvailable` on the default network. |
| A11 | History reads the latest 50 interactions, not the whole table. | The history screen is documented as bounded at fifty, but the query had no limit, so every new answer re-read and redrew every interaction recorded on the phone. |
| A12 | An unreadable server reply at start or sign-in counts as "could not reach the server" rather than crashing. | Only `IOException` was caught; a Moshi `JsonDataException` (a proxy error page, an older schema) escaped from the start screen or the sign-in coroutine and crashed the app. The bundle download and sync already handled it this way. |
| A13 | Remove code nothing calls and fix comments that described something else: `TextScrubber.looksLikeIdentifier` (said "used by the repository"; it was not), `AskUiState.isDeferral`, two unused voice path constants, `ServiceLocator.syncQueue`, three unused dependencies of `:core:api` (one of them `coroutines-android` in a JVM module). The inverted `Scrubbed.isClean` comment, the misplaced KDoc of `ask()`, and two glyph comments are corrected. | Dead code that claims behaviour misleads the next reader. Each removal was checked for callers in main and test sources. |
| A14 | Two tests added: a 304 bundle is not stored again, and an unreadable release status falls back like no signal. | They pin A8 and A12. The other fixes are UI wiring or platform callbacks that the JVM tests cannot reach. |

## For review

Ordered by how much they matter. Each has a recommendation.

| # | Finding | Recommendation |
|---|---|---|
| R1 | **Voice is not wired in.** No `VoiceEngine` is ever built, so the hold-to-speak button does nothing online, and nothing asks for the microphone permission. Making it work also needs: the multipart upload to carry a filename (without one FastAPI reads the part as text and answers 422; the fix is in `scripts/generate_kotlin_client.py`), a WAV header on the recorded PCM (Sarvam is sent raw PCM labelled `audio/wav`), playback that reads the format of the server's TTS audio instead of assuming 16 kHz PCM, and `isSpeaking` set and cleared around playback so the Stop button appears and goes away. | Either finish it as one ticket, tested on a phone against the server, or hide the voice controls until then. A dead control is the failure the app's own accessibility notes warn about. My pick: finish it, since the server side already exists. |
| R2 | **Households, reminders and the observation queue have no screen and no scheduler.** Ticket 07's "reminders fire offline" is a pure function with tests; no alarm ever fires one. `HouseholdRepository.recordObservation` returns true when no sink is set, so an observation is reported saved and dropped. Household member names would be stored on the phone. | Decide whether these are in the study build. If not, remove them; if so, they need UI, an alarm, and a decision on storing names on the device. |
| R3 | **The Room schema for database version 4 is not committed.** `PalliDatabase` is version 4 and `exportSchema` is on, but `core/data/schemas/` stops at 3.json. There is also no migration test, though `androidTest` assets are configured for one. | Build once locally and commit `4.json`; add a `MigrationTestHelper` test for 1 to 4 before the first phone upgrades. |
| R4 | The interactions table is never pruned (`pruneBefore` has no caller), and the study-metric queries on `InteractionDao` are unused on the phone. | Agree a retention period with the study team, then prune on start. |
| R5 | When the server answers with an error while the phone is online, the Ask screen says "No connection, and this question is not one of the saved ones". | Distinguish "server error" from "offline". Needs a new string, so it waits for the clinical-language review. |
| R6 | The offline-miss notice that tells "no bundle downloaded yet" apart from "not in the bundle" exists only in a test; the Ask screen shows the single generic message. | Wire `OfflineMissNotice` into the Ask screen when the next UI change goes in. |
| R7 | `text_is_scrubbed` is true when nothing was found and false when an identifier was found and removed, which reads backwards for anyone analysing the column. | Rename or document the column before the study analysis plan uses it. It is a study data field, so not changed here. |
| R8 | The Ask screen keeps its own in-memory "Earlier" list beside the History tab, which reads the same interactions from the database. | Drop the in-memory list and point to History. |
| R9 | Moshi reads every response by reflection (`KotlinJsonAdapterFactory`, which pulls in kotlin-reflect), and R8 keep rules exist only to make that work. | Switch to Moshi's KSP code generation: smaller APK, faster parsing on low-end phones, and the keep rules can go. |
| R10 | Dead or test-only API surface: `GateDecision.Blocked.headline/detail` (English text the screen never shows; it uses `reason`), `CareRole.iconKey` (Material icon names; the icons are drawn by hand), `AskViewModel.clearError`, `HistoryRepository.recent/hasHistory`, `OfflineAnswerResolver.hasBundle`, `SyncQueue.enqueueObservation/pendingCount`, `CachedAnswerDao.versions/deleteOtherVersions`, `ActiveSettings.activeRole`, `PreferencesStore.languageChosen/clear`. | Remove in a follow-up together with their tests, once R1 and R2 decide which features stay. |
| R11 | Release build: `versionCode` is fixed at 1 and there is no signing config, so a release APK cannot be installed as an update. | Derive `versionCode` from the release record or CI run number and add signing from CI secrets before the first phone install. |
| R12 | AGENTS.md says no in-code comments; the Android sources are heavily commented (round 3, D15). | Still the project lead's call. |

## Round 2: fixing the review list (2026-11-19)

tp53 asked for the review list above to be fixed without a spec step, and chose, on R2, to build the household, observation and reminder screens and the alarm now. Decisions continue the numbering as A15 onward. The rows above are left as they were written; this section says what became of each.

Nothing here was run on a phone or an emulator. The cloud session cannot download the Android SDK, so the GitHub Android job (compile, unit and Robolectric tests, the release build and its size budget) is the only build. The microphone, the speaker, the permission dialogs, the alarm and the notification cannot be exercised by those tests. They need a session on a real phone against the study server before the build goes to participants; the list is under "Not verified" below.

### What became of R1 to R12

| # | Outcome |
|---|---|
| R1 | Fixed (A15 to A19). Needs a phone test. |
| R2 | Built (A20 to A28). Needs a phone test, and the clinical lead's review of the new words and the symptom list. |
| R3 | Fixed (A29). The migration test runs every older schema through the real migrations. `4.json` is committed from the CI build's output, and CI now fails when a schema file the build writes is not committed. |
| R4 | Not pruned (A30). The unused study-metric queries are removed. |
| R5 | Fixed with an existing string (A31). |
| R6 | Fixed (A31). |
| R7 | Documented, not renamed (A32). |
| R8 | Fixed (A33). |
| R9 | Not done (A34). |
| R10 | Removed (A35). |
| R11 | Fixed in the build; the key itself is the study team's to make and keep (A36). |
| R12 | Unchanged: still the project lead's call. New code follows the style of the files around it. |

### Decisions

| # | Decision | Why |
|---|---|---|
| A15 | One voice engine for the process, built at start and given to the Ask screen's ViewModel. | Nothing ever constructed one, so the hold-to-speak button did nothing. Two recorders cannot share a microphone, and a second player would talk over the first. |
| A16 | The Ask screen asks for the microphone with Android's own dialog, the first time the mic is pressed without it. The press that raised the dialog records nothing; a refusal shows the existing "Microphone access is needed" notice. | Only a screen can show the dialog. No new wording was needed. After two refusals Android stops showing the dialog and answers "no" itself, so the notice still appears, but the participant then has to allow it in system settings and the app does not yet say so. |
| A17 | The recording is uploaded as a WAV file with a header and a filename. The generated client takes a ready-made file part, and the form field's name comes from the contract as a generated constant. | Without a filename FastAPI reads the part as a text field and answers 422. Without a header the speech-to-text is told "WAV" and given bare samples. The fix is in `scripts/generate_kotlin_client.py`, so regeneration keeps it. |
| A18 | The answer is played with MediaPlayer from memory. | The server sends Sarvam's WAV file at 22.05 kHz; the old player assumed bare 16 kHz samples, so it would have played the header as noise and the speech slow. MediaPlayer reads the file's own format, and nothing is written to disk. |
| A19 | "Speaking" is true from the start of playback until it ends, fails or is stopped. A recording stops at one minute. Leaving the screen for good releases the microphone and the speaker. | The Stop button depends on that state, which was never set after an answer and never cleared after a replay. A button held down in a pocket must not fill a 2 GB phone. |
| A20 | A fourth tab, Household, holds the people cared for, symptom observations and medicine reminders. | tp53's choice on the decision card. One screen, with one form open at a time under the person it is about. |
| A21 | A person's name is stored on the phone only and is never sent. The server knows a person by a random reference made on the phone. | The household table's own design, and the answer to R2's question about names on the device: the app's data is private to the app, excluded from backup, and a name never leaves it. The first observation pushed under a new reference makes that record this participant's (`claim_or_check` on the server). |
| A22 | An observation is one of four symptoms (pain, breathlessness, nausea or vomiting, constipation) with a severity from none to very severe. Nothing is typed. | Free text bound for the patient record could carry a name or a dose. The names and the 0 to 4 scale are the server's own. Which four symptoms is a clinical choice: I took the first of the server's symptom list, and the clinical lead should confirm or change them. |
| A23 | "Saved" is shown only once the observation is in the sync queue; otherwise the screen says it was not saved. The repository's sink is now required. | With no sink the repository answered "saved" and kept nothing. |
| A24 | A queued observation reaches the server on the next sync push: at once when signal returns, else within 15 minutes while online and the app is open. | The existing sync scheduler, unchanged. The screen says "when there is a connection". |
| A25 | A reminder is daily, at a time, and says the medicine's name and the person's. It is stored as seven weekly rows, with no schema change. | The table keeps one row a weekday. Reminders on chosen weekdays can use the same rows later. |
| A26 | A medicine name that carries an amount is refused, with a notice to write only the name. | Project rule: never give a dose. The on-device dose detector that guards answers is run on the name. |
| A27 | Reminders stay on the phone and are not pushed to the server. | The server turns a synced reminder into a voice-call reminder, which the default release refuses and which needs a phone number. The phone's own alarm is what criterion 5 asks for. |
| A28 | The alarm is inexact (`setAndAllowWhileIdle`), one at a time for the next reminder, and set again after a restart, a clock or time zone change, an app update and each time the app opens. A reminder is shown once a day, as a notification with sound, in the language it was made in. | An exact alarm needs a permission Android gives only to clock and calendar apps or that the user must grant in system settings. The cost is that a reminder can be some minutes late; it counts as due for 30 minutes. On Android 13 and later the participant must allow notifications, which the app asks for when a reminder is saved. |
| A29 | Migrations are tested by building each older database from its committed schema file, adding an interaction, and opening it with the real migrations under Robolectric. | Room checks the result against the compiled schema and throws on any difference, and the test also checks the interaction survived. An instrumented test would need an emulator CI does not have. |
| A30 | Interactions are not pruned. | Deleting study records from a phone cannot be undone, and no retention period has been agreed with the study team. `pruneBefore` stays for when one is. |
| A31 | A server error while online shows "Could not reach the service. Please try again." An offline miss shows which of its two reasons it is, with the notice that already existed. | The first told someone with signal to go and find signal. Both use existing strings, so no new clinical-language review is needed. Two strings no screen uses any more (`history_heading`, `problem_offline_not_saved`) are removed from both languages. |
| A32 | `text_is_scrubbed` keeps its name and gains a comment that says which way round it reads. | It is a study data field already in three schema versions; renaming it is a migration on every phone for a label. The analysis plan should quote the comment. |
| A33 | The Ask screen no longer keeps its own "Earlier" list. | It duplicated the Visits tab without the answer kinds and forgot everything on restart. The Ask screen now scrolls, since an answer can be longer than the screen. |
| A34 | Moshi stays on reflection. | Generated adapters change how every reply is read in the minified release build, and a mistake there shows only on a release build running on a phone, which nothing here can run. Worth doing with a phone to hand. |
| A35 | Removed with their tests: `GateDecision.Blocked.headline/detail`, `CareRole.iconKey`, `AskViewModel.clearError`, `HistoryRepository.hasHistory`, `OfflineAnswerResolver.hasBundle`, `SyncQueue.pendingCount`, `CachedAnswerDao.versions/deleteOtherVersions`, `ActiveSettings.activeRole`, `PreferencesStore.languageChosen/clear`, and the five study-metric queries on `InteractionDao`. `HistoryRepository.recent` stays: ten tests read through it. `SyncQueue.enqueueObservation` is now used. | Each was checked for callers in main and test sources. None changes the database schema. |
| A36 | `versionCode` comes from `-PpalliVersionCode` (CI passes its run number; a local build is 1). A release is signed when `PALLI_KEYSTORE_FILE`, `PALLI_KEYSTORE_PASSWORD`, `PALLI_KEY_ALIAS` and `PALLI_KEY_PASSWORD` are set, and unsigned otherwise, as before. | A phone installs an update only with a higher version code and the same signing key. The key must be made once, kept out of the repository and kept for the length of the study. |

### Not verified

To check on a real phone against the study server, before participants:

1. Hold to speak: the permission dialog, a question recorded and answered, the answer spoken at the right speed, Stop and replay.
2. A refused microphone permission, twice, then allowing it in system settings.
3. A reminder set a few minutes ahead: with the app closed, with the phone idle and unplugged, after a restart, and with notifications refused.
4. An observation saved offline, then seen on the server's patient record after signal returns.
5. The release build (minified) doing all of the above, not only the debug build.

### For review

| # | For whom | What |
|---|---|---|
| R13 | Clinical lead | The 33 new strings in English and machine-drafted Hindi (the household block at the end of each `strings.xml`), and the choice of four symptoms (A22). |
| R14 | Study team | Whether observations from the app belong in the protocol's data flows, and that they sit in the server's patient record under a random reference (A21). |
| R15 | Project lead | A spoken reminder. The reminder is a notification with the phone's sound; reading it aloud offline needs the phone's own text-to-speech, which varies by language and phone. |
| R16 | Project lead | People cannot yet be renamed or removed, and a reminder is daily only. |
| R17 | Project lead | When the server answers "nothing was heard" to a voice question, the app shows "Could not reach the service". A truer sentence needs a new string. |
| R18 | Project lead | A reminder due in the last half hour before midnight that is delivered after midnight is missed for that day (`ReminderDueResolver` compares within one day). |
