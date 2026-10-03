# Palli Sahayak Android App — Implementation Spec

**Status**: Ready for agent
**Date**: 3 October 2026
**Supersedes**: nothing. Amends `docs/android-app-specs/v0_27march2026_0923st_210s_detailed-android-app-specs.md` to 0.2.0, which remains the normative technical specification for modules, schemas and UI.
**Decisions**: `docs/adr/0001` through `0007`
**Glossary**: `CONTEXT.md`
**Study constraints**: EVAH grant; study protocol v3, particularly §7.7 and §7.8

This document specifies *what to build*. Where it disagrees with the amended
0.2.0 specification or with an ADR, the ADR governs.

---

## Problem Statement

An ASHA Worker making a home visit in Barak Valley needs palliative care guidance
for a patient in front of them. Today they have three bad options: ask a
supervising physician who may be hours away, guess from memory, or abandon the
visit. The guidance exists, in English, in three PDFs that live on a server nobody
can reach from a village.

The problems compound.

**The knowledge is not available at the point of need.** Palli Sahayak answers
questions, but only as a backend with no client. There is no Android application.
The backend's mobile API exists and is instrumented, but a server with no client is
not a tool a health worker can hold.

**When the query is not in English, the answer is ungrounded.** The live index was
built with an English-only embedding model. Measured against the real corpus
across all eleven supported languages, it scores 0.00 on every Indic language while
scoring 1.00 on English. A Marathi question retrieves none of the correct passages
and returns confident, unfounded palliative advice. This is the single most
dangerous defect in the system, and it fails silently.

**The language is not the worker's.** A worker in Silchar speaks Bengali or Assamese.
The tool can transcribe 22 languages but speak back in 11. In the other 11 it is a
one-way channel that goes quiet.

**The network is not there.** CCF and CCHRC serve areas with intermittent coverage.
A tool that requires connectivity at the moment of need is unavailable at exactly
the moment it is needed.

**The household is not one person.** A patient recovering at home and the relative
caring for them share one phone, because a second device is unaffordable. The app
must serve both without pretending to offer privacy it cannot deliver.

**And nothing that happens is measurable.** The study protocol requires an
adoption figure, a safety figure and a per-language equity figure, all derived from
system logs, before any participant can use the system.

## Solution

A voice-first Android application that an ASHA Worker can use one-handed during a
home visit, in their own language, with or without a network, and which refuses —
clearly and kindly — to do the one thing it must never do.

The app asks a question by voice and answers by voice. The answer is grounded in a
pinned, versioned knowledge base retrieved in the user's own language. When the
network is absent the app still shows past interactions and a curated set of common
answers, and says plainly that it cannot answer anything new. When the question is
about a specific dose of a medicine, the app removes that part, keeps the useful
remainder, and explains what to take to the next consultation. When the situation
is genuinely an emergency, nothing is withheld.

Every interaction is recorded so the study can report what happened, which release
produced it, and in which language.

## User Stories

### Asking and answering

> User stories are numbered 1–86 as a single continuous list, grouped here by
> theme for readability. The numbering does not restart at each heading, so a story
> can be referenced unambiguously from a ticket or a review. A linter that expects a
> per-section restart will flag MD029 here; that is expected and should not be
> "fixed".

1. As an ASHA Worker, I want to ask a question by voice without typing, so that my hands stay free while I examine the patient.
2. As an ASHA Worker, I want the app to answer aloud, so that I do not have to read while the patient watches.
3. As an ASHA Worker, I want to see the answer as text as well as hear it, so that I can re-read it later when writing notes.
4. As an ASHA Worker, I want to ask a follow-up question and keep the conversation going, so that I do not have to restate the patient's situation.
5. As a Family Caregiver, I want to ask a simple question in my own language, so that I can understand what is happening without waiting for the ASHA Worker.
6. As a Patient, I want to ask about my own symptoms by voice, so that I can get reassurance without travelling to a clinic.
7. As an ASHA Worker, I want to interrupt the app while it is speaking, so that I can move on when I have what I needed.
8. As an ASHA Worker, I want to know immediately whether the app is currently able to hear me, so that I do not talk at a dead phone.
9. As an ASHA Worker, I want to stop the app speaking instantly, so that the patient is not made to listen to the rest.
10. As an ASHA Worker, I want to replay the answer, so that I can check something I missed.
11. As an ASHA Worker, I want the answer to be short enough to read between questions, so that it does not overrun the visit.
12. As an ASHA Worker, I want a clear visual indication that the app has finished speaking, so that I know it is my turn.

### Language

13. As an ASHA Worker, I want to choose my language once and have it remembered, so that I never have to select it again.
14. As an ASHA Worker, I want the app to answer in the same language I asked in, so that I can understand it without translation.
15. As a Patient, I want the app to never answer in a language I did not choose, so that I am not given something I cannot understand and told nothing about it.
16. As an ASHA Worker, I want to be told plainly when a language is not yet supported, so that I can switch to one that is.
17. As an ASHA Worker, I want the eleven supported languages available, so that a worker in any of our four states can use it.
18. As a user, I want every screen and every spoken prompt available in all supported languages, so that nothing is only in English.
19. As a user, I want the voice used to speak be natural and familiar, so that the app is not startling to a patient.
20. As a user, I want to hear the answer in my language even when I asked in another, so that I can follow along.

### Medication safety

21. As an ASHA Worker, I want to ask what a medicine is for and get a real answer, so that I can explain it to the family.
22. As an ASHA Worker, I want to ask about general precautions for a medicine, so that I know what to watch for.
23. As an ASHA Worker, I want to be told about non-drug ways to manage a symptom, so that I have something to offer when a dose is out of reach.
24. As an ASHA Worker, I want the app to decline a specific dose in a way that does not feel like a scolding, so that I can accept the answer gracefully in front of a family.
25. As an ASHA Worker, I want the useful part of an answer kept when only the dose is removed, so that I do not lose the whole answer over one sentence.
26. As an ASHA Worker, I want the app to tell me what to take to my next consultation, so that the question is not wasted.
27. As an ASHA Worker, I want the app to tell me to call an ambulance when the patient is in real distress, rather than deferring to the next appointment, so that nobody waits.
28. As an ASHA Worker, I want an answer withheld by the dose rule to look different from a real answer, so that I do not later mistake a deferral for clinical advice.
29. As a clinician reviewing the log, I want deferrals visibly distinct from answers, so that a refusal is never counted as guidance delivered.
30. As a user, I want the app never to claim a person has been contacted, so that I am not told a nurse is coming when nothing has been sent.

### Emergencies

31. As an ASHA Worker, I want the app to detect an emergency on my phone even without a connection, so that detection does not depend on the network.
32. As an ASHA Worker, I want a one-tap way to call an ambulance, so that I do not lose time finding a number.
33. As an ASHA Worker, I want an urgent case never to have its guidance removed by the dose rule, so that the restriction does not swallow a genuine emergency.
34. As an ASHA Worker, I want "severe pain" treated as serious but not as an emergency that overrides everything, so that the app stays useful for ordinary pain.
35. As an ASHA Worker, I want emergency detection to work in the languages I work in, so that detection is not limited to English and Hindi.
36. As a clinician, I want the severity of a detected emergency preserved end to end, so that a high-severity case is never escalated to critical by accident.

### Offline

37. As an ASHA Worker, I want to see past interactions when I have no connection, so that I can re-read what was said at the last visit.
38. As an ASHA Worker, I want the app to tell me plainly that it is offline, so that I do not think it has failed.
39. As an ASHA Worker, I want the microphone to remain visible but disabled when offline, so that I understand why it will not work instead of concluding the app is broken.
40. As an ASHA Worker, I want the offline state conveyed by picture as well as words, so that I understand it if I cannot read the words.
41. As an ASHA Worker, I want common questions answered offline, so that I am not helpless when the network is down.
42. As an ASHA Worker, I want to be told when my question is not one of the offline ones, so that I know to find signal and retry.
43. As an ASHA Worker, I want medication reminders already on my phone to keep working offline, so that reminders do not depend on a connection.
44. As an ASHA Worker, I want offline answers marked so they are not mistaken for live ones, so that I know how much to trust them.
45. As an ASHA Worker, I want the app to recover cleanly when signal returns, so that I do not have to restart it.

### Household and roles

46. As a Family Caregiver, I want to switch to my own view when the phone is handed to me, so that I see what is relevant to me.
47. As a Family Caregiver, I want to see the household's care situation without switching, so that I can answer a question immediately.
48. As a Patient, I want to see clearly at a glance whose view I am looking at, so that I am not confused about whose questions I am seeing.
49. As a user, I want to see who is active as a picture, so that I can tell even if I cannot read the label.
50. As a user, I want switching roles to take one tap, so that handing the phone around is not a chore.
51. As a user, I want the app never to imply that switching roles hides anything, so that I am not misled about who can see what.
52. As a user, I want my role remembered for next time, so that I do not have to set it every visit.
53. As an ASHA Worker, I want to record an observation against the right household member, so that notes do not get mixed up.

### Accessibility and low literacy

54. As a user with limited literacy, I want to do the main things with large pictures rather than text, so that I can use the app without reading.
55. As a user with limited literacy, I want each role to have a picture that is not like the other two, so that I can tell them apart.
56. As a user with limited literacy, I want controls large enough to hit reliably, so that I do not fail by accident.
57. As a user, I want important states shown by colour as well as words, so that a colour-blind user is not excluded.
58. As a user, I want no gesture-only actions, so that I can use the app without discovering hidden gestures.
59. As a user who speaks no English, I want no screen to require English to operate, so that I am not locked out of a function.

### Study and evaluation

60. As the study team, I want every interaction recorded with the release that produced it, so that we can say exactly what participants used.
61. As the study team, I want to count adoption using the protocol's own definition, so that our number matches what we promised.
62. As the study team, I want training demonstrations and test messages excluded from adoption, so that the figure is not inflated.
63. As the study team, I want the denominator reported alongside every percentage, so that the number is interpretable.
64. As the study team, I want per-language metrics, so that we can detect a language where the tool works worse.
65. As the study team, I want low-volume languages flagged rather than ranked, so that we do not over-claim from small samples.
66. As the study team, I want latency measured per stage, so that we can tell which part is slow.
67. As the study team, I want answers with no retrieved source surfaced for review, so that ungrounded answers are caught.
68. As the study team, I want responses sampled for expert review at the promised rates, so that supervision happens as designed.
69. As a study auditor, I want to browse the log without being able to change it, so that the record is trustworthy.
70. As a study auditor, I want participant identifiers pseudonymised, so that the log cannot identify a person.
71. As a study auditor, I want contact details and identity numbers stripped from recorded text, so that the log carries no direct identifiers.
72. As the study team, I want to know when the running system no longer matches the approved release, so that drift is visible rather than silent.
73. As the study team, I want to know how often each voice path answered, so that we can report residency accurately.
74. As the study team, I want to know who changed the voice path and why, so that an override cannot become permanent unnoticed.
75. As the study team, I want the app to refuse to start against an unapproved release, so that no participant uses the wrong build.
76. As a participant, I want my care details not to leak to another household, so that one family's data cannot reach another's.

### Reliability and operations

77. As an ASHA Worker, I want the app to survive losing signal mid-answer, so that I do not get a half-sentence.
78. As an ASHA Worker, I want a clear message when something has genuinely failed, so that I know to try again rather than wait.
79. As the backend administrator, I want to force the India-resident voice path for the whole deployment, so that I can respond to a provider incident.
80. As a site worker, I want no ability to change the voice path, so that an operational switch cannot quietly become permanent.
81. As an ASHA Worker, I want the app to work on an old, cheap phone, so that I can use the phone I actually own.
82. As an ASHA Worker, I want the app to be small enough to download on a slow connection, so that I can install it at all.
83. As an ASHA Worker, I want my PIN to work even after a restart, so that I am not locked out.
84. As an ASHA Worker, I want several wrong PIN attempts to lock the phone temporarily rather than permanently, so that a mistake does not end my access forever.
85. As the backend administrator, I want the app's API client checked against the server schema, so that a server change cannot silently empty answers.
86. As an ASHA Worker, I want my questions and answers to survive the app being closed, so that I can find them at the next visit.

## Implementation Decisions

### Modules

The app is organised into a bounded Gradle project under `android-app/`, with an
explicit `settings.gradle.kts` that includes only app modules and never scans the
Python tree. Gradle files do not exist at repository root.

Layering, outside-in:

- **presentation** — Compose screens and view models. No business rules.
- **domain** — pure Kotlin. Use cases and domain models. No Android framework types.
- **data** — repositories, Room, the API client, the offline bundle reader.
- **voice** — the VoiceEngine interface and its three implementations.
- **safety** — the client-side safety invariants, kept separate because they are the
  part that must not be refactored casually.

### Safety invariants are normative

Seven invariants constrain the whole app. A change violating one does not merge.
They are specified in §7.0 of the amended technical specification and summarised here
because they shape module boundaries rather than just behaviour.

- **SI-1 Dose Boundary** — never state a specific dose, strength, titration step,
  escalation schedule, interaction judgement, or start/stop decision.
- **SI-2 Redacted Answer, not silent omission** — cross the boundary by removing
  dose-bearing sentences and keeping the remainder; fall back to a full Deferral only
  when too little survives. A Redacted Answer is never rendered, logged or counted as
  an Answer.
- **SI-3 Deferral never promises a person** — no implied clinician contact.
- **SI-4 Emergency Override is CRITICAL only** — severity must survive to the client.
  `EmergencyKeywordDetector` returns a severity, never a flat CRITICAL. This is the
  invariant most easily broken by a "simplification".
- **SI-5 Release provenance on every interaction.**
- **SI-6 No Hindi substitution** — language selection returns null rather than
  coercing.
- **SI-7 Care Role is not an access boundary** — one Household Record; cross-household
  ownership checks are required.

The Dose Boundary is enforced **server-side** as a post-generation filter over the
response text, not as a prompt instruction, because protocol §7.7 states that a prompt
instruction or a successful connection test alone does not satisfy the checks. The
client additionally redacts locally when the response was produced by a voice path
that does not traverse the server filter.

### Voice

Three implementations of one interface:

- **LiveVoiceEngine** — primary. WebSocket to our backend, which proxies Gemini 3.8
  Live and grounds answers through a mandatory tool call back to the query endpoint.
- **ServerVoiceEngine** — automatic fallback. Turn-based Sarvam path. India-resident.
  Also the primary path when offline-capable text is all that is needed.
- **OnDeviceVoiceEngine** — offline speech recognition against the cached bundle.

Fallback is **automatic and health-based**: first-token timeout and per-turn stall
detection, not merely a failed connection, because a session that connects and then
goes quiet leaves the user mid-sentence.

The backend administrator can force the Sarvam path deployment-wide, for operational
reasons such as quota exhaustion, cost or a suspected provider incident. The control
is server-side only; the client has no ability to change it. Site coordinators and
field staff do not hold this right. Every change writes an attributed audit record
with actor, timestamp, previous and new value, and a required reason.

The provider that answered is recorded on every interaction.

**The UI indicates which voice mode is active.** The two modes are perceptibly
different — one can be interrupted, the other cannot — and a user who interrupts a
Sarvam response will otherwise conclude the app has frozen.

### Model and region pinning

- **Language model**: Gemini 3.5 Flash, pinned to `asia-south1`. Chosen because
  Gemini 3.8, 3.7 and 3.6 Flash have no Asia-Pacific availability at all; 3.5 Flash
  is the newest model available in India.
- **Embeddings**: bge-m3, self-hosted, 1024 dimensions. Chosen by measurement — mean
  hit@5 of 0.96 against 0.72 for LaBSE and 0.09 for the previous English-only index,
  which scored 0.00 on all ten Indic languages.
- **Retrieval**: cross-language. Query translation is a fallback for weak retrieval,
  never the default path, and the previous translation path is known-defective because
  it silently proceeded when the translation call failed.

### Residency, stated precisely

The language model, the embeddings and the knowledge base are India-resident or local.
The Live voice path is not, because no Indian region offers the Live API. This is
narrower than the grant's blanket claim and must be described exactly this way. A
selected region is not on its own evidence of in-region processing; the documentation
warns that endpoints do not guarantee data residency. That confirmation must be
obtained in writing before the IRB sees the claim.

### Offline

A cached answer bundle plus local history. On-device query embedding is ruled out:
bge-m3 at 568M parameters is roughly 600 MB int8, which is hours of download on a 2G
link and 600 MB to 1 GB resident on the 2 GB reference device. Frozen corpus vectors
alone are 263 KB but cannot be searched without the encoder, so they are not a
solution.

The offline affordance is a **disabled control that explains itself**, not a banner
and not a hidden button. Users may be semi-literate, so state is carried by icon and
colour together with plain text.

The 25 MB APK figure is a self-imposed design target derived from the 2G field
constraint, not a platform limit. It must not be used to reject a feature on its own;
raising the ceiling does not make a 2G download viable.

### Language

Eleven Supported Languages, bounded by TTS capability. `tu-IN` is not a Sarvam
language and is removed. `mr-IN`, `od-IN`, `pa-IN` and `gu-IN` are added. Bulbul v3
uses language-agnostic speakers, so the per-language voice map is gone. `toBcp47`
returns null for an unsupported language rather than coercing.

### Household and roles

One Household Record. Care Role frames the view and is recorded for stratification;
it partitions nothing. The Active Role is shown as a persistent large icon. Switching
is one tap with no PIN and is never presented with a lock or shield, because it is not
a privacy control and must not appear to be one.

Role icons are differentiated by **context object and posture**, not by person
figures, since three similar human silhouettes are indistinguishable at small sizes to
a semi-literate user: `medical_services` for the ASHA Worker, `diversity_2` for the
Family Caregiver, `bed` for the Patient. Sizes exceed Material defaults — 48–56 dp for
the Active Role indicator, 72 dp for role tiles, 48 dp minimum touch target.

### Data and sync

Room is the system of record on device. Sync is a WorkManager job. Conflict resolution
follows the CommCare pattern: server-authoritative for clinical content, with local
observations queued as append-only until acknowledged.

The offline bundle schema is a contract shared with the backend. Both sides live in
this repository precisely so a schema change is one commit.

### Study instrumentation

The client does not compute study metrics. It records what happened and the server
derives everything, so that the analysis has one source of truth and cannot disagree
with itself across pages.

The client is responsible for: care role, household, locale, connectivity state,
whether the answer came from the cached bundle, voice mode used, and the release
identifier.

## Testing Decisions

### What makes a good test here

Test external behaviour through the highest seam available. Do not mock our own code
in a UI test — mock only the outside world: network, clock, randomness, hardware
boundaries. In a safety-critical app the failure mode we most fear is a test that
passes because it asserted against our own internal shape rather than what a user
actually sees.

Safety invariants are tested as invariants, not as incidental coverage. A regression
in SI-1 or SI-4 is a protocol breach, not a bug, and should fail the build with the
invariant named.

### Seams

**Seam 1 — App UI, against a faked server.** Robolectric driving the real app. This
is the primary seam: one test per screen or flow, exercising the real Room database,
the real repositories and the real view models. Only network, clock, randomness and
hardware boundaries are faked.

**Seam 2 — Backend HTTP API.** The existing HTTP seam. Prior art exists: the study
logging and dosage guard suites drive the real app through `TestClient`.

**Seam 3 — Contract, app against server schema.** Regenerate the API client from the
server's OpenAPI schema and fail when it moves. Both sides live in one repository, so
drift is more likely than usual, and the failure it catches — a changed field name
producing silently empty answers — is invisible in review and fatal in the field.

**Seam 4 — Safety invariants.** Dedicated, named per invariant. The server-side prior
art is the dosage guard suite: twenty-five tests including a regression sweep over
every real output in the evaluation corpus, which found the defect that this whole
spec responds to. The client needs the equivalent for severity preservation and for
the no-Hindi-substitution rule.

**Seam 5 — Retrieval quality harness.** Not a unit test. A measurement over the real
corpus in all eleven languages, re-run whenever the embedding model, chunking or
retrieval settings change. It is what makes an embedding change an evidenced decision
rather than an opinion, and it is what protocol §7.7 requires when naming the
retriever in the release record.

### Modules tested

- presentation — through Seam 1
- domain — through Seam 1 where behaviour is user-visible, and directly for pure logic
- data — through Seam 1 and Seam 2
- voice — through Seam 1 with a faked transport, plus directly for fallback
  arbitration
- safety — through Seam 4 only

### Known gaps in the test surface

The corpus-based retrieval harness currently uses translations authored by engineers
rather than reviewed by native speakers. Until that is done it is adequate for
choosing a model and not adequate for publication. The harness must support re-running
against a reviewed translation set without code changes.

## Out of Scope

**The backend RAG pipeline.** This spec is the client. Retrieval, grounding, the Dose
Boundary filter and the evidence badge are server-side and are specified in the
companion backend document. The one exception is client-side redaction described under
SI-2.

**WhatsApp, telephony and Bolna channels.** The grant describes telephone and
WhatsApp pathways. They are not part of the Android app and are not specified here.

**The study's survey instruments.** SUS, AIM/IAM/FIM and the vignette instruments are
administered by the study team. The app supplies interaction data only.

**On-device embeddings.** Ruled out in ADR 0005, with numbers. Revisit only if the
field cohort proves to be on 4G rather than intermittent 2G.

**A lexical offline index.** ADR 0005 defers it to a later version as the honest
answer to "why can I not ask anything else offline". Build it only if the bundle
proves insufficient, and measure it against the same harness.

**Play Store distribution.** Internal deployment for the study first.

**Real-time translation between languages.** The app answers in the chosen language.
It does not translate an answer from one language into another.

**Within-household privacy.** Explicitly ruled out by ADR 0006, which is a product
decision, not an omission.

## Further Notes

**Two things block participant use and are not code.** Protocol §7.7 requires the
restriction on dosage advice to be enforced *and tested* before any participant can
use the build. The guard is enforced and tested for new responses, but the evaluation
corpus already in the repository contains 51 outputs that breach it. They are evidence
of the defect this spec addresses, and they need regenerating before the build faces a
participant. Separately, the release record is `draft` with both required approvals
unsigned.

**The corpus is 263 chunks from three PDFs, and 35 of those are a table of
contents.** The 0.96 hit@5 was measured against that ceiling. A model cannot retrieve
guidance that is not there, so corpus work may matter more than retrieval tuning. This
should be confirmed as the intended knowledge base before optimisation effort is spent
on the index.

**Residency requires a written confirmation.** The Live path leaves India. Before the
IRB sees the claim, obtain Google's written confirmation of in-region processing for
`asia-south1`, because a selected region is not by itself evidence and the
documentation explicitly warns on this point.

**Patient identifiers on shared devices.** A caregiver and a patient share one phone.
The Household Record decision means the device is protected by whatever the household
protects it with. Whether the device additionally needs a lock before launch is a
consent design question for the study team, and should be resolved before fielding
rather than during it.
