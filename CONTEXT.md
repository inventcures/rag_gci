# Palli Sahayak

A voice-first clinical decision support system that gives palliative care guidance to
frontline health workers, family caregivers, and patients in India, across Indic languages.

## Language

**ASHA Worker**:
An Accredited Social Health Activist — a community health worker who makes twice-weekly
home visits to palliative care patients. Not clinically licensed; operates under the
supervision of a palliative care physician.
_Avoid_: nurse, clinician, doctor, medical officer

**Family Caregiver**:
A relative living with or near a palliative patient who provides daily care between ASHA
visits, under ASHA supervision.
_Avoid_: attendant, carer, relative

**Patient**:
A person receiving palliative care who uses the app directly for self-management, usually
sharing a device with their Family Caregiver.
_Avoid_: client, user, case

**Care Role**:
The capability tier a person registers under — ASHA Worker, Family Caregiver, or
Patient. It attributes every interaction for analysis and stratification, and shapes
what the app emphasises. It is **not** an access boundary: household members share one
Household Record (ADR 0006).
_Avoid_: permission, access level, account type

**Registration**:
First-time enrollment of a person and their device, establishing their Care Role and home
site, authenticated by a self-chosen PIN.
_Avoid_: signup, onboarding, enrolment

## Language

**Supported Language**:
One of eleven languages the app both listens in and speaks: Hindi, Bengali, Kannada,
Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Gujarati, and English (India). Bound by
TTS capability, not ASR capability — a language is only supported if the app can answer
aloud in it.
_Avoid_: Indic language, regional language, vernacular

**Speech-Language Mismatch**:
The condition where the app transcribes a query in a language it cannot synthesize, so it
must answer in English or text. Treated as a defect for Patient use, not a degradation.
_Avoid_: fallback, degradation

## Retrieval

**Cross-Language Retrieval**:
Matching a query in any Supported Language directly against the knowledge base, without
translating the query first. Achieved via a multilingual embedding model over a single
shared index.
_Avoid_: multilingual search, language-aware search

**Query Translation**:
Converting a non-English query into English to improve retrieval. Retained only as a
fallback when cross-language retrieval returns weak evidence — never as the default path.
_Avoid_: query normalization, preprocessing

## Medication Safety

**Dose Boundary**:
The line dividing what Palli Sahayak will say about a medicine from what it will not.
Above the line: what a medicine is, its class, what it treats, general precautions,
non-drug management. Below it: any specific dose, strength, titration step, escalation
schedule, interaction judgement, or start/stop decision — regardless of whether the
medicine is available without a prescription.
_Avoid_: OTC line, drug-class boundary, prescription gate

**Redacted Answer**:
An answer in which dose-bearing sentences have been removed and the remainder kept. The
preferred outcome when the Dose Boundary is crossed, because a general question about a
medicine should not lose its useful half just because one sentence contained a dose.
_Avoid_: sanitized response, filtered answer

**Deferral**:
The empathetic response that replaces or accompanies a Redacted Answer. It names
concrete next steps — write down symptoms, bring the packets, raise it at the next
consultation — and never claims a person has been contacted, because no notification
path exists yet.
_Avoid_: refusal, rejection, redirect

**Emergency Override**:
The rule that a life-threatening Emergency takes precedence over the Dose Boundary. Only
a CRITICAL alert may invoke it; a HIGH alert such as "severe pain" may not, or a routine
dose question phrased as severe pain would bypass the restriction entirely.
_Avoid_: safety bypass, guard exemption

## Household

**Household Record**:
The single shared record of a home's care situation, readable by every Care Role in
that home. A role switch changes what the app surfaces first and how it frames
answers; it does not partition the record.
_Avoid_: shared profile, family record, care record

**Active Role**:
The Care Role currently driving the app, shown as a persistent icon so that whoever is
holding the phone can see at a glance whose view they are seeing and whose voice the
app answers in. Switching it is deliberate and always visible; it is a framing choice,
not a privacy control.
_Avoid_: current user, logged-in-as, persona

## Connectivity

**Offline State**:
The condition in which Palli Sahayak has no connectivity. It shows local history and
the cached answer bundle, and does not answer new questions. The microphone control
remains visible but disabled with a plain-language explanation, rather than hidden or
failing silently.
_Avoid_: degraded mode, fallback, disconnected state

**Cached Answer Bundle**:
The small set of anticipated question-and-answer pairs downloaded to the device for
offline use. Not a retrieval index: it answers only what it was built to answer.
_Avoid_: offline cache, cache bundle, downloaded corpus

**Local History**:
Interactions stored on the device and readable while Offline. A Redacted Answer in
history must render distinctly from an Answer, so a refusal saved weeks earlier is
never mistaken for clinical advice.
_Avoid_: chat history, conversation log

## Retrieval

**Unsupported Answer**:
A substantive response returned with no retrieved source, meaning the text was
generated without grounding. This is the operational proxy the system log reports
where the grant asks for a hallucination rate; it is not a hallucination measure,
and the reported figure requires clinician adjudication.
_Avoid_: hallucination, ungrounded response
