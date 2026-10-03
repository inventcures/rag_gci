# The dosage boundary is drawn at the dose, not at OTC versus prescription

Palli Sahayak answers general questions about medicines, what they are, what class they belong to, what they treat, general precautions, non-drug management, but never gives a specific dose, strength, titration step, escalation schedule, interaction judgement, or start/stop decision. When a response contains one, the dose-bearing sentences are removed and the remainder is kept, and an empathetic deferral explains what to take to the next consultation. A life-threatening emergency is never withheld.

The grant proposal states the system "never provides specific medication dosages" and the study protocol (§7.7) requires that this restriction be *enforced and tested* before any participant uses the build. The obvious alternative, refusing anything prescription-only while allowing over-the-counter, was rejected for four reasons.

Over-the-counter is not a safety property for this cohort. Participants have advanced kidney and liver disease, frailty and dementia. A freely available medicine can still be unsafe for these patients, and the distinction that matters clinically is whether a clinician can safely authorise it, not whether it can be bought without a prescription.

There is no Rx/OTC classification data in this repository, so that boundary would be unenforceable and untestable, which is exactly what §7.7 forbids. A dose is a dose regardless of how it was purchased, so "paracetamol 500mg once daily" breaches the promise exactly as much as a morphine instruction does. The over-the-counter carve-out would have let the failure mode straight through.

Finally, the corpus is 263 chunks about opioid titration, the WHO pain ladder, haloperidol and NG feeding. The questions ASHA workers actually ask *are* prescription questions, so that carve-out would have refused most of what the tool is for while permitting the part nobody was concerned about.

## Implementation

`dosage_guard.py` enforces the restriction as a *post-generation filter* over the response text, not as a prompt instruction, because §7.7 states that "a prompt instruction or a successful connection test alone does not satisfy these checks". `tests/test_dosage_guard.py` is that evidence, and includes a regression test over the 80 real outputs already in `data/evaluation/rag_outputs/`. Fifty-one of them contain an explicit dose, and all 51 are now blocked.

Crossing the boundary redacts the dose-bearing sentences and keeps the remainder, so a general question about a medicine does not lose its useful half. Redaction is verified by re-running the detector over its own output and escalating to a full refusal on any residue, so it fails closed rather than open.

Three failure modes were found and closed while building this, each of which would have silently disabled the control. A response-text fallback matched the bare word "emergency", so any answer merely mentioning emergencies bypassed the restriction. A HIGH-severity alert ("severe pain") could override the guard when only CRITICAL should. And Indic numerals do not match `\d`, so doses written as `१० मि.ग्रा.` or `๑๐ มิลลิกรัม` evaded detection.

The deferral wording never implies a person has been contacted, per protocol §7.6, since no notification path exists yet.

## Consequences

- Refusing a whole response is blunt. If the model volunteers a dose while answering a tier-1 question such as "what is morphine used for?", the entire answer is discarded. Sentence-level redaction, keeping the non-dose remainder, would recover that content. It is not implemented here because the redaction would itself need proving, and the conservative behaviour is what should face the IRB first.
- A refusal is recorded as `validation_status: "dosage_restricted"` with evidence level E and zero confidence, so refusals are never counted as grounded answers in the safety metrics the grant promises.
- Refusals are a safety-protocol event and must be logged per interaction, with the reason and the language, or the study cannot report how often the restriction fired.
- Refusal copy is translated for Hindi, Bengali and Tamil. The remaining supported languages fall back to English rather than refusing to answer, which should be closed before the study build.
