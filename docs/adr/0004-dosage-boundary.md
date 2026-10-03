# The dosage boundary is drawn at the dose, not at OTC versus prescription

Palli Sahayak answers general questions about medicines — what they are, what class
they belong to, what they treat, general precautions, non-drug management — but never
gives a specific dose, strength, titration step, escalation schedule, interaction
judgement, or start/stop decision. When a response contains one, it is replaced with
an empathetic deferral that names concrete next steps and, where relevant, the 108
emergency number. An emergency always takes precedence over the deferral.

The grant proposal states the system "never provides specific medication dosages" and
the study protocol (§7.7) requires that this restriction be *enforced and tested*
before any participant uses the build. The obvious alternative — refusing anything
prescription-only, allowing over-the-counter — was rejected for four reasons.

OTC is not a safety property for this cohort. Participants have advanced kidney and
liver disease, frailty and dementia. A freely available medicine can still be unsafe
for these patients, and the distinction that matters clinically is whether a clinician
can safely authorise it, not whether it can be bought without a prescription.

There is no Rx/OTC classification data in this repository, so the boundary would be
unenforceable and untestable — precisely what §7.7 forbids. And a dose is a dose
regardless of how it was purchased: "paracetamol 500mg once daily" breaches the
promise exactly as much as a morphine instruction does, so the OTC carve-out would
have let the failure mode straight through.

Finally, the corpus is 263 chunks about opioid titration, the WHO pain ladder,
haloperidol and NG feeding. The questions ASHA workers actually ask *are* prescription
questions, so the OTC carve-out would have refused most of what the tool is for while
permitting the part nobody was concerned about.

## Implementation

`dosage_guard.py` enforces the restriction as a *post-generation filter* on the
response text, not as a prompt instruction. §7.7 states that "a prompt instruction or
a successful connection test alone does not satisfy these checks", so the control is
placed where it can be tested. `tests/test_dosage_guard.py` is that evidence, and
includes a regression test over the 80 real outputs already in
`data/evaluation/rag_outputs/` — 51 of which contain an explicit dose, and all 51 are
now blocked.

Three failure modes were found and closed while building this, each of which would
have silently disabled the control:

- A response-text fallback matched the bare word "emergency", so any answer merely
  mentioning emergencies bypassed the restriction. Tightened to require an explicit
  instruction to summon help.
- Only a `CRITICAL` alert may override the guard. The emergency detector also fires at
  `HIGH`, and "severe pain" is one of its keywords — so any dose question phrased as
  severe pain would otherwise have bypassed it entirely.
- Indic numerals do not match `\d`, so doses written as `१० मि.ग्रा.` or `๑๐ มิลลิกรัม`
  evaded detection. Numerals are normalised to ASCII before matching.

The deferral wording never implies a human has been contacted. Protocol §7.6 forbids
implying a transfer succeeded before receipt is confirmed, and no notification path
exists yet, so the copy is about the advice rather than about a dispatch the system
cannot perform.

## Consequences

- The guard redacts at the sentence level rather than discarding whole answers, so
  that a tier-1 question is not answered with a tier-2 refusal when only part of the
  answer was dosing. Across the 80 real outputs in `data/evaluation/rag_outputs/`, 51
  contained a dose; all 51 were redacted and **none fell back to a full refusal**,
  because every one retained enough surviving content to be useful.
- Redaction is verified after the fact: the detector is re-run over the redacted text
  and anything still matching escalates to a full refusal. Redaction must never be the
  thing that lets a dose through, so it fails closed rather than open. Measured leak
  count across the corpus is zero.
- List numbering is deliberately left with gaps when an item is removed. Renumbering
  would break any cross-reference the model made to "step 3", and the gap is a useful
  signal to the reader that something was withheld rather than silently dropped.
- A remainder under 80 characters falls back to a full refusal, because an orphaned
  clause reads worse than a clean deferral. The threshold was set by measurement: two
  clean sentences of real clinical content run to roughly 110 characters.
- A refusal is recorded as `validation_status: "dosage_restricted"` with evidence level
  E and zero confidence, so that refusals are never counted as grounded answers in the
  safety metrics the grant promises.
- Refusals are a safety-protocol event and must be logged per interaction, with the
  reason and the language, or the study cannot report how often the restriction fired.
- Refusal copy is translated for Hindi, Bengali and Tamil. The remaining supported
  languages fall back to English rather than refusing to answer, which should be closed
  before the study build.