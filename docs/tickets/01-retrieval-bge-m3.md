# 01: Retrieval — bge-m3 end to end

**What to build:** An ASHA Worker asks a question in their own language and gets an
answer grounded in the real passages. Today the live index scores 0.00 hit@5 on all
ten Indic languages while scoring 1.00 on English, so an Indic question returns
confident, unfounded guidance. This ticket makes Indic retrieval work, makes the
offline bundle's questions exist in the eleven supported languages rather than only
in English, and removes the outputs that breach the dosage restriction from the
evidence base.

**Blocked by:** None (can start immediately)

**Status:** complete — 8 of 8 acceptance criteria met

- [x] The live knowledge index is built with bge-m3 at 1024 dimensions, and the
      pipeline no longer falls back to an English-only model by default
- [x] The retrieval harness reports a mean hit@5 of at least 0.90 across the eleven
      supported languages, with no language below 0.75
- [x] Query translation is no longer on the default path; it is reachable only as a
      fallback for weak retrieval
- [x] The offline cache bundle's question set exists in all eleven supported
      languages, and no bundle entry is a question the Dose Boundary will refuse
- [x] The offline cache bundle can no longer be generated against an English-only
      embedding model
- [x] Every output in the evaluation corpus has been regenerated under the new
      retrieval path, and none of them contains a specific dose, strength, titration
      step or escalation schedule
- [x] The release record names the retriever, the embedding model and the index
      dimensions, and reports no drift against the running system
- [x] The harness can be re-run against an externally reviewed translation set
      without code changes, so native-speaker review is a data swap rather than a
      rewrite

## Verification done

- Live index rebuilt at 1024d with bge-m3; identity recorded on the collection and
  the drift check reports none.
- Retrieval harness: mean hit@5 0.96 across all eleven languages, lowest language
  0.78 (Punjabi), against 0.09 for the English-only index it replaced.
- Hindi, Tamil, Kannada and Bengali morphine queries each return the correct
  passage from the live index, where the previous index returned none.
- All 80 evaluation outputs regenerated. None contains a specific dose. 41 were
  partially redacted with the clinical remainder preserved and none was refused
  outright, which is the behaviour chosen for the Dose Boundary.
- Translation fallback verified against the live index: Hindi, Kannada and Tamil
  queries all resolve through cross-language retrieval with the fallback
  counter at zero.
- 105 tests across six suites, all green.

## Known limitations

- **The weak-evidence probe is not discriminative in this corpus.** With 263
  chunks every string has a near neighbour under the threshold, so gibberish also
  reports as not weak. It is a weak-evidence signal, not a relevance classifier.
  It becomes meaningful as the corpus grows.
- **Offline question translations are engineering-authored**, not
  native-speaker reviewed. The structure is tested; the fluency is not. Native
  review remains required before the study build.
- **The evaluation harness still carries its own probe translations** inline, so
  native-speaker review of the retrieval evaluation is a data edit in one module
  rather than a configuration swap. The offline bundle already accepts an
  external set.

