# Eleven languages, bounded by speech output not speech input

The app supports exactly eleven languages: Hindi, Bengali, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Gujarati, and English (India). The boundary is speech **output** capability, not speech input — Sarvam transcribes 22 languages but synthesises only 11, and a Patient asking in Assamese, Nepali, Konkani, or Santali would hear nothing back.

Because this coincides exactly with the eleven languages the study protocol (§7.2) records as configured for two-way Sarvam support, the two sources agree and no reconciliation is needed. What remains open is the protocol's own requirement that each site approve a language-by-interface list after testing comprehension, clinical content, and escalation — so eleven is the ceiling, not the launch set.

## Considered Options

- **All 22 for speech input, English text fallback for the rest.** Rejected: a one-way voice channel is a defect for Patients, not a graceful degradation.
- **Add a second TTS provider for the remaining 11.** Deferred, not rejected — it is the only route to full coverage and should be revisited if the study sites require it.

## Consequences

- §10.4 of the Android spec is stale and must not be implemented as written. Its `codeMap` includes `tu-IN` (Tulu, which is not a Sarvam language at all) and omits `mr-IN`, `od-IN`, `pa-IN`, and `gu-IN`. Its `voiceMap` requests `"meera"`, which `sarvam_integration/config.py` records as deprecated — Bulbul v3 uses language-agnostic speakers, defaulting to `priya` (F) / `karun` (M).
- The app must never silently substitute Hindi for a language the user does not understand (§7.2). A language switch requires the user's understanding and agreement.
- Server-side emergency detection in `safety_enhancements.py` is English-only. It will not fire on a Marathi or Tamil query, so the client-side detector proposed in §10.5 is load-bearing for Patient safety, not an optimization.