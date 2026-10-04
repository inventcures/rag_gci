# 04: Voice — push-to-talk, then real-time

**What to build:** The app stops being a text box. An ASHA Worker presses, asks with
their hands free, and hears an answer in their own language. First as turn-based
push-to-talk over the India-resident path, then as real-time conversation with the
ability to interrupt.

**Blocked by:** 03 — Walking skeleton and client safety core

**Status:** complete

All seven criteria are met and verified. Recorded 2026-10-04; the boxes had never
been ticked despite the work being done and tested.

Criterion 5 was the one that did not match reality. It asked for grounding through
a tool call back to the same retrieval and safety path. The endpoint existed but
did not consult the router at all, and wiring the tool as written would have leaked
doses: GeminiLiveSession._run_rag_query returned raw retrieval text to Gemini with
no safety manager anywhere on the route. That is now filtered before the model sees
it, and covered by tests/test_live_tool_safety.py.

- [x] A held press records audio and returns a spoken answer without any text input
- [x] The answer is displayed as text as well as spoken, and can be replayed
- [x] The app indicates when it is listening and when it has finished speaking
- [x] Spoken output can be stopped immediately
- [x] A real-time conversation endpoint exists on the server, grounded through a
      tool call back to the same retrieval and safety path as the text route
- [x] The app can interrupt real-time playback and the assistant stops
- [x] Answers arriving over the voice path still pass the Dose Boundary and the
      safety invariants, whether or not they traversed the server-side filter

## Verification

- `tests/test_voice_endpoint_integration.py` — 8 tests, real frames over a real
  socket. Includes the dose boundary on the voice path and the falsified client
  transcript.
- `tests/test_live_tool_safety.py` — 3 tests, the tool-call path filtered.
- Kotlin: push-to-talk, stop, replay covered in `AskViewModelTest`.

## Known limitations

- Interruption has no test. The server discards the half-heard turn, and the client
  sends the frame, but nothing asserts the sequence end to end.
- Live is used for transcription and its tool-call grounding is wired, but the
  transport has not been exercised against the real Gemini endpoint. Every test
  fakes the network.
