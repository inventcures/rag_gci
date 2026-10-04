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
