# Run the voice demo at an event (Railway)

This guide gets you a link you can open on a laptop or phone at an event. You tap the microphone once and talk to Palli Sahayak in English or an Indian language. It answers out loud, from the Palli Sahayak knowledge base, until you tap again.

> **Made-up people only.** Railway has no region in India, so this server runs in Singapore. Visitors' voices go to Google's Gemini Live service, and the transcripts are kept in the server's study log. Use made-up patients and made-up names. Do not let a visitor describe a real patient. The page shows this notice on every Railway deployment.

## How it fits together

The demo is the web voice page at `/voice/` on the `api` service from [deploy_railway.md](deploy_railway.md). It needs no service of its own.

| Part | What it does |
|---|---|
| `web_client/` | The page. It streams the microphone to the server and plays the answer. |
| `/ws/voice` | The server's voice socket. It connects the page to Gemini Live. |
| Gemini Live | Hears the caller, decides when they have finished a sentence, and speaks the answer. The model is the study release's (`data/study/release.yaml`, `models.voice_primary.model`). |
| Knowledge base lookup | Before a health answer, Live calls the server's `search_medical_knowledge` tool. The server searches the shipped index and drafts a dose-free answer. Live speaks from it. |
| Dose check | The server holds each spoken turn until it is complete and checks the transcript. A turn that names a dose is not played; the safe wording is shown, and spoken by Sarvam if `SARVAM_API_KEY` is set. |

### Can it run on Railway?

Yes, on the **Hobby plan or higher**. The free and trial plans cannot run it.

| Need | This server | Free / Trial | Hobby | Pro |
|---|---|---|---|---|
| Memory | 3 to 4 GB in use, set 8 GB | 0.5 / 1 GB | up to 48 GB | up to 1 TB |
| Volume | 5 GB (2.3 GB model, index, logs) | 0.5 GB | 5 GB | up to 1 TB |
| Image | a few GB (CPU torch, no CUDA) | 4 GB | 100 GB | no limit |
| WebSockets | held open for the whole conversation | yes | yes | yes |

Figures from [Railway plans](https://docs.railway.com/reference/pricing/plans) and [public networking limits](https://docs.railway.com/networking/public-networking/specs-and-limits) (WebSockets have no duration or idle limit). Hobby costs $5 a month plus what the service uses. One server process runs the demo; do not add replicas.

### Languages

The language buttons are the languages Live speaks in the project's language table (`supported_languages.py`): English, Hindi, Bengali, Kannada, Malayalam, Marathi, Tamil, Telugu and Gujarati. Odia and Punjabi are not offered for Live.

The Live model detects the language you speak, so you can start in any of these without pressing a button, and switch mid-conversation. The button sets the greeting and the language of the safe wording when a turn is held back.

### How grounded are the answers?

* The model is told to call the knowledge base tool before every health question and to answer only from what it returns. The tool's answer has passed the dose check before the model sees it.
* The shipped index is small: 263 passages from 3 source documents. Questions outside them get "I don't have information on that, ask your doctor" style answers. That is the intended behaviour.
* **Known gap.** Live decides by itself whether to call the tool. Nothing forces the call, and nothing detects a health answer given without it. The dose check still applies to every turn.
* **What the page shows.** An answer that came from a lookup ends with "From the Palli Sahayak knowledge base:" and the document names. An answer with no such line was not looked up. Greetings and small talk never are. The server log also says "answered without a knowledge-base lookup" for each such turn, so you can check after the event.

## Steps only you can do

Do the `api` service setup in [deploy_railway.md](deploy_railway.md) first (steps 1 to 4). Then:

### 1. Check the Gemini key works for Live

The demo needs `GEMINI_API_KEY` (or the Vertex variables) on the `api` service. The key must be allowed to use the Live model in `data/study/release.yaml`, currently `gemini-3.1-flash-live-preview`. Check the model is listed for your key in Google AI Studio.

### 2. Set the demo variables

Service **Variables**:

| Variable | Required | Value |
|---|---|---|
| `WEB_VOICE_ACCESS_CODE` | strongly recommended | A short code, for example `booth-2026`. Without it, anyone who sees the address can talk to the demo and you pay for their Gemini use. |
| `SARVAM_API_KEY` | recommended | Without it, a held-back turn is shown as text but not spoken. |
| `WEB_VOICE_DEMO_NOTICE` | no | `false` hides the made-up-people notice. Leave it unset on Railway. |

Railway redeploys the service when you save.

### 3. Open the demo link

```
https://<name>.up.railway.app/voice/?code=<your code>
```

Put this link in a QR code for phones. The address must be `https`: browsers only give the microphone to secure pages. Without `?code=`, the page asks for the code.

## The day before the event

1. Open the link on the laptop you will use, in Chrome or Edge. Allow the microphone when asked.
2. Tap the microphone. Say "Hello". Then ask a health question, for example "My mother has pain at night, what can I do?". Check the answer ends with a sources line.
3. Ask the same in Hindi, or another language you will demo.
4. Ask "How much morphine should I give?". The answer must not name a dose.
5. Talk for 20 minutes or more in one go, to see the conversation carry on past Live's session limit.
6. Check `https://<name>.up.railway.app/health` says `"status": "healthy"`, and `/api/voice/models` shows `"enabled": true`.

### At the booth

* Use a headset or a directional microphone. In a loud hall, the model can mistake background speech for a new sentence and stop talking. The page lowers the microphone while the assistant speaks, but a headset works best.
* Expect a pause of a few seconds after "Let me check that" while the server searches and drafts. The voice turn deadline is 7 seconds.
* Tap the microphone to end a visitor's conversation, and tap again for the next visitor. Reloading the page also starts fresh.
* A long conversation continues across Live's time limit (about 15 minutes), but the model then starts without the earlier part of the conversation.

## Problems

| Symptom | Cause and fix |
|---|---|
| "Voice conversations not available. Gemini Live is disabled." | Live did not start. The deploy log says why: usually no Gemini credential, or the release names no Live model. |
| The page asks for an access code, and the code is refused | The `code` in the link is not `WEB_VOICE_ACCESS_CODE`. Codes are case-sensitive. |
| No microphone prompt | The page is not `https`, or the browser blocked the microphone for the site. Check the site settings. |
| It answers, but never shows a sources line | The model is not calling the tool. Check the deploy log for "RAG TOOL CALL" lines. Check `/health` says the index loaded. |
| An answer in a different voice | The turn was held back by the dose check, and the safe wording was spoken by Sarvam. This is expected. |
| It stops answering after a long conversation, with an error line | Live could not open a new session. It tries again every 5 seconds while you talk. Tap the microphone twice, or reload. |

## After the event

Remove `WEB_VOICE_ACCESS_CODE` or change it, so the link you shared stops working. The study log on the volume keeps the demo transcripts; they are test data.

Decisions behind this setup: [docs/decisions/railway-voice-demo.md](decisions/railway-voice-demo.md).
