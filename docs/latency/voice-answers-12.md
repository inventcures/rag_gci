# Voice-length answers (gemini-3.5-flash)

`scripts/benchmark_voice_answers.py`, 40 vignettes, 6 calls in flight. Text: text prompt at the text level. Voice: voice prompt at the voice level.

| | text | voice |
|---|---|---|
| output tokens (sum) | 21461 | 3584 |
| words p50 | 312 | 74 |
| first token p50 (s) | 7.33 | 0.95 |
| total p50 (s) | 10.62 | 1.39 |

```json
{
  "n": 40,
  "errors": 0,
  "output_tokens_text": 21461,
  "output_tokens_voice": 3584,
  "output_token_drop": 0.833,
  "voice_words_p50": 74,
  "voice_words_max": 87,
  "text_words_p50": 312,
  "voice_first_token_p50_s": 0.9527193539997825,
  "voice_total_p50_s": 1.3864916620004806,
  "voice_total_p95_s": 1.991045712999039,
  "text_first_token_p50_s": 7.326740471999074,
  "text_total_p50_s": 10.623864942001092,
  "cut_by_safety_net": 0,
  "cut_off_by_token_cap": 0,
  "dose_before_guard": 2,
  "dose_after_guard": 0,
  "dose_referral": 37
}
```
## Voice answer quality by thinking level

The same 40 voice prompts, judged pairwise by the model at `high` (order
randomised). First token and total in seconds, p50 (total p95).

| level | first token | total | vs high | vs low |
|---|---|---|---|---|
| minimal | 0.99 | 1.39 (2.01) | 18 wins, 22 losses | 10 wins, 30 losses |
| low | 6.27 | 6.58 (8.89) | 25 wins, 15 losses | - |
| high | 12.90 | 13.19 (16.84) | - | - |

With the voice prompt, `minimal` is close to `high` (it lost 28 of 40 with the
text prompt). `low` is judged better than both but its p95 runs past the 7 s
voice deadline before retrieval is counted. See docs/decisions/latency.md#12.

The clinician review sheet (each spoken answer as a caller hears it) is in
project files: `research/latency-evals/voice-answers-review.md`.
