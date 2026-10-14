# Thinking level per purpose (gemini-3.5-flash)

Measured 2026-10-14 with `scripts/benchmark_thinking.py`, 6 calls in flight, streamed. Seconds.

| purpose | level | n | first token p50 | p95 | total p50 | p95 | errors |
|---|---|---|---|---|---|---|---|
| answer | minimal | 40 | 1.03 | 1.57 | 4.65 | 5.64 | 0 |
| answer | low | 40 | 7.51 | 9.23 | 10.82 | 12.91 | 0 |
| answer | medium | 40 | 11.98 | 16.30 | 14.92 | 18.94 | 0 |
| answer | high | 40 | 14.20 | 18.62 | 16.94 | 21.58 | 0 |
| query_translation | minimal | 36 | 0.77 | 1.01 | 0.82 | 1.08 | 0 |
| query_translation | low | 36 | 2.11 | 2.90 | 2.12 | 2.96 | 0 |
| query_translation | medium | 36 | 3.07 | 4.96 | 3.14 | 4.96 | 0 |
| query_translation | high | 36 | 3.02 | 4.87 | 3.02 | 4.88 | 0 |
| answer_translation | minimal | 16 | 0.85 | 1.44 | 3.81 | 4.54 | 0 |
| answer_translation | low | 16 | 8.51 | 11.26 | 10.28 | 11.98 | 0 |
| answer_translation | medium | 16 | 18.50 | 20.15 | 19.34 | 21.00 | 0 |
| answer_translation | high | 16 | 34.68 | 35.78 | 35.94 | 37.21 | 0 |
| clarification | minimal | 24 | 0.82 | 1.44 | 1.91 | 2.67 | 0 |
| clarification | low | 24 | 0.80 | 3.71 | 1.94 | 4.59 | 0 |
| clarification | medium | 24 | 4.53 | 7.95 | 5.30 | 8.77 | 0 |
| clarification | high | 24 | 5.27 | 7.18 | 6.27 | 7.98 | 0 |

## Quality against `high`

```json
{
  "answer": {
    "minimal": {
      "doses_before_guard": 19,
      "doses_after_guard": 0,
      "cut_off": 0,
      "wins": 12,
      "losses": 28,
      "ties": 0,
      "net_loss_rate": 0.4
    },
    "low": {
      "doses_before_guard": 18,
      "doses_after_guard": 0,
      "cut_off": 0,
      "wins": 21,
      "losses": 19,
      "ties": 0,
      "net_loss_rate": -0.05
    },
    "medium": {
      "doses_before_guard": 12,
      "doses_after_guard": 1,
      "cut_off": 0,
      "wins": 18,
      "losses": 22,
      "ties": 0,
      "net_loss_rate": 0.1
    },
    "high": {
      "doses_before_guard": 17,
      "doses_after_guard": 0,
      "cut_off": 0
    }
  },
  "query_translation": {
    "minimal": {
      "same_meaning": 36,
      "n": 36
    },
    "low": {
      "same_meaning": 36,
      "n": 36
    },
    "medium": {
      "same_meaning": 36,
      "n": 36
    },
    "high": {
      "same_meaning": 36,
      "n": 36
    }
  },
  "answer_translation": {
    "minimal": {
      "faithful": 7,
      "n": 16,
      "over_1500": 11,
      "cut_off": 0
    },
    "low": {
      "faithful": 9,
      "n": 16,
      "over_1500": 6,
      "cut_off": 3
    },
    "medium": {
      "faithful": 9,
      "n": 16,
      "over_1500": 5,
      "cut_off": 8
    },
    "high": {
      "faithful": 7,
      "n": 16,
      "over_1500": 3,
      "cut_off": 10
    }
  },
  "clarification": {
    "minimal": {
      "parsed": 24,
      "n": 24,
      "agrees_with_high": 17
    },
    "low": {
      "parsed": 24,
      "n": 24,
      "agrees_with_high": 17
    },
    "medium": {
      "parsed": 24,
      "n": 24,
      "agrees_with_high": 21
    },
    "high": {
      "parsed": 24,
      "n": 24,
      "agrees_with_high": 24
    }
  }
}
```

## Reading it

- **Before this ticket every call ran at the model's default level, which is
  `high`,** with no headroom for thinking tokens. On the same 40 answer
  prompts, that cut all four checked answers off mid-sentence (~400
  characters) after ~11.5 s (`finish_reason=MAX_TOKENS`). The `high` row
  above was measured with headroom, so it shows what `high` costs when it
  works, not what production did.
- **Answers:** `low` is level with `high` (judged 21 wins, 19 losses) and
  halves time to first token. `minimal` is the only level that fits a voice
  turn (first token 1.0 s, whole answer 4.7 s), but it lost to `high` on 28 of
  40. Voice turns use `minimal`; ticket 12 rewrites the voice prompt for
  ~80-word answers, and clinicians review a sample before release.
- **Dose Boundary:** 12-19 drafts per level contained a dose; none did after
  the guard. The one detector hit after the guard (`medium`) is "immediately
  escalate to your local health worker", an escalation cue, not a dose.
- **Query translation:** every level gave the probe's meaning on 36 of 36.
  `minimal` is 0.8 s against 3.0 s.
- **Answer translation:** `high` takes 36 s and is cut off on 10 of 16;
  `minimal` takes 3.8 s and is never cut off. All levels run past the prompt's
  1500-character limit on some answers; faithfulness judged similar (7-9 of
  16). Its deadline is 8 s, not 3 s: `minimal` p95 is 4.5 s.
- **Clarification:** `minimal` and `low` agree with `high` on 17 of 24
  (`medium` 21). `high` is a reference, not ground truth; `minimal` is chosen
  for latency. Revisit if clarification prompts misfire in the field.

Judging is by the same model at `high`, with answer order randomised; it is
a screening comparison, not a clinical evaluation. Raw outputs and verdicts:
`/mnt/project-files/research/latency-evals/thinking-11.json` (project files).
Retrieval evaluation is unaffected: retrieval uses no model call except query
translation, which was judged above.
