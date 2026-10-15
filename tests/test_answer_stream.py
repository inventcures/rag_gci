"""
Answers are spoken sentence by sentence, each through the Dose Boundary.

Before this, a Retell caller heard nothing until the whole answer had been
written, guarded and sent as one message.

What these pin down:

  * sentences split on . ? ! and the danda (Hindi, Marathi), not on decimals,
    abbreviations or list numbers
  * a stubbed stream with a dose sentence in the middle sends the sentences
    before and after it, then the redaction note (or the Deferral when too
    little is left); no dose text is ever sent
  * an emergency question gets the 108 guidance first, then passes untouched,
    as the whole-text guard does
  * Retell: the first sentence goes out before the answer is finished; a
    barge-in cancels the stream cleanly; what was sent is kept as spoken
  * the LLM stream stays inside the caller's budget and falls back only before
    its first text
"""

import asyncio
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dosage_guard  # noqa: E402
from answer_stream import AnswerStream, SentenceSplitter, active_stream, streaming_to  # noqa: E402
from llm import client as llm_client  # noqa: E402
from llm.client import LLMClient, LLMSettings, LLMStreamError  # noqa: E402

DOSE = "Give morphine 5 mg every 4 hours."
SAFE = [
    "Noisy breathing near the end of life is common and rarely distressing for the patient.",
    "Turn her gently onto her side and keep her mouth moist with a damp cloth.",
    "Call the palliative care team at once if she struggles for breath.",
]


async def deltas(text, size=7, delay=0.0):
    for i in range(0, len(text), size):
        if delay:
            await asyncio.sleep(delay)
        yield text[i:i + size]


def split(text, size=5):
    s = SentenceSplitter()
    out = []
    for i in range(0, len(text), size):
        out += s.feed(text[i:i + size])
    return out + s.flush()


class Splitting(unittest.TestCase):
    def test_english(self):
        self.assertEqual(
            split("Give paracetamol 2.5 mg by mouth. Dr. Rao agreed, e.g. at night. Is it ok? Yes!"),
            ["Give paracetamol 2.5 mg by mouth.", "Dr. Rao agreed, e.g. at night.", "Is it ok?", "Yes!"])

    def test_hindi_and_marathi_danda(self):
        self.assertEqual(split("उन्हें करवट दिलाएँ। मुँह को गीला रखें। डॉक्टर को बुलाएँ?"),
                         ["उन्हें करवट दिलाएँ।", "मुँह को गीला रखें।", "डॉक्टर को बुलाएँ?"])
        self.assertEqual(split("त्यांना कुशीवर वळवा। तोंड ओले ठेवा। गरज असल्यास डॉक्टरांना बोलवा।"),
                         ["त्यांना कुशीवर वळवा।", "तोंड ओले ठेवा।", "गरज असल्यास डॉक्टरांना बोलवा।"])

    def test_lines_and_list_numbers(self):
        self.assertEqual(split("Steps:\n1. Turn her. 2. Moisten lips.\nDone"),
                         ["Steps:", "1. Turn her.", "2. Moisten lips.", "Done"])

    def test_closing_quote_stays_with_its_sentence(self):
        self.assertEqual(split('She said "rest." Then call.'), ['She said "rest."', "Then call."])


class Guarding(unittest.TestCase):
    def run_stream(self, text, query="noisy breathing", language="en"):
        sent = []

        async def send(sentence):
            sent.append(sentence)

        stream = AnswerStream(send, query=query, language=language)
        asyncio.run(stream.consume(deltas(text)))
        return stream, sent

    def test_dose_in_the_middle_is_redacted_then_noted(self):
        stream, sent = self.run_stream(" ".join([SAFE[0], SAFE[1], DOSE, SAFE[2]]))
        self.assertEqual(sent[:3], [SAFE[0], SAFE[1], SAFE[2]])
        self.assertEqual(sent[3], dosage_guard._redaction_note("en").strip())
        self.assertEqual(stream.redacted, 1)
        for piece in sent:
            self.assertFalse(dosage_guard.detect_doses(piece), piece)
        self.assertNotIn("5 mg", " ".join(sent))

    def test_too_little_left_gets_the_deferral(self):
        stream, sent = self.run_stream(f"Yes. {DOSE}", language="hi")
        self.assertEqual(sent, ["Yes.", dosage_guard._refusal_for("hi")])

    def test_dose_split_across_sentences_is_caught(self):
        stream, sent = self.run_stream(f"{SAFE[0]} Give morphine 5. mg every 4 hours at night. {SAFE[1]}")
        self.assertNotIn("mg every 4 hours", " ".join(sent))

    def test_clean_answer_passes_whole(self):
        stream, sent = self.run_stream(" ".join(SAFE))
        self.assertEqual(sent, SAFE)
        self.assertEqual(stream.text, " ".join(SAFE))

    def test_emergency_question_gets_the_guidance_then_passes_untouched(self):
        from safety_enhancements import EmergencyLevel

        class Critical:
            level = EmergencyLevel.CRITICAL

        class Emergency(dosage_guard.DosageGuard):
            def detect_alert(self, query, language="en"):
                return Critical()

        sent = []

        async def send(s):
            sent.append(s)

        stream = AnswerStream(send, query="he stopped breathing", guard=Emergency())
        asyncio.run(stream.consume(deltas(f"{SAFE[2]} {DOSE}")))
        self.assertEqual(sent, [dosage_guard.emergency_guidance_for("en"), SAFE[2], DOSE])

    def test_word_cap_stops_release(self):
        sent = []

        async def send(s):
            sent.append(s)

        stream = AnswerStream(send, max_words=20)
        asyncio.run(stream.consume(deltas(" ".join(SAFE * 3))))
        self.assertLessEqual(sum(len(s.split()) for s in sent), 20)
        self.assertTrue(stream.truncated)

    def test_failure_before_any_sentence_raises_after_stands(self):
        async def broken(after):
            yield after
            raise RuntimeError("stream died")

        async def go(first):
            stream = AnswerStream(lambda s: asyncio.sleep(0))
            return await stream.consume(broken(first))

        with self.assertRaises(RuntimeError):
            asyncio.run(go("Half a sent"))
        self.assertEqual(asyncio.run(go(SAFE[0] + " Half a sent")), SAFE[0])

    def test_active_stream_is_single_use(self):
        stream = AnswerStream(lambda s: asyncio.sleep(0))
        with streaming_to(stream):
            self.assertIs(active_stream(), stream)
            asyncio.run(stream.consume(deltas("Hi.")))
            self.assertIsNone(active_stream())
        self.assertIsNone(active_stream())


class StreamingClient(LLMClient):
    def __init__(self, settings, pieces, delay=0.0, fail_after=None):
        super().__init__(settings)
        self.pieces, self.delay, self.fail_after = pieces, delay, fail_after

    async def _gemini_stream(self, prompt, system, max_tokens, temperature, thinking):
        async def gen():
            for n, piece in enumerate(self.pieces):
                if self.fail_after is not None and n == self.fail_after:
                    raise RuntimeError("connection reset")
                await asyncio.sleep(self.delay)
                yield type("Chunk", (), {"text": piece, "usage_metadata": None})()
        return gen()

    async def _deepseek(self, prompt, system, max_tokens, temperature, json_mode):
        return "fallback answer."


class ClientStream(unittest.TestCase):
    def collect(self, client):
        async def go():
            return [p async for p in client.stream("x", purpose="answer")]
        return asyncio.run(go())

    def test_yields_pieces(self):
        c = StreamingClient(LLMSettings(api_key="k"), ["Hello ", "there."])
        self.assertEqual(self.collect(c), ["Hello ", "there."])

    def test_budget_bounds_a_stalled_stream(self):
        c = StreamingClient(LLMSettings(api_key="k", timeout_s=0.2), ["a", "b"], delay=5)
        started = time.perf_counter()
        with self.assertRaises(LLMStreamError) as caught:
            self.collect(c)
        self.assertFalse(caught.exception.partial)
        self.assertLess(time.perf_counter() - started, 0.6)

    def test_fallback_only_before_first_text(self):
        settings = LLMSettings(api_key="k", fallback_enabled=True, fallback_api_key="d")
        self.assertEqual(self.collect(StreamingClient(settings, ["a"], fail_after=0)), ["fallback answer."])
        with self.assertRaises(LLMStreamError) as caught:
            self.collect(StreamingClient(settings, ["a", "b"], fail_after=1))
        self.assertTrue(caught.exception.partial)


class Retell(unittest.TestCase):
    def setUp(self):
        self.previous = llm_client._client

    def tearDown(self):
        llm_client.set_llm_client(self.previous)

    def pipeline(self, pieces, delay):
        from tests.test_answer_path_retrieval import make_pipeline

        llm_client.set_llm_client(StreamingClient(LLMSettings(api_key="k"), pieces, delay=delay))
        p = make_pipeline()
        del p._generate_answer_with_citations  # the real one, streaming into the call
        import tempfile
        from pathlib import Path
        p.conversation_file = Path(tempfile.mkdtemp(prefix="retell-stream-")) / "conversations.json"
        return p

    def call(self, rag, script):
        from retell_integration.custom_llm_server import RetellCustomLLMHandler
        from tests.test_retell_turns import FakeSocket

        async def go():
            handler = RetellCustomLLMHandler(rag_pipeline=rag)
            ws = FakeSocket()
            call = asyncio.create_task(handler.handle_websocket(ws, "call-s"))
            await ws.inbox.put({"interaction_type": "call_details", "call": {"metadata": {"language": "en-IN"}}})
            out = await script(ws, handler)
            await ws.inbox.put(None)
            await call
            return ws, out

        return asyncio.run(go())

    def test_first_sentence_before_the_answer_is_finished(self):
        from tests.test_retell_turns import turn, wait_for

        text = " ".join([SAFE[0], SAFE[1], DOSE, SAFE[2]])
        pieces = [text[i:i + 12] for i in range(0, len(text), 12)]
        rag = self.pipeline(pieces, delay=0.03)

        async def script(ws, handler):
            loop = asyncio.get_running_loop()
            asked = loop.time()
            await ws.inbox.put(turn(1, "noisy breathing at the end of life"))
            await wait_for(lambda: ws.responses())
            first = loop.time() - asked
            await wait_for(lambda: any(r["content_complete"] for r in ws.responses()), timeout=5)
            done = loop.time() - asked
            return first, done, handler.active_sessions["call-s"].last_spoken

        ws, (first, done, spoken) = self.call(rag, script)
        responses = ws.responses()
        self.assertLess(first, done - 0.2, "the first sentence must not wait for the whole answer")
        self.assertFalse(responses[0]["content_complete"])
        heard = "".join(r["content"] for r in responses)
        self.assertNotIn("5 mg", heard)
        self.assertIn("not included here", heard)
        self.assertEqual(" ".join(heard.split()), " ".join(spoken.split()))

    def test_barge_in_cancels_the_stream(self):
        from tests.test_retell_turns import turn, wait_for

        text = " ".join(SAFE * 4)
        pieces = [text[i:i + 10] for i in range(0, len(text), 10)]
        rag = self.pipeline(pieces, delay=0.05)

        async def script(ws, handler):
            await ws.inbox.put(turn(1, "noisy breathing"))
            await wait_for(lambda: ws.responses())
            await ws.inbox.put(turn(2, "and she is restless"))
            await wait_for(lambda: any(r["response_id"] == 2 and r["content_complete"]
                                       for r in ws.responses()), timeout=10)
            await asyncio.sleep(0.2)

        ws, _ = self.call(rag, script)
        ones = [r for r in ws.responses() if r["response_id"] == 1]
        self.assertFalse(any(r["content_complete"] for r in ones), "turn 1 must stop, not finish")
        last_one = max(i for i, r in enumerate(ws.responses()) if r["response_id"] == 1)
        first_two = min(i for i, r in enumerate(ws.responses()) if r["response_id"] == 2)
        self.assertLess(last_one, first_two)

    def test_acknowledgement_when_the_first_sentence_is_slow(self):
        from tests.test_retell_turns import turn, wait_for

        rag = self.pipeline([SAFE[0]], delay=1.8)

        async def script(ws, handler):
            await ws.inbox.put(turn(1, "noisy breathing"))
            await wait_for(lambda: any(r["content_complete"] for r in ws.responses()), timeout=5)

        ws, _ = self.call(rag, script)
        first = ws.responses()[0]
        self.assertIn("One moment", first["content"])
        self.assertFalse(first["content_complete"])


if __name__ == "__main__":
    unittest.main()
