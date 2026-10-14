"""
Sarvam calls share one keep-alive connection.

Before this, every STT, TTS and translate call opened its own HTTP session,
so each paid a TCP connect and a TLS handshake; a mobile voice turn (STT then
TTS) paid it twice.

What these pin down, against a local HTTP server standing in for Sarvam:

  * an STT call, a TTS call and a translate call in a row open one connection
  * the trace says so: the first `sarvam.*` span has `new_connection: true`,
    the rest `false`
  * the session is per event loop (a second loop gets its own), and closes
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sarvam_integration import SarvamClient  # noqa: E402
from sarvam_integration.http import close_shared_session, shared_session  # noqa: E402
from tests.test_tracing import read_spans  # noqa: E402


def fake_sarvam():
    app = web.Application()
    app["peers"] = set()

    def seen(request):
        app["peers"].add(request.transport.get_extra_info("peername"))

    async def stt(request):
        seen(request)
        await request.read()
        return web.json_response({"transcript": "saans phool rahi hai", "language_code": "hi-IN"})

    async def tts(request):
        seen(request)
        return web.json_response({"audios": ["UENN"]})

    async def translate(request):
        seen(request)
        return web.json_response({"translated_text": "breathless"})

    app.router.add_post("/speech-to-text", stt)
    app.router.add_post("/text-to-speech", tts)
    app.router.add_post("/translate", translate)
    return app


class KeepAlive(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traces-")
        self.env = patch.dict(os.environ, {"TRACE_DIR": self.dir, "TRACING_ENABLED": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_one_connection_for_a_turn_and_the_trace_says_so(self):
        async def go():
            app = fake_sarvam()
            server = TestServer(app)
            await server.start_server()
            try:
                client = SarvamClient(api_key="k", base_url=str(server.make_url("")).rstrip("/"))
                stt = await client.speech_to_text(b"RIFF....", "hi-IN")
                tts = await client.text_to_speech("thoda baith jaiye", "hi-IN")
                tr = await client.translate("saans", "hi-IN", "en-IN")
                await close_shared_session()
                return app["peers"], (stt, tts, tr)
            finally:
                await server.close()

        peers, results = asyncio.run(go())
        self.assertTrue(all(r.success for r in results), results)
        self.assertEqual(len(peers), 1, "each call opened its own connection")
        spans = {s["name"]: s for s in read_spans(self.dir)}
        self.assertTrue(spans["sarvam.stt"]["attrs"]["new_connection"])
        self.assertFalse(spans["sarvam.tts"]["attrs"]["new_connection"])
        self.assertFalse(spans["sarvam.translate"]["attrs"]["new_connection"])

    def test_session_per_loop_and_closed_on_shutdown(self):
        async def get():
            s = shared_session()
            self.assertIs(s, shared_session())
            await close_shared_session()
            return s

        first, second = asyncio.run(get()), asyncio.run(get())
        self.assertIsNot(first, second)
        self.assertTrue(first.closed and second.closed)


if __name__ == "__main__":
    unittest.main()
