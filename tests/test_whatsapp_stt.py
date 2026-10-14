"""WhatsApp voice notes are transcribed by Sarvam (India-hosted), not Groq Whisper."""

import asyncio
from types import SimpleNamespace

from whatsapp_bot import EnhancedSTTService


class FakeSarvam:
    def __init__(self, fail_unknown=False, heard="ta-IN"):
        self.calls = []
        self.fail_unknown = fail_unknown
        self.heard = heard

    def is_available(self):
        return True

    async def speech_to_text(self, audio, language, with_timestamps=False):
        self.calls.append(language)
        if language == "unknown":
            if self.fail_unknown:
                return SimpleNamespace(success=False, error="bad request", transcript="", language_code="")
            return SimpleNamespace(success=True, transcript="வணக்கம்", language_code=self.heard, error=None)
        return SimpleNamespace(success=True, transcript="नमस्ते", language_code=language, error=None)


def _service(fake):
    s = EnhancedSTTService.__new__(EnhancedSTTService)
    EnhancedSTTService.__init__(s)
    s.sarvam = fake
    return s


def _run(service, tmp_path, language=None):
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"\x00" * 16)
    return asyncio.run(service.transcribe_audio(str(audio), language))


def test_auto_detects_language_in_one_call(tmp_path):
    fake = FakeSarvam()
    r = _run(_service(fake), tmp_path)
    assert r["status"] == "success" and r["detected_language"] == "ta"
    assert fake.calls == ["unknown"]


def test_odia_code_maps_back(tmp_path):
    r = _run(_service(FakeSarvam(heard="od-IN")), tmp_path)
    assert r["detected_language"] == "or"


def test_given_language_is_used_directly(tmp_path):
    fake = FakeSarvam()
    r = _run(_service(fake), tmp_path, language="mr")
    assert r["status"] == "success" and fake.calls == ["mr-IN"]


def test_falls_back_to_hindi_when_detection_fails(tmp_path):
    fake = FakeSarvam(fail_unknown=True)
    r = _run(_service(fake), tmp_path)
    assert r["status"] == "success" and r["detected_language"] == "hi"
    assert fake.calls == ["unknown", "hi-IN"]
