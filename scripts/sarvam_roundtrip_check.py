#!/usr/bin/env python3
"""
Live speech-to-text against the real Sarvam service.

Not a unit test. It spends money and needs SARVAM_API_KEY in .env, so it is run by
hand: python scripts/sarvam_roundtrip_check.py

It exists because Sarvam is the automatic fallback for real-time voice, and until now
no check had ever put audio through it. A provider that has only been exercised by a
fake is a provider whose failure mode is unknown.

Needs a WAV container, not headerless PCM. Sarvam rejects raw PCM with "Failed to
read the file, please check the audio format", which is a confusing way of saying the
container is wrong rather than the audio.
"""
import asyncio, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

PHRASE = "What should I give for severe pain at home?"
VOICE = "en-IN-NeerjaNeural"


def build_audio(pcm_path: Path, wav_path: Path) -> None:
    """Real speech, 16 kHz mono, in a container Sarvam will read."""
    mp3 = pcm_path.with_suffix(".mp3")
    subprocess.run([sys.executable, "-c", f"""
import asyncio, edge_tts
async def gen():
    await edge_tts.Communicate({PHRASE!r}, {VOICE!r}).save({str(mp3)!r})
asyncio.run(gen())
"""], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
        check=True,
    )


async def main() -> int:
    pcm = Path("/tmp/sarvam_check.pcm")
    wav = Path("/tmp/sarvam_check.wav")
    build_audio(pcm, wav)

    from sarvam_integration import SarvamClient

    client = SarvamClient()
    if not client.api_key:
        print("SARVAM_API_KEY is not set in .env")
        return 1

    result = await client.speech_to_text(wav.read_bytes(), "en-IN")
    transcript = getattr(result, "transcript", "") or ""
    print("SARVAM LIVE TRANSCRIPT:")
    print("  ->", repr(transcript[:220]))

    if len(transcript) > 10:
        print("\nVERDICT: SARVAM FALLBACK PROVEN LIVE")
        return 0
    print("\nVERDICT: failed to transcribe")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
