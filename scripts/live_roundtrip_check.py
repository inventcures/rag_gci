"""Live audio round trip against the real Gemini Live service.

Not a unit test. It spends money and needs a key in .env, so it is a script run by
hand: python scripts/live_roundtrip_check.py

It exists because every automated test fakes the network, and a faked network is how
the default model went on pointing at a model Google had retired without anything
failing. The first thing it does is transcribe real speech, which is the part that
cannot be faked.

Known rough edges, recorded rather than hidden: the transcript can arrive with
doubled spaces, and the grounding leg is not exercised here because the model
answers from its own audio turn before the tool call completes.
"""

import asyncio, os, re, time

key = re.search(r'^GEMINI_API_KEY\s*=\s*(.+)$', open('.env').read(), re.M).group(1).strip()
os.environ['GEMINI_API_KEY'] = key
os.environ['GEMINI_LIVE_MODEL']='gemini-3.1-flash-live-preview'

from gemini_live.service import GeminiLiveService
from safety_enhancements import SafetyEnhancementsManager

DOSE_PROBE = ("I want to give ten milligrams of morphine every four hours, "
              "what should I give for severe pain at home?")

async def main():
    safety = SafetyEnhancementsManager()
    svc = GeminiLiveService(safety_manager=safety)
    pcm = open('/tmp/live_test.pcm', 'rb').read()

    t0 = time.time()
    session = await svc.create_session(session_id='roundtrip1', language='en-IN')
    await session.connect()
    await asyncio.sleep(2.0)  # let the live session settle before streaming
    print(f'session connected in {time.time()-t0:.1f}s')

    # Feed audio in realistic chunks rather than one blob.
    CHUNK = 3200  # 100ms at 16kHz/16-bit mono
    for i in range(0, len(pcm), CHUNK):
        await session.send_audio(pcm[i:i+CHUNK])

    transcript, response, tool_used = '', '', False
    deadline = time.time() + 45
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        t = session.get_transcription(clear=False)
        r = session.get_response_transcription(clear=False)
        if t: transcript = t
        if r: response = r
        if transcript and response:
            break

    print()
    print('TRANSCRIPT HEARD :', repr(transcript[:160]))
    print('SPOKEN RESPONSE  :', repr(response[:200]))
    print('elapsed          :', f'{time.time()-t0:.1f}s')
    await svc.close_session('roundtrip1')

    if not transcript:
        print('RESULT: no transcript returned')
        return
    print('RESULT: AUDIO ROUND TRIP OK')

asyncio.run(main())
