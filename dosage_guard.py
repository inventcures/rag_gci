#!/usr/bin/env python3
"""
Palli Sahayak Dosage Guard
==========================
Enforces the study restriction that the system never provides specific medication
dosages, while still answering general drug questions.

Rationale (ADR-0004):
    The grant proposal states the system "never provides specific medication dosages"
    and the study protocol (SS7.7) requires that this restriction be *enforced and
    tested* before any participant uses the build.  The boundary is drawn at the
    *dose*, not at the OTC/prescription distinction, because:

    1. "OTC" is not a safety property for this cohort.  Participants have advanced
       kidney and liver disease, frailty and dementia, where a freely available
       medicine can still be unsafe.
    2. There is no Rx/OTC classification data in this repository to test against.
    3. A dose is a dose regardless of how it was purchased.  "Paracetamol 500mg once
       daily" breaches the promise exactly as a morphine instruction does.
    4. A dose boundary is machine-checkable, which SS7.7 requires.

Answered freely (tier 1):
    What a medicine is, what class it belongs to, what it treats, general
    precautions, what symptoms warrant a doctor, non-drug management, and how to
    take a medicine a prescriber has already supplied.

Refused and deferred (tier 2):
    Any specific dose, strength, titration step, escalation schedule,
    drug-interaction judgement, or start/stop decision.

Severity ordering:
    An emergency always wins.  A refusal must never swallow a myocardial
    infarction.  When a query is both a dosage question and an emergency trigger,
    the emergency path runs and the deferral never fires.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import dose_detector
from supported_languages import get_language

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dose detection
# ---------------------------------------------------------------------------

# Every dose pattern lives in dose_detector. DoseFinding and detect_doses are
# re-exported here because callers and the safety record already import them
# from this module.
DoseFinding = dose_detector.DoseFinding
detect_doses = dose_detector.detect_doses
_dedupe = dose_detector.dedupe_findings


# ---------------------------------------------------------------------------
# Deferral copy
# ---------------------------------------------------------------------------

# The deferral must never imply that a human has been contacted. Protocol SS7.6
# forbids implying a transfer has succeeded before receipt is confirmed, and no
# notification path exists yet.  The wording below is therefore about the advice,
# not about a dispatch that the system cannot perform.
_REFUSAL_EN = (
    "I understand this is what you need right now, and I am not able to help with "
    "the specific quantity or schedule of a medicine — deciding that needs someone who "
    "can see the patient.\n\n"
    "What I can do is help you prepare for that conversation:\n"
    "• Write down the symptoms, when they started, and what has already been given\n"
    "• Bring the medicine packets or prescription along\n"
    "• Raise it at the next consultation — with the palliative care doctor, the primary "
    "health centre, or the OPD you are registered with\n\n"
    "If the person is in severe distress right now — struggling to breathe, unresponsive, "
    "or in pain that cannot be settled — call 108 for an ambulance rather than waiting "
    "for the next visit."
)

_REFUSAL_LOCALIZED = {
    "hi": (
        "मैं समझ सकता हूँ कि यह अभी आपको ज़रूरी है। लेकिन किसी दवा की खास मात्रा या समय-सारणी "
        "मैं नहीं बता सकता — यह तय करने के लिए मरीज़ को देखने वाले चिकित्सक की ज़रूरत है।\n\n"
        "मैं इस बातचीत की तैयारी में मदद कर सकता हूँ:\n"
        "• लक्षण, उन्हें कब शुरू हुए, और अब तक क्या दिया गया — लिख लें\n"
        "• दवा के पाउच या प्रिस्क्रिप्शन साथ ले जाएँ\n"
        "• अगली मुलाक़ात में डॉक्टर के सामने यह बात रखें — पैलिएटिव केयर डॉक्टर, प्राथमिक स्वास्थ्य "
        "केंद्र, या जिस OPD में पंजीकरण है\n\n"
        "अगर मरीज़ को अभी बहुत तेज़ तकलीफ़ है — साँस लेने में दिक्कत, बेहोशी, या न ठीक होने "
        "वाला दर्द — तो अगली मुलाक़ात का इंतज़ार न करें, 108 पर एम्बुलेंस बुलाएँ।"
    ),
    "bn": (
        "আমি বুঝতে পারছি এটি এখন আপনার দরকার। তবে কোনো ওষুধের নির্দিষ্ট পরিমাণ বা সময়সূচি "
        "আমি বলতে পারি না — সেটা ঠিক করতে রোগীকে দেখেন এমন চিকিৎসক দরকার।\n\n"
        "আমি এই কথোপকথনের প্রস্তুতিতে সাহায্য করতে পারি:\n"
        "• লক্ষণ, কখন শুরু হয়েছে, এখন পর্যন্ত কী দেওয়া হয়েছে — লিখে রাখুন\n"
        "• ওষুধের প্যাকেট বা প্রেসক্রিপশন সঙ্গে নিন\n"
        "• পরবর্তী সাক্ষাতে ডাক্তারের সঙ্গে এ বিষয়ে কথা বলুন — প্যালিয়েটিভ কেয়ার ডাক্তার, "
        "প্রাথমিক স্বাস্থ্য কেন্দ্র, বা আপনার নিবন্ধিত OPD\n\n"
        "রোগী এখন খুব কষ্টে থাকলে — শ্বাসকষ্ট, অচেতন, বা সামলানো যায় না এমন ব্যথা — "
        "পরের দিনের জন্য অপেক্ষা না করে 108 নম্বরে অ্যাম্বুলেন্স ডাকুন।"
    ),
    "ta": (
        "இது இப்போது உங்களுக்கு தேவை என்று புரிகிறது. ஆனால் மருந்தின் குறிப்பிட்ட அளவு அல்லது "
        "நேரத்தை நான் தெரிவிக்க முடியாது — அதைத் தீர்மானிக்க நோயாளியைப் பார்க்கும் மருத்துவர் தேவை.\n\n"
        "இந்தப் பேச்சுக்கு தயார்படுத உதவலாம்:\n"
        "• அறிகுறிகள், எப்போது தொடங்கியது, இதுவரை என்ன கொடுக்கப்பட்டது — எழுதி வையுங்கள்\n"
        "• மருந்துப் பொதி அல்லது மருந்துச்சான்று எடுத்துச் செல்லுங்கள்\n"
        "• அடுத்த சந்திப்பில் மருத்துவரிடம் இதை முன்வைக்கவும் — பாலியேட்டிவ் கேர் மருத்துவர், "
        "முதல்நிலை சுகாதார மையம், அல்லது நீங்கள் பதிவு செய்த OPD\n\n"
        "நோயாளர் இப்போது மிகவும் வேதனையில் இருந்தால் — மூச்சுத் திணறல், நினைவிழப்பு, அல்லது "
        "சமாளிக்க முடியாத வலி — அடுத்த விஜயத்தை காத்திருக்காமல் 108-ல் ஆம்புலன்ஸை அழைக்கவும்."
    ),
    # mr, kn, ml, or, pa, te and gu were added in round 3 (ticket 04). They are
    # machine-drafted and await the clinical lead's language review. "Next time"
    # is written "next visit" throughout: the Indic words for "time"/"day"
    # (बार, ಬಾರಿ, ਵਾਰ, సారి, ...) are dose-frequency words to the detector.
    "mr": (
        "मला समजते की तुम्हाला आत्ता याची गरज आहे. पण एखाद्या औषधाचे नेमके प्रमाण किंवा "
        "वेळापत्रक मी सांगू शकत नाही — ते ठरवण्यासाठी रुग्णाला प्रत्यक्ष पाहणाऱ्या डॉक्टरांची गरज आहे.\n\n"
        "या संभाषणाच्या तयारीसाठी मी मदत करू शकतो:\n"
        "• लक्षणे, ती केव्हा सुरू झाली आणि आतापर्यंत काय दिले आहे — हे लिहून ठेवा\n"
        "• औषधांची पाकिटे किंवा प्रिस्क्रिप्शन सोबत न्या\n"
        "• पुढील भेटीत हा प्रश्न डॉक्टरांसमोर मांडा — पॅलिएटिव्ह केअर डॉक्टर, प्राथमिक आरोग्य "
        "केंद्र, किंवा तुमची नोंदणी असलेली OPD\n\n"
        "रुग्णाला आत्ता खूप त्रास होत असेल — श्वास घेण्यास अडचण, बेशुद्धी, किंवा न थांबणारी "
        "वेदना — तर पुढील भेटीची वाट न पाहता 108 वर रुग्णवाहिका बोलवा."
    ),
    "kn": (
        "ಇದು ನಿಮಗೆ ಈಗ ಅಗತ್ಯ ಎಂದು ನನಗೆ ಅರ್ಥವಾಗುತ್ತದೆ. ಆದರೆ ಯಾವುದೇ ಔಷಧಿಯ ನಿರ್ದಿಷ್ಟ ಪ್ರಮಾಣ ಅಥವಾ "
        "ವೇಳಾಪಟ್ಟಿಯನ್ನು ನಾನು ಹೇಳಲು ಸಾಧ್ಯವಿಲ್ಲ — ಅದನ್ನು ನಿರ್ಧರಿಸಲು ರೋಗಿಯನ್ನು ನೋಡುವ ವೈದ್ಯರು ಬೇಕು.\n\n"
        "ಈ ಮಾತುಕತೆಗೆ ತಯಾರಾಗಲು ನಾನು ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:\n"
        "• ಲಕ್ಷಣಗಳು, ಅವು ಯಾವಾಗ ಶುರುವಾದವು, ಮತ್ತು ಇಲ್ಲಿಯವರೆಗೆ ಏನು ಕೊಡಲಾಗಿದೆ — ಬರೆದಿಟ್ಟುಕೊಳ್ಳಿ\n"
        "• ಔಷಧಿಯ ಪ್ಯಾಕೆಟ್‌ಗಳು ಅಥವಾ ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್ ಜೊತೆಗೆ ತೆಗೆದುಕೊಂಡು ಹೋಗಿ\n"
        "• ಮುಂದಿನ ಭೇಟಿಯಲ್ಲಿ ಇದನ್ನು ವೈದ್ಯರ ಮುಂದೆ ಇಡಿ — ಪ್ಯಾಲಿಯೇಟಿವ್ ಕೇರ್ ವೈದ್ಯರು, ಪ್ರಾಥಮಿಕ "
        "ಆರೋಗ್ಯ ಕೇಂದ್ರ, ಅಥವಾ ನೀವು ನೋಂದಾಯಿಸಿರುವ OPD\n\n"
        "ರೋಗಿಗೆ ಈಗ ತೀವ್ರ ತೊಂದರೆ ಇದ್ದರೆ — ಉಸಿರಾಟದ ತೊಂದರೆ, ಪ್ರಜ್ಞೆ ಇಲ್ಲದಿರುವುದು, ಅಥವಾ ಶಮನವಾಗದ "
        "ನೋವು — ಮುಂದಿನ ಭೇಟಿಗಾಗಿ ಕಾಯದೆ 108 ಗೆ ಕರೆ ಮಾಡಿ ಆಂಬ್ಯುಲೆನ್ಸ್ ಕರೆಸಿ."
    ),
    "ml": (
        "ഇത് ഇപ്പോൾ നിങ്ങൾക്ക് ആവശ്യമാണെന്ന് എനിക്ക് മനസ്സിലാകുന്നു. പക്ഷേ ഒരു മരുന്നിന്റെ കൃത്യമായ "
        "അളവോ സമയക്രമമോ എനിക്ക് പറയാൻ കഴിയില്ല — അത് തീരുമാനിക്കാൻ രോഗിയെ നേരിട്ട് കാണുന്ന ഡോക്ടർ വേണം.\n\n"
        "ഈ സംഭാഷണത്തിന് തയ്യാറെടുക്കാൻ ഞാൻ സഹായിക്കാം:\n"
        "• ലക്ഷണങ്ങൾ, അവ എപ്പോൾ തുടങ്ങി, ഇതുവരെ എന്തൊക്കെ നൽകി — എഴുതി വയ്ക്കുക\n"
        "• മരുന്നിന്റെ പാക്കറ്റുകളോ കുറിപ്പടിയോ കൂടെ കൊണ്ടുപോകുക\n"
        "• അടുത്ത സന്ദർശനത്തിൽ ഇത് ഡോക്ടറോട് ചോദിക്കുക — പാലിയേറ്റീവ് കെയർ ഡോക്ടർ, "
        "പ്രാഥമികാരോഗ്യ കേന്ദ്രം, അല്ലെങ്കിൽ നിങ്ങൾ രജിസ്റ്റർ ചെയ്ത OPD\n\n"
        "രോഗിക്ക് ഇപ്പോൾ കടുത്ത ബുദ്ധിമുട്ട് ഉണ്ടെങ്കിൽ — ശ്വാസതടസ്സം, ബോധമില്ലായ്മ, അല്ലെങ്കിൽ "
        "ശമിക്കാത്ത വേദന — അടുത്ത സന്ദർശനത്തിനായി കാത്തിരിക്കാതെ 108-ൽ വിളിച്ച് ആംബുലൻസ് വരുത്തുക."
    ),
    "or": (
        "ମୁଁ ବୁଝିପାରୁଛି ଯେ ଏହା ଆପଣଙ୍କୁ ବର୍ତ୍ତମାନ ଦରକାର। କିନ୍ତୁ କୌଣସି ଔଷଧର ନିର୍ଦ୍ଦିଷ୍ଟ ପରିମାଣ ବା "
        "ସମୟସୂଚୀ ମୁଁ କହିପାରିବି ନାହିଁ — ତାହା ସ୍ଥିର କରିବାକୁ ରୋଗୀଙ୍କୁ ଦେଖୁଥିବା ଡାକ୍ତର ଆବଶ୍ୟକ।\n\n"
        "ଏହି ଆଲୋଚନା ପାଇଁ ପ୍ରସ୍ତୁତ ହେବାରେ ମୁଁ ସାହାଯ୍ୟ କରିପାରିବି:\n"
        "• ଲକ୍ଷଣ, ସେଗୁଡ଼ିକ କେବେ ଆରମ୍ଭ ହେଲା, ଏବଂ ଏପର୍ଯ୍ୟନ୍ତ କଣ ଦିଆଯାଇଛି — ଲେଖି ରଖନ୍ତୁ\n"
        "• ଔଷଧର ପ୍ୟାକେଟ୍ କିମ୍ବା ପ୍ରେସକ୍ରିପସନ୍ ସାଙ୍ଗରେ ନିଅନ୍ତୁ\n"
        "• ପରବର୍ତ୍ତୀ ସାକ୍ଷାତରେ ଡାକ୍ତରଙ୍କ ସହ ଏହି ବିଷୟରେ କଥା ହୁଅନ୍ତୁ — ପାଲିଏଟିଭ୍ କେୟାର ଡାକ୍ତର, "
        "ପ୍ରାଥମିକ ସ୍ୱାସ୍ଥ୍ୟ କେନ୍ଦ୍ର, କିମ୍ବା ଆପଣ ପଞ୍ଜୀକୃତ ହୋଇଥିବା OPD\n\n"
        "ଯଦି ରୋଗୀ ବର୍ତ୍ତମାନ ବହୁତ କଷ୍ଟରେ ଅଛନ୍ତି — ନିଶ୍ୱାସ ନେବାରେ ଅସୁବିଧା, ଚେତନା ନାହିଁ, କିମ୍ବା ଆୟତ୍ତ "
        "ହେଉନଥିବା ଯନ୍ତ୍ରଣା — ପରବର୍ତ୍ତୀ ସାକ୍ଷାତକୁ ଅପେକ୍ଷା ନକରି 108 ରେ ଆମ୍ବୁଲାନ୍ସ ଡାକନ୍ତୁ।"
    ),
    "pa": (
        "ਮੈਂ ਸਮਝਦਾ ਹਾਂ ਕਿ ਤੁਹਾਨੂੰ ਇਸ ਵੇਲੇ ਇਸ ਦੀ ਲੋੜ ਹੈ। ਪਰ ਕਿਸੇ ਦਵਾਈ ਦੀ ਖ਼ਾਸ ਮਾਤਰਾ ਜਾਂ ਸਮਾਂ-ਸਾਰਣੀ "
        "ਮੈਂ ਨਹੀਂ ਦੱਸ ਸਕਦਾ — ਇਹ ਤੈਅ ਕਰਨ ਲਈ ਮਰੀਜ਼ ਨੂੰ ਦੇਖਣ ਵਾਲੇ ਡਾਕਟਰ ਦੀ ਲੋੜ ਹੈ।\n\n"
        "ਮੈਂ ਇਸ ਗੱਲਬਾਤ ਦੀ ਤਿਆਰੀ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ:\n"
        "• ਲੱਛਣ, ਉਹ ਕਦੋਂ ਸ਼ੁਰੂ ਹੋਏ, ਅਤੇ ਹੁਣ ਤੱਕ ਕੀ ਦਿੱਤਾ ਗਿਆ ਹੈ — ਲਿਖ ਕੇ ਰੱਖੋ\n"
        "• ਦਵਾਈ ਦੇ ਪੈਕਟ ਜਾਂ ਪਰਚੀ ਨਾਲ ਲੈ ਕੇ ਜਾਓ\n"
        "• ਅਗਲੀ ਮੁਲਾਕਾਤ ਵਿੱਚ ਡਾਕਟਰ ਨਾਲ ਇਹ ਗੱਲ ਕਰੋ — ਪੈਲੀਏਟਿਵ ਕੇਅਰ ਡਾਕਟਰ, ਮੁੱਢਲਾ ਸਿਹਤ ਕੇਂਦਰ, "
        "ਜਾਂ ਉਹ OPD ਜਿੱਥੇ ਤੁਹਾਡਾ ਨਾਮ ਦਰਜ ਹੈ\n\n"
        "ਜੇ ਮਰੀਜ਼ ਨੂੰ ਇਸ ਵੇਲੇ ਬਹੁਤ ਤਕਲੀਫ਼ ਹੈ — ਸਾਹ ਲੈਣ ਵਿੱਚ ਮੁਸ਼ਕਲ, ਬੇਹੋਸ਼ੀ, ਜਾਂ ਨਾ ਠੀਕ ਹੋਣ ਵਾਲਾ "
        "ਦਰਦ — ਤਾਂ ਅਗਲੀ ਮੁਲਾਕਾਤ ਦੀ ਉਡੀਕ ਨਾ ਕਰੋ, 108 ਤੇ ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ।"
    ),
    "te": (
        "ఇది మీకు ఇప్పుడు అవసరమని నాకు అర్థమవుతోంది. కానీ ఏ మందు యొక్క నిర్దిష్ట పరిమాణం లేదా "
        "సమయపట్టికను నేను చెప్పలేను — దాన్ని నిర్ణయించడానికి రోగిని నేరుగా చూసే వైద్యులు కావాలి.\n\n"
        "ఈ సంభాషణకు సిద్ధం కావడంలో నేను సహాయం చేయగలను:\n"
        "• లక్షణాలు, అవి ఎప్పుడు మొదలయ్యాయి, ఇప్పటివరకు ఏమి ఇచ్చారు — రాసి పెట్టుకోండి\n"
        "• మందుల ప్యాకెట్లు లేదా ప్రిస్క్రిప్షన్ వెంట తీసుకెళ్లండి\n"
        "• వచ్చే సందర్శనలో దీని గురించి వైద్యులతో మాట్లాడండి — పాలియేటివ్ కేర్ వైద్యులు, ప్రాథమిక "
        "ఆరోగ్య కేంద్రం, లేదా మీరు నమోదు చేసుకున్న OPD\n\n"
        "రోగి ఇప్పుడు తీవ్ర బాధలో ఉంటే — ఊపిరి తీసుకోవడంలో ఇబ్బంది, స్పృహ లేకపోవడం, లేదా తగ్గని "
        "నొప్పి — వచ్చే సందర్శన కోసం ఎదురు చూడకుండా 108 కి ఫోన్ చేసి అంబులెన్స్ పిలవండి."
    ),
    "gu": (
        "હું સમજું છું કે તમને અત્યારે આની જરૂર છે. પરંતુ કોઈ દવાની ચોક્કસ માત્રા કે સમયપત્રક "
        "હું કહી શકતો નથી — તે નક્કી કરવા માટે દર્દીને જોનાર ડૉક્ટરની જરૂર છે.\n\n"
        "હું આ વાતચીતની તૈયારીમાં મદદ કરી શકું છું:\n"
        "• લક્ષણો, તે ક્યારે શરૂ થયાં, અને અત્યાર સુધી શું આપવામાં આવ્યું છે — લખી રાખો\n"
        "• દવાનાં પેકેટ અથવા પ્રિસ્ક્રિપ્શન સાથે લઈ જાઓ\n"
        "• આગામી મુલાકાતમાં ડૉક્ટર સમક્ષ આ વાત મૂકો — પેલિએટિવ કેર ડૉક્ટર, પ્રાથમિક આરોગ્ય "
        "કેન્દ્ર, અથવા જ્યાં તમારી નોંધણી છે તે OPD\n\n"
        "જો દર્દીને અત્યારે ખૂબ તકલીફ હોય — શ્વાસ લેવામાં મુશ્કેલી, બેભાન અવસ્થા, અથવા કાબૂમાં "
        "ન આવતો દુખાવો — તો આગામી મુલાકાતની રાહ ન જુઓ, 108 પર ઍમ્બ્યુલન્સ બોલાવો."
    ),
}

# The copy for a language outside the eleven, or for one with no translation.
# ADR 0004 allows the English fallback; the safety record says it was used.
_FALLBACK_LANGUAGE = "en"
# Recorded as the fallback when the caller passed no language at all.
_UNSPECIFIED_LANGUAGE = "unspecified"


def _copy_language(language: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Resolve `language` to the code of the Deferral copy the guard will use.

    Returns ``(code, fallback)``. ``code`` is a Supported Language code that has
    both a Deferral and a Redacted Answer note. ``fallback`` is None when that
    copy is in the language asked for; otherwise it is the language that was
    asked for (or "unspecified"), and ``code`` is English. Every spelling the
    Supported Language table accepts ("or", "od", "or-IN", "hi_IN") resolves.
    """
    lang = get_language(language)
    if lang is not None and lang.code == _FALLBACK_LANGUAGE:
        return _FALLBACK_LANGUAGE, None
    if lang is not None and lang.code in _REFUSAL_LOCALIZED and lang.code in _REDACTION_NOTE:
        return lang.code, None
    return _FALLBACK_LANGUAGE, (language or _UNSPECIFIED_LANGUAGE)


def _refusal_for(language: str) -> str:
    """Return the Deferral copy for `language`, falling back to English."""
    code, _ = _copy_language(language)
    return _REFUSAL_LOCALIZED.get(code, _REFUSAL_EN)


def _redaction_note(language: str) -> str:
    """Short explanation appended after a partial redaction, in `language`."""
    code, _ = _copy_language(language)
    return _REDACTION_NOTE[code]


def _is_critical_alert(alert: Optional[Any]) -> bool:
    """
    Only a CRITICAL alert may override the dosage restriction.

    The emergency detector also fires at HIGH severity, and "severe pain" is one
    of its HIGH keywords. Allowing HIGH to disable the guard would let any dose
    question phrased as severe pain bypass the restriction entirely, which is a
    far more common phrasing than a genuine life-threatening emergency.
    """
    if alert is None:
        return False
    level = getattr(alert, "level", None)
    level_value = getattr(level, "value", level)
    if isinstance(level_value, str) and level_value.lower() in (
        "critical",
        "life_threatening",
    ):
        return True
    return bool(getattr(alert, "contact_emergency_services", False))




# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class EmergencyDetector(Protocol):
    """
    The slice of the emergency detector that DosageGuard depends on.

    Declared structurally rather than imported: safety_enhancements imports this
    module, so a direct import would be circular. Typing it as bare `object` and
    then calling a method on it would defeat the type checker, and the point of
    this module is that its dependency ordering is correct.
    """

    def detect_emergency(
        self, query: str, user_id: Optional[str] = ..., language: str = ...
    ) -> Optional[Any]: ...

# Explicit instructions to summon urgent help. Used only as a fallback when no
# emergency detector is injected, and matched tightly on purpose: requiring the
# imperative plus the vehicle/number prevents ordinary prose such as "in an
# emergency seek help" from switching the guard off.
_EMERGENCY_INSTRUCTION_RE = re.compile(
    r"(?:call|dial|phone|ring)\s*(?:\s*an?)?\s*"
    r"(?:108|102|ambulance|emergency\s+(?:service|services|number))"
    r"|\bambulance\s+(?:now|immediately)\b"
    r"|\bemergency\s+(?:services|help)\s+(?:now|immediately)\b",
    re.I,
)


@dataclass
class DosageGuardResult:
    """Outcome of applying the guard to a single response."""
    blocked: bool
    response: str
    findings: List[DoseFinding] = field(default_factory=list)
    reason: Optional[str] = None
    emergency_overrode: bool = False
    redacted: bool = False
    removed_units: int = 0
    # The language asked for, when the Deferral or Redacted Answer note had to
    # be served in English because no copy exists for it. None otherwise.
    language_fallback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "findings": [f.to_dict() for f in self.findings],
            "emergency_overrode": self.emergency_overrode,
            "redacted": self.redacted,
            "removed_units": self.removed_units,
            "language_fallback": self.language_fallback,
        }


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Shortest remainder worth keeping. Roughly one and a half sentences: below this
# the survivor is an orphaned clause and a plain deferral reads better. Set by
# measurement, not taste - two clean sentences of real clinical content run to
# roughly 110 characters, which a higher threshold was discarding.
_MIN_REMAINING_CHARS = 80

# Appended when specific dosing was removed from an otherwise useful answer, so
# the user is not left wondering what the silence means.
_REDACTION_NOTE = {
    "en": "\n\n_(The specific amount and schedule of any medicine are not included here. Take that part of your question to your next consultation.)_",
    "hi": "\n\n_(किसी भी दवा की खास मात्रा और समय-सारणी यहाँ नहीं दी गई है। यह सवाल अपनी अगली डॉक्टर की विज़िट में पूछें।)_",
    "bn": "\n\n_(ওষুধের নির্দিষ্ট পরিমাণ ও সময়সূচি এখানে দেওয়া হয়নি। এই প্রশ্নটি আপনার পরবর্তী ডাক্তারের সাক্ষাতে করুন।)_",
    "ta": "\n\n_(எந்த மருந்தின் குறிப்பிட்ட அளவு மற்றும் நேரம் இங்கே தரப்படவில்லை. இந்தக் கேள்வியை அடுத்த மருத்துவர் சந்திப்பில் கேளுங்கள்.)_",
    # Added in round 3 (ticket 04); machine-drafted, awaiting clinical language review.
    "mr": "\n\n_(कोणत्याही औषधाचे नेमके प्रमाण आणि वेळापत्रक येथे दिलेले नाही. हा प्रश्न तुमच्या पुढील डॉक्टर भेटीत विचारा.)_",
    "kn": "\n\n_(ಯಾವುದೇ ಔಷಧಿಯ ನಿರ್ದಿಷ್ಟ ಪ್ರಮಾಣ ಮತ್ತು ವೇಳಾಪಟ್ಟಿಯನ್ನು ಇಲ್ಲಿ ನೀಡಿಲ್ಲ. ಈ ಪ್ರಶ್ನೆಯನ್ನು ನಿಮ್ಮ ಮುಂದಿನ ವೈದ್ಯರ ಭೇಟಿಯಲ್ಲಿ ಕೇಳಿ.)_",
    "ml": "\n\n_(ഒരു മരുന്നിന്റെയും കൃത്യമായ അളവും സമയക്രമവും ഇവിടെ നൽകിയിട്ടില്ല. ഈ ചോദ്യം നിങ്ങളുടെ അടുത്ത ഡോക്ടർ സന്ദർശനത്തിൽ ചോദിക്കുക.)_",
    "or": "\n\n_(କୌଣସି ଔଷଧର ନିର୍ଦ୍ଦିଷ୍ଟ ପରିମାଣ ଓ ସମୟସୂଚୀ ଏଠାରେ ଦିଆଯାଇନାହିଁ। ଏହି ପ୍ରଶ୍ନ ଆପଣଙ୍କ ପରବର୍ତ୍ତୀ ଡାକ୍ତର ସାକ୍ଷାତରେ ପଚାରନ୍ତୁ।)_",
    "pa": "\n\n_(ਕਿਸੇ ਵੀ ਦਵਾਈ ਦੀ ਖ਼ਾਸ ਮਾਤਰਾ ਅਤੇ ਸਮਾਂ-ਸਾਰਣੀ ਇੱਥੇ ਨਹੀਂ ਦਿੱਤੀ ਗਈ। ਇਹ ਸਵਾਲ ਆਪਣੀ ਅਗਲੀ ਡਾਕਟਰੀ ਮੁਲਾਕਾਤ ਵਿੱਚ ਪੁੱਛੋ।)_",
    "te": "\n\n_(ఏ మందు యొక్క నిర్దిష్ట పరిమాణం మరియు సమయపట్టిక ఇక్కడ ఇవ్వబడలేదు. ఈ ప్రశ్నను మీ తదుపరి వైద్య సందర్శనలో అడగండి.)_",
    "gu": "\n\n_(કોઈ પણ દવાની ચોક્કસ માત્રા અને સમયપત્રક અહીં આપ્યાં નથી. આ પ્રશ્ન તમારી આગામી ડૉક્ટરની મુલાકાતમાં પૂછો.)_",
}

# A trailing colon marks a heading whose contents may all be removed.
_HEADING_RE = re.compile(r"^[\s#*\-–—]*[^\n:]{0,60}:\s*$")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।!?])\s+")


def _split_units(text: str) -> List[str]:
    """
    Split a response into independently removable units.

    Every line is sentence-split, so that redacting one clause does not discard
    the rest of the line. A list marker ("1.", "-", "\u2022") is re-attached to
    the first sentence of its line so that removing a numbered item does not
    leave an orphaned "1." in the output.
    """
    units: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            units.append("")
            continue
        marker_match = re.match(r"^((?:[\-\u2022\*]|\d+[.)])\s+)", stripped)
        marker = marker_match.group(1) if marker_match else ""
        body = stripped[len(marker):] if marker else stripped
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
        if not sentences:
            units.append(stripped)
            continue
        units.append(marker + sentences[0].strip())
        units.extend(sent.strip() for sent in sentences[1:])
    return units


def _drop_orphan_headings(units: List[str]) -> List[str]:
    """
    Remove a heading whose entire block beneath it was redacted.

    A heading is orphaned when the next non-blank unit is either another heading
    or nothing at all, which means everything the heading introduced is gone.
    """
    out = list(units)
    i = 0
    while i < len(out):
        if out[i] and _HEADING_RE.match(out[i]):
            j = i + 1
            while j < len(out) and out[j] == "":
                j += 1
            block_gone = j >= len(out) or (
                bool(out[j]) and _HEADING_RE.match(out[j])
            )
            if block_gone:
                del out[i:j]
                continue
        i += 1
    return out


def redact_doses(response: str) -> Tuple[str, List[DoseFinding], int]:
    """
    Remove dose-bearing units from `response`, keeping the remainder.

    Returns the redacted text, the findings that triggered redaction, and the
    number of units removed.
    """
    units = _split_units(response)
    kept: List[str] = []
    findings: List[DoseFinding] = []
    removed = 0

    for unit in units:
        if not unit:
            kept.append(unit)
            continue
        unit_findings = detect_doses(unit)
        if unit_findings:
            findings.extend(unit_findings)
            removed += 1
            continue
        kept.append(unit)

    kept = _drop_orphan_headings(kept)

    # Collapse runs of blank lines left behind by removal.
    collapsed: List[str] = []
    for unit in kept:
        if not unit and collapsed and not collapsed[-1]:
            continue
        collapsed.append(unit)

    text = "\n".join(collapsed).strip()
    return text, _dedupe(findings), removed


class DosageGuard:
    """
    Applies the dosage restriction to generated responses.

    The guard is deliberately a *post-generation* filter rather than a prompt
    instruction.  Protocol SS7.7 states that "a prompt instruction or a successful
    connection test alone does not satisfy these checks", so the restriction is
    enforced on the response text itself and is independently testable.
    """

    def __init__(self, emergency_system: Optional[EmergencyDetector] = None):
        # Injected rather than imported to avoid a circular dependency on
        # safety_enhancements, which imports this module.
        self.emergency_system = emergency_system

    def apply(
        self,
        response: str,
        language: str = "en",
        query: str = "",
    ) -> DosageGuardResult:
        """
        Filter `response`, returning either the original text or the deferral.

        If the query is an emergency trigger the response is passed through
        untouched and `emergency_overrode` is set, because suppressing an urgent
        instruction in order to enforce a style restriction is the worse failure.
        """
        if self._is_emergency(query, response):
            return DosageGuardResult(
                blocked=False,
                response=response,
                emergency_overrode=True,
            )

        findings = detect_doses(response)
        if not findings:
            return DosageGuardResult(blocked=False, response=response)

        logger.warning(
            "Dosage restriction applied (reason=%s, findings=%d, lang=%s)",
            findings[0].reason,
            len(findings),
            language,
        )
        _, language_fallback = _copy_language(language)
        if language_fallback is not None:
            logger.warning(
                "No Deferral copy for language %r; serving English", language_fallback
            )

        # Attempt sentence-level redaction first, so that a tier-1 question is not
        # answered with a tier-2 refusal when only part of the answer was dosing.
        redacted, redacted_findings, removed = redact_doses(response)
        if redacted and len(redacted) >= _MIN_REMAINING_CHARS:
            # Safety net: re-run the detector over the redaction. Redaction must
            # never be the thing that lets a dose through, so if anything still
            # matches we fall back to refusing the whole answer.
            residue = detect_doses(redacted)
            if not residue:
                return DosageGuardResult(
                    blocked=True,
                    response=redacted + _redaction_note(language),
                    findings=_dedupe(findings),
                    reason=findings[0].reason,
                    redacted=True,
                    removed_units=removed,
                    language_fallback=language_fallback,
                )
            logger.warning(
                "Redaction left %d dose match(es); escalating to full refusal",
                len(residue),
            )

        return DosageGuardResult(
            blocked=True,
            response=_refusal_for(language),
            findings=_dedupe(findings),
            reason=findings[0].reason,
            removed_units=removed,
            language_fallback=language_fallback,
        )

    def _is_emergency(self, query: str, response: str) -> bool:
        """
        Decide whether the emergency path takes precedence over the deferral.

        The detector is authoritative. The response-text fallback is deliberately
        narrow: it matches an explicit instruction to summon help, never the bare
        word "emergency". A loose match here would silently disable the entire
        dosage restriction on any answer that merely mentions emergencies, which
        is a failure mode that must not exist in a safety control.
        """
        if self.emergency_system is not None:
            try:
                alert = self.emergency_system.detect_emergency(
                    query, "dosage-guard", "en"
                )
            except Exception:  # pragma: no cover - must never break a response
                logger.warning(
                    "Emergency detector failed inside DosageGuard; "
                    "falling back to response-text check",
                    exc_info=True,
                )
            else:
                if _is_critical_alert(alert):
                    return True

        return bool(_EMERGENCY_INSTRUCTION_RE.search(response or ""))


_dosage_guard: Optional[DosageGuard] = None


def get_dosage_guard(emergency_system: Optional[EmergencyDetector] = None) -> DosageGuard:
    """Get or create the DosageGuard singleton."""
    global _dosage_guard
    if _dosage_guard is None:
        _dosage_guard = DosageGuard(emergency_system=emergency_system)
    return _dosage_guard