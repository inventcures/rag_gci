#!/usr/bin/env python3
"""
Cross-language retrieval evaluation
==================================
Settles ADR 0003's open question: which multilingual embedding model retrieves
best on the *actual* Palli Sahayak corpus, in the eleven languages the app
actually supports.

Why this needs measuring rather than assuming
---------------------------------------------
bge-m3's model card claims "more than 100 working languages", but the retrieval
benchmark its authors released with it (MLDR) covers 13 languages of which only
Hindi is Indic. LaBSE is explicitly tuned across 22 Indian languages, so
Kannada, Tamil, Bengali, Malayalam, Marathi, Gujarati, Punjabi and Odia are its
home turf rather than a tail of a 100-language average. Neither vendor number
speaks to dyspnoea in Malayalam against this 263-chunk corpus.

Three models are compared:
  * bge-m3        - 1024-dim, XLM-RoBERTa large, 100+ languages
  * LaBSE         - 471M params, tuned on 22 Indian languages
  * bge-small-en-v1.5 - the model the live index was actually built with
                      (384-dim, English-only). Included as the honest baseline:
                      if the current English-only index already retrieves well
                      for English queries, that is the bar the others must beat.

Ground truth is anchored to chunk indices, not to topics, so a model cannot
score well by retrieving a thematically adjacent chunk.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CHROMA_DIR = "./data/chroma_db"
COLLECTION = "documents"

MODELS = {
    "bge-m3": {
        "path": "/tmp/emb_eval/models/bge-m3",
        "dim": 1024,
        "note": "1024-dim multilingual, XLM-RoBERTa-large backbone",
    },
    "labse": {
        "path": "/tmp/emb_eval/models/labse",
        "dim": 471,
        "note": "471M params, explicitly tuned on 22 Indian languages",
    },
    "bge-small-en-v1.5": {
        "path": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "note": "current live index; English-only, the honest baseline",
    },
}

# Languages the app supports (ADR 0002, bounded by TTS).
LANGUAGES = [
    "en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN",
    "ml-IN", "mr-IN", "gu-IN", "pa-IN", "od-IN",
]

# (question in each language, gold chunk indices in the corpus)
# Gold is anchored to specific chunks verified against the corpus content.
SEED_PROBES: List[dict] = [
    {
        "id": "laxative",
        "gold": [63, 64, 71],
        "en": "What must always be given with morphine?",
        "translations": {
            "hi-IN": "मॉर्फीन के साथ क्या हमेशा देना चाहिए?",
            "bn-IN": "মরফিনের সঙ্গে সবসময় কী দেওয়া উচিত?",
            "ta-IN": "மார்பினுடன் எப்போதும் என்ன கொடுக்க வேண்டும்?",
            "te-IN": "మార్ఫిన్‌తో ఎప్పుడూ ఏమి ఇవ్వాలి?",
            "kn-IN": "ಮಾರ್ಫಿನ್ ಜೊತೆಯೇನಾದರೂ ಏನು ಕೊಡಬೇಕು?",
            "ml-IN": "മോർഫിനോടൊപ്പം എപ്പോഴും എന്ത് കൊടുക്കണം?",
            "mr-IN": "मॉर्फिनसोबत नेहमी काय द्यावे?",
            "gu-IN": "મોર્ફિન સાથે હંમેશા શું આપવું?",
            "pa-IN": "ਮੌਰਫੀਨ ਨਾਲ ਹਮੇਸ਼ਾ ਕੀ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ?",
            "od-IN": "ମର୍ଫିନ ସହିତ ସର୍ବଦେ କ'ଣ ଦେବା?",
        },
    },
    {
        "id": "pursed_lip",
        "gold": [67, 68],
        "en": "How can breathlessness be helped without medicine?",
        "translations": {
            "hi-IN": "बिना दवा के सांस फूलना कैसे कम करें?",
            "bn-IN": "ঔষধ ছাড়া শ্বাসকষ্ট কীভাবে কমানো যায়?",
            "ta-IN": "மருந்து இல்லாமல் மூச்சுத் திணறலை எப்படி குறைக்கலாம்?",
            "te-IN": "మందులు లేకుండా ఊపిరితిత్తులను ఎలా తగ్గిస్తారు?",
            "kn-IN": "ಔಷಧವಿಲ್ಲದೆ ಉಸಿರಕಟ್ಟಿಕೆ ಹೇಗೆ ಕಡಿಮೆ ಮಾಡುವುದು?",
            "ml-IN": "മരുന്നുകടയാതെ ശ്വാസം കുറയ്ക്ക് എങ്ങനെ?",
            "mr-IN": "औषधाशिवाय श्वासकुश कसा कमी करावा?",
            "gu-IN": "દવા વગર શ્વાસ કેવી રીતે ઘટાડવું?",
            "pa-IN": "ਬਿਨਾਂ ਦਵਾ ਸਾਹ ਕਿਵੇਂ ਘੱਟ ਕਰੀਏ?",
            "od-IN": "ଔଷଧ ବିନା ଶ୍ୱାସକଷ୍ଟ କିପରି କମାଇବା?",
        },
    },
    {
        "id": "pressure_sores",
        "gold": [87, 88, 181],
        "en": "What causes a pressure sore and how is it prevented?",
        "translations": {
            "hi-IN": "दबाव से घाव क्यों होता है और इससे कैसे बचें?",
            "bn-IN": "চাপে ঘা হয় কীভাবে এবং তা থেকে কীভাবে বাঁচবেন?",
            "ta-IN": "அழுத்தத்தால் புணர் ஏன் ஏற்படுகிறது, அதை எப்படி தவிர்ப்பது?",
            "te-IN": "ఒత్తిడి వల్ల గాయం ఎందుకు ఏర్పడుతుంది, దాని నుంచి ఎలా బాచుకోవాలి?",
            "kn-IN": "ಒತ್ತಡದಿಂದ ಗಾಯ ಏಕೆ ಆಗುತ್ತದೆ ಮತ್ತು ಹೇಗೆ ತಪ್ಪಿಸುವುದು?",
            "ml-IN": "സമ്മർച്ചം മൂലം വടവും എന്തുകൊണ്ട് ഉണ്ടാകുകം, എങ്ങനെ ഒഴിവാക്കണം?",
            "mr-IN": "दाबामुळे जखम का होते आणि ते कसे टाळावे?",
            "gu-IN": "દબાણથી ઘાવ કેમ થાય છે અને તે કેવી રીતે ટાળવું?",
            "pa-IN": "ਦਬਾਵ ਨਾਲ ਜ਼ਖਮ ਕਿਉਂ ਬਣਦਾ ਹੈ ਅਤੇ ਇਸ ਤੋਂ ਕਿਵੇਂ ਬਚਣਾ?",
            "od-IN": "ଚାପ ଫଳରେ ଘା କାହିଁକି ହୁଏ ଏବଂ ତାରୁରୁ କିପରି ବଞ୍ଚିବ?",
        },
    },
    {
        "id": "terminal_phase",
        "gold": [125, 126],
        "en": "How do you recognise the terminal phase of illness?",
        "translations": {
            "hi-IN": "बीमारी की अंतिम अवस्था कैसे पहचानें?",
            "bn-IN": "রোগের শেষ পর্যায় কীভাবে চিনবেন?",
            "ta-IN": "நோயின் இறுதி நிலையை எப்படி அறியலாம்?",
            "te-IN": "వ్యాధి చివరి దశను ఎలా గుర్తిస్తారు?",
            "kn-IN": "ಕಾಯಿಲೆಯ ಕೊನೆಯ ಹಂತ ಹೇಗೆ ಗುರುತಿಸುತ್ತಾರೆ?",
            "ml-IN": "രോഗത്തിന്റെ അവസാന ഘട്ടം എങ്ങനെ തിരിച്ചറിയണം?",
            "mr-IN": "आजाराचा अंतिम टप्पा कसा ओळखावा?",
            "gu-IN": "રોગનો અંતિમ તબક્કો કેવી રીતે ઓળખવો?",
            "pa-IN": "ਬਿਮਾਰੀ ਦਾ ਅੰਤਮ ਪੜਾਅ ਕਿਵੇਂ ਪਛਾਣੀਏ?",
            "od-IN": "ରୋଗର ଶେଷ ସ୍ତର କିପରି ଚିହ୍ନିବ?",
        },
    },
    {
        "id": "colostomy",
        "gold": [103, 104, 105],
        "en": "How should a colostomy bag be cared for?",
        "translations": {
            "hi-IN": "कोलोस्टॉमी बैग की देखभाल कैसे करें?",
            "bn-IN": "কোলোস্টমি ব্যাগের যত্ন কীভাবে করবেন?",
            "ta-IN": "கோலோஸ்டோமி பைட்டை எப்படி பராமரிக்கலாம்?",
            "te-IN": "కోలోస్టమీ బ్యాగ్‌ను ఎలా సంరక్షించాలి?",
            "kn-IN": "ಕೊಲೊಸ್ಟಮಿ ಚೀಲದ ಆರೈಕೆ ಹೇಗೆ ಮಾಡುವುದು?",
            "ml-IN": "കൊലോസ്റ്റമി ബാഗ് എങ്ങനെ പരിചരിക്കണം?",
            "mr-IN": "कोलोस्टॉमी पिशवाची काळजी कसी घ्यावी?",
            "gu-IN": "કોલોસ્ટોમી બેગની સંભાળ કેવી રીતે રાખવી?",
            "pa-IN": "ਕੋਲੋਸਟੋਮੀ ਥੈਲੀ ਦੀ ਸੰਭਾਲ ਕਿਵੇਂ ਕਰਨੀ ਹੈ?",
            "od-IN": "କୋଲୋଷ୍ଟୋମୀ ବ୍ୟାଗ ରକ୍ଷଣାବେଳା କିପରି କରିବ?",
        },
    },
    {
        "id": "asha_role",
        "gold": [46, 48, 52],
        "en": "What is the role of an ASHA worker in palliative care?",
        "translations": {
            "hi-IN": "पैलिएटिव केयर में आशा कार्यकर्ता की क्या भूमिका है?",
            "bn-IN": "প্যালিয়েটিভ কেয়ারে আশা কর্মীর ভূমিকা কী?",
            "ta-IN": "பலியேட்டிவ் கேரில் ஆஷா தொழிலாளியின் பங்கு என்ன?",
            "te-IN": "పల్లియేటివ్ కేర్‌లో ఆశా కార్మికుడి పాత్ర ఏమిటి?",
            "kn-IN": "ಪ್ಯಾಲಿಯೇಟಿವ್ ಆರೈಕೆಯಲ್ಲಿ ಆಶಾ ಕಾರ್ಮಿಕರ ಪಾತ್ರ ಏನು?",
            "ml-IN": "പാലിയേറ്റീവ് കെയർയിലെ ആശാ പ്രവർത്തിന്റെ പങ്കാളമെന്ത്?",
            "mr-IN": "पॅलिएटिव्ह केअरमध्ये आशा कर्म्याऱ्याची भूमिका काय आहे?",
            "gu-IN": "પેલિયેટિવ કેરમાં આશા કર્મચારીની ભૂમિકા શું છે?",
            "pa-IN": "ਪੈਲੀਏਟਿਵ ਕੇਅਰ ਵਿੱਚ ਆਸ਼ਾ ਕਰਮਚਾਰੀ ਦੀ ਭੂਮਿਕਾ ਕੀ ਹੈ?",
            "od-IN": "ପାଲିଏଟିଭ୍ କେଆରରେ ଆଶା କର୍ମଚାରୀର ଭୂମିକା କ'ଣ?",
        },
    },
    {
        "id": "constipation",
        "gold": [70, 71, 221],
        "en": "Why does a patient on opioids get constipation?",
        "translations": {
            "hi-IN": "ओपिओइड लेने वाले रोगी को कब्ज क्यों होता है?",
            "bn-IN": "ওপিওয়েড নেওয়া রোগীর কোষ্ঠকাঠিন্য কেন হয়?",
            "ta-IN": "ஓபியாய்டு எடுக்கும் நோயாளருக்கு மலக்குறி ஏன் ஏற்படுகிறது?",
            "te-IN": "ఓపియాయిడ్లు తీసుకునే రోగికి మలదూపం ఎందుకు వస్తుంది?",
            "kn-IN": "ಒಪಿಯಾಯ್ಡ್ ತೆಗೆದುಕೊಳ್ಳುವ ರೋಗಿಗೆ ಮಲಬಂಧ ಏಕೆ?",
            "ml-IN": "ഒപ്പിയോയിഡ് എടുക്കുന്ന രോഗിക്ക് മലക്കറി എന്തുകൊണ്ട്?",
            "mr-IN": "ओपिऑइड घेणाऱ्या रुग्णाला कब्ज का होते?",
            "gu-IN": "ઓપિયોઇડ લઈ રહેલા દર્દીને કબજી કેમ થાય છે?",
            "pa-IN": "ਓਪੀਔਇਡ ਲੈਣ ਵਾਲੇ ਮਰੀਜ਼ ਨੂੰ ਕਬਜ਼ੀ ਕਿਉਂ ਹੁੰਦੀ ਹੈ?",
            "od-IN": "ଓପିୟାଇଡ୍ ନେଉଥିବା ରୋଗୀକୁ କଷ୍ଟକରୋଗ କାହିଁକି?",
        },
    },
    {
        "id": "oral_care",
        "gold": [76, 77],
        "en": "How should the mouth of an unconscious patient be cared for?",
        "translations": {
            "hi-IN": "बेहोश रोगी के मुंह की देखभाल कैसे करें?",
            "bn-IN": "অচেতন রোগীর মুখের যত্ন কীভাবে করবেন?",
            "ta-IN": "உணர்வில்லா நோயாளரின் வாயை எப்படி பராமரிக்கலாம்?",
            "te-IN": "స్పృహను లేని రోగికి నోటి శుచ్ధి ఎలా చేయాలి?",
            "kn-IN": "ಸ್ಪೃಹ ಇಲ್ಲದ ರೋಗಿಯ ಬಾಯಿಯ ಆರೈಕೆ ಹೇಗೆ ಮಾಡುವುದು?",
            "ml-IN": "അറിയാത്ത രോഗിയുടെ വായിന്റെ പരിചരണം എങ്ങനെ?",
            "mr-IN": "बेहोश रुग्णाच्या तोंडाची काळजी कसी घ्यावी?",
            "gu-IN": "બેહોશ દર્દીના મોંની સંભાળ કેવી રીતે રાખવી?",
            "pa-IN": "ਬੇਹੋਸ਼ ਮਰੀਜ਼ ਦੇ ਮੂੰਹ ਦੀ ਸੰਭਾਲ ਕਿਵੇਂ ਕਰਨੀ ਹੈ?",
            "od-IN": "ବେହୋଶ ରୋଗୀର ମୁଖ ଯତ୍ନ କିପରି କରିବ?",
        },
    },
    {
        "id": "confirming_death",
        "gold": [132, 133],
        "en": "What should be done when a patient dies at home?",
        "translations": {
            "hi-IN": "रोगी के घर पर मरने पर क्या करना चाहिए?",
            "bn-IN": "রোগী বাড়িতে মারা গেলে কী করতে হবে?",
            "ta-IN": "நோயாளர் வீட்டில் இறந்தால் என்ன செய்வது?",
            "te-IN": "రోగి ఇంట్లో చనిపిస్తే ఏమి చేయాలి?",
            "kn-IN": "ರೋಗಿ ಮನೆಯಲ್ಲಿ ಮರಿದರೆ ಏನು ಮಾಡಬೇಕು?",
            "ml-IN": "രോഗി വീട്ടിൽ മരിച്ചാൽ എന്ത് ചെയ്യണം?",
            "mr-IN": "रुग्ण घरात मरणावर काय करावे?",
            "gu-IN": "દર્દી ઘરે મરે તો શું કરવું?",
            "pa-IN": "ਮਰੀਜ਼ ਘਰ 'ਤੇ ਮਰ ਜਾਵੇ ਤਾਂ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?",
            "od-IN": "ରୋଗୀ ଘରରେ ମରିଗଲେ କ'ଣ କରିବ?",
        },
    },
]


def load_corpus() -> Tuple[List[str], List[dict]]:
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    docs = [
        re.sub(r"\s+", " ", (d or "").replace("\t", " ")).strip()
        for d in data["documents"]
    ]
    return docs, data["metadatas"]


def build_queries() -> List[dict]:
    out = []
    for probe in SEED_PROBES:
        for lang in LANGUAGES:
            text = probe["en"] if lang == "en-IN" else probe["translations"].get(lang)
            if text:
                out.append({
                    "probe": probe["id"],
                    "language": lang,
                    "text": text,
                    "gold": set(probe["gold"]),
                })
    return out


def evaluate(model_name: str, corpus: List[str], queries: List[dict], top_k: int = 5) -> dict:
    from sentence_transformers import SentenceTransformer

    spec = MODELS[model_name]
    t0 = time.time()
    model = SentenceTransformer(spec["path"], device="cpu")
    load_s = time.time() - t0

    t0 = time.time()
    doc_emb = model.encode(corpus, normalize_embeddings=True, batch_size=32,
                           show_progress_bar=False)
    index_s = time.time() - t0

    t0 = time.time()
    q_emb = model.encode([q["text"] for q in queries], normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False)
    query_s = (time.time() - t0) / max(len(queries), 1)

    sims = q_emb @ doc_emb.T

    per_language: Dict[str, Dict[str, float]] = {}
    overall_hits = {k: 0 for k in (1, 3, 5, 10)}
    mrr_sum = 0.0
    total = 0

    for i, q in enumerate(queries):
        order = np.argsort(-sims[i])
        ranked = [int(j) for j in order]
        gold = q["gold"]
        total += 1

        for k in overall_hits:
            if gold & set(ranked[:k]):
                overall_hits[k] += 1

        rr = 0.0
        for rank, idx in enumerate(ranked[:10], start=1):
            if idx in gold:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        bucket = per_language.setdefault(
            q["language"], {"hit@1": 0, "hit@5": 0, "mrr": 0.0, "n": 0}
        )
        bucket["n"] += 1
        if gold & set(ranked[:1]):
            bucket["hit@1"] += 1
        if gold & set(ranked[:5]):
            bucket["hit@5"] += 1
        bucket["mrr"] += rr

    for lang, b in per_language.items():
        b["hit@1"] = b["hit@1"] / b["n"]
        b["hit@5"] = b["hit@5"] / b["n"]
        b["mrr"] = b["mrr"] / b["n"]

    del model
    return {
        "model": model_name,
        "dim": spec["dim"],
        "note": spec["note"],
        "overall": {f"hit@{k}": v / total for k, v in overall_hits.items()},
        "mrr@10": mrr_sum / total,
        "per_language": per_language,
        "load_seconds": round(load_s, 1),
        "index_seconds": round(index_s, 1),
        "query_ms": round(query_s * 1000, 1),
        "corpus_size": len(corpus),
        "query_count": total,
    }


def main() -> None:
    corpus, _ = load_corpus()
    queries = build_queries()
    print(f"corpus: {len(corpus)} chunks | queries: {len(queries)} "
          f"({len(SEED_PROBES)} probes x {len(LANGUAGES)} languages)\n")

    results = []
    for name in MODELS:
        print(f"--- {name} ---", flush=True)
        r = evaluate(name, corpus, queries)
        results.append(r)
        print(f"  hit@1 {r['overall']['hit@1']:.3f}  hit@5 {r['overall']['hit@5']:.3f}  "
              f"mrr@10 {r['mrr@10']:.3f}  index {r['index_seconds']}s  "
              f"query {r['query_ms']}ms", flush=True)

    Path("evaluation/retrieval_eval").mkdir(parents=True, exist_ok=True)
    out = Path("evaluation/retrieval_eval/results.json")
    out.write_text(json.dumps({
        "corpus_chunks": len(corpus),
        "probes": len(SEED_PROBES),
        "languages": LANGUAGES,
        "queries": len(queries),
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    print_ranked_table(results)


def print_ranked_table(results: List[dict]) -> None:
    print("\n### Per-language hit@5")
    header = "| Model | " + " | ".join(LANGUAGES) + " | mean |"
    print(header)
    print("|" + "---|" * (len(LANGUAGES) + 2))
    for r in sorted(results, key=lambda x: -x["overall"]["hit@5"]):
        cells = [
            f"{r['per_language'].get(l, {}).get('hit@5', 0):.2f}" for l in LANGUAGES
        ]
        print(f"| {r['model']} | " + " | ".join(cells) + f" | **{r['overall']['hit@5']:.2f}** |")


if __name__ == "__main__":
    main()
