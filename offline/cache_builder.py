import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/cache/mobile")

SUPPORTED_LANGUAGES = ["ta-IN", "te-IN", "kn-IN", "ml-IN", "bn-IN", "as-IN", "hi-IN", "en-IN"]

EMERGENCY_KEYWORDS = {
    "en": ["bleeding", "unconscious", "not breathing", "chest pain", "seizure", "suicide", "severe pain", "choking"],
    "hi": ["खून बह रहा", "बेहोश", "सांस नहीं", "छाती में दर्द", "दौरा", "तेज दर्द"],
    "ta": ["இரத்தப்போக்கு", "மயக்கம்", "மூச்சு விடவில்லை", "நெஞ்சு வலி", "வலிப்பு"],
    "bn": ["রক্তপাত", "অচেতন", "শ্বাস নেই", "বুকে ব্যথা", "খিঁচুনি"],
    "kn": ["ರಕ್ತಸ್ರಾವ", "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದ", "ಉಸಿರಾಟ ಇಲ್ಲ", "ಎದೆ ನೋವು"],
    "ml": ["രക്തസ്രാവം", "ബോധം ഇല്ല", "ശ്വാസം ഇല്ല", "നെഞ്ചുവേദന"],
    "te": ["రక్తస్రావం", "స్పృహ లేదు", "ఊపిరి ఆడటం లేదు", "ఛాతీ నొప్పి"],
    "as": ["ৰক্তক্ষৰণ", "অচেতন", "উশাহ নাই", "বুকুৰ বিষ"],
}

# Replaced by offline.questions, which carries all eleven supported languages.
# The previous single English list meant a Marathi user received a bundle whose
# questions were in English and matched nothing, and one of the twenty was a
# dosage question the Dose Boundary would refuse.
from offline.questions import ENGLISH as DEFAULT_TOP_QUERIES  # noqa: E402,F401
from offline.questions import has_translation, questions_for  # noqa: E402


class CacheBundleBuilder:
    def __init__(self, rag_pipeline=None, kg_client=None, usage_analytics=None):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.rag_pipeline = rag_pipeline
        self.kg_client = kg_client
        self.usage_analytics = usage_analytics

    async def build_bundle(self, language: str) -> dict:
        """Build a complete offline cache bundle for the given language."""
        bundle_version = time.strftime("%Y%m%d-%H%M%S")

        queries = await self._build_query_cache(language)
        treatments = await self._build_treatment_cache()

        bundle = {
            "version": bundle_version,
            "language": language,
            "questions_translated": has_translation(language),
            "question_count": len(queries),
            "generated_at": time.time(),
            "queries": queries,
            "treatments": treatments,
            "emergency_keywords": EMERGENCY_KEYWORDS,
            "evidence_badge_metadata": {
                "A": {"label": "Strong Evidence", "color": "#1B5E20"},
                "B": {"label": "Good Evidence", "color": "#2E7D32"},
                "C": {"label": "Moderate Evidence", "color": "#F57F17"},
                "D": {"label": "Limited Evidence", "color": "#E65100"},
                "E": {"label": "Insufficient Evidence", "color": "#C62828"},
            },
        }

        filepath = CACHE_DIR / f"bundle_{language}_{bundle_version}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        logger.info(f"Built cache bundle for {language}: {len(queries)} queries, {len(treatments)} treatments")
        return bundle

    async def _build_query_cache(self, language: str) -> List[dict]:
        """Pre-compute responses for the twenty anticipated questions."""
        queries = list(questions_for(language))
        if not has_translation(language):
            logger.warning(
                "No translated question set for %s; bundle will be in English and "
                "will not match a non-English query. Recorded so this is visible "
                "rather than mistaken for a translation failure.", language,
            )

        if self.usage_analytics:
            try:
                top = await self.usage_analytics.get_top_queries(n=20, language=language)
                if top:
                    queries = top
            except Exception:
                logger.exception("Usage analytics unavailable; using the curated set")

        cached_queries = []
        for query_text in queries:
            try:
                if self.rag_pipeline:
                    result = await self.rag_pipeline.query(
                        query_text=query_text,
                        language=language,
                    )
                    cached_queries.append({
                        "query_hash": hashlib.sha256(query_text.lower().strip().encode()).hexdigest(),
                        "query_text": query_text,
                        "query_language": language,
                        "response_text": result.answer,
                        "evidence_level": result.evidence_level if hasattr(result, "evidence_level") else "C",
                        "sources": result.sources if hasattr(result, "sources") else [],
                    })
            except Exception as e:
                logger.warning(f"Failed to cache query '{query_text}': {e}")

        return cached_queries

    async def _build_treatment_cache(self) -> List[dict]:
        """Cache top 50 symptom-treatment pairs from knowledge graph."""
        common_symptoms = [
            "pain", "nausea", "vomiting", "breathlessness", "constipation",
            "anxiety", "depression", "insomnia", "fatigue", "appetite_loss",
            "mouth_sores", "fever", "cough", "diarrhea", "edema",
            "itching", "confusion", "delirium", "hiccups", "bleeding",
        ]

        treatments = []
        if self.kg_client:
            for symptom in common_symptoms:
                try:
                    result = await self.kg_client.get_treatments(symptom)
                    if result:
                        treatments.append({
                            "symptom": symptom,
                            "treatments": result,
                        })
                except Exception:
                    pass

        return treatments
