import asyncio
import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from participants import system

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


def _normalise_for_hash(question_text: str) -> str:
    """Reduce a question to its lookup key.

    Lowercase, trim, and drop a trailing question mark or exclamation point. The
    Android client applies the same normalisation in OfflineAnswerResolver, because
    the two cannot share code and a difference between them turns every offline
    lookup into a silent miss. People type "How do I manage pain?"; the bundle holds
    "how to manage pain at home" with no punctuation.
    """
    text = question_text.lower().strip()
    for suffix in ("?", "!"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text

# language -> (bundle key, bundle). One process-wide cache.
_BUNDLES: Dict[str, Tuple[str, dict]] = {}
_build_locks: Dict[str, asyncio.Lock] = {}


def _read_bundle(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_bundle(language: str, key: str, bundle: dict) -> None:
    """Write atomically, then remove this language's older bundles."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"bundle_{language}_{key}.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)
    os.replace(tmp, path)
    for old in CACHE_DIR.glob(f"bundle_{language}_*.json"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                logger.warning("Could not prune old bundle %s", old.name)


class CacheBundleBuilder:
    def __init__(self, rag_pipeline=None, kg_client=None, usage_analytics=None,
                 safety_manager=None):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.rag_pipeline = rag_pipeline
        # Cached answers are shown offline with no server in the loop, so they
        # must pass the same dose boundary as a live answer before they are
        # stored. None falls back to the process-wide manager.
        self.safety_manager = safety_manager
        self.kg_client = kg_client
        self.usage_analytics = usage_analytics

    async def get_bundle(self, language: str) -> Tuple[dict, str]:
        """The bundle for `language` and its ETag, built at most once per
        language, release, corpus and question set.

        Building runs twenty full answers; it used to run on every download.
        Bundles are kept in memory and on disk (so a restart does not rebuild),
        and concurrent requests for the same language share one build.
        """
        questions = await self._questions(language)
        key = self._bundle_key(language, questions)
        cached = _BUNDLES.get(language)
        if cached and cached[0] == key:
            return cached[1], key

        async with _build_locks.setdefault(language, asyncio.Lock()):
            cached = _BUNDLES.get(language)
            if cached and cached[0] == key:
                return cached[1], key
            path = CACHE_DIR / f"bundle_{language}_{key}.json"
            bundle = await asyncio.to_thread(_read_bundle, path)
            if bundle is None:
                bundle = await self.build_bundle(language, questions=questions, key=key)
            _BUNDLES[language] = (key, bundle)
            return bundle, key

    def _bundle_key(self, language: str, questions: List[str]) -> str:
        """What the bundle's content depends on: release, corpus, questions."""
        from study_release import get_study_release

        parts = {
            "language": language,
            "release": get_study_release().release_id,
            "corpus": self._corpus_version(),
            "questions": questions,
        }
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _corpus_version(self) -> str:
        """The indexed documents and their chunk counts; changes when the
        corpus is re-indexed, added to or pruned."""
        metadata = getattr(self.rag_pipeline, "document_metadata", None) or {}
        summary = sorted((str(doc_id), (info or {}).get("chunk_count")) for doc_id, info in metadata.items())
        return hashlib.sha256(json.dumps(summary).encode()).hexdigest()[:16]

    async def _questions(self, language: str) -> List[str]:
        queries = list(questions_for(language))
        if self.usage_analytics:
            try:
                top = await self.usage_analytics.get_top_queries(n=20, language=language)
                if top:
                    queries = top
            except Exception:
                logger.exception("Usage analytics unavailable; using the curated set")
        return queries

    async def build_bundle(self, language: str, questions: Optional[List[str]] = None,
                           key: Optional[str] = None) -> dict:
        """Build a complete offline cache bundle for the given language."""
        questions = questions if questions is not None else await self._questions(language)
        key = key or self._bundle_key(language, questions)
        bundle_version = time.strftime("%Y%m%d-%H%M%S")

        queries = await self._build_query_cache(language, questions)
        treatments = await self._build_treatment_cache()

        bundle = {
            "version": bundle_version,
            "bundle_key": key,
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

        await asyncio.to_thread(_write_bundle, language, key, bundle)
        logger.info(f"Built cache bundle for {language}: {len(queries)} queries, {len(treatments)} treatments")
        return bundle

    # Answers drafted at once while building a bundle.
    BUILD_CONCURRENCY = 4

    async def _build_query_cache(self, language: str, queries: Optional[List[str]] = None) -> List[dict]:
        """Pre-compute responses for the twenty anticipated questions."""
        queries = queries if queries is not None else await self._questions(language)
        if not has_translation(language):
            logger.warning(
                "No translated question set for %s; bundle will be in English and "
                "will not match a non-English query. Recorded so this is visible "
                "rather than mistaken for a translation failure.", language,
            )
        if not self.rag_pipeline:
            return []

        limit = asyncio.Semaphore(self.BUILD_CONCURRENCY)

        async def one(query_text: str) -> Optional[dict]:
            async with limit:
                try:
                    # SimpleRAGPipeline.query's real signature; it returns a dict.
                    # The system id marks these as build rows in the retrieval
                    # audit, and keeps them out of per-person records.
                    result = await self.rag_pipeline.query(
                        question=query_text,
                        user_id=system("offline_bundle"),
                        source_language=language,
                    )
                    answer = (result or {}).get("answer") or ""
                    if (result or {}).get("status", "success") != "success" or not answer:
                        return None
                    safety = self._safety_manager().process_response(
                        query=query_text,
                        response=answer,
                        sources=result.get("sources") or [],
                        language=language,
                    )
                    return {
                        "query_hash": hashlib.sha256(
                            _normalise_for_hash(query_text).encode()
                        ).hexdigest(),
                        "query_text": query_text,
                        "query_language": language,
                        "response_text": safety.response,
                        "evidence_level": safety.evidence_level,
                        "sources": result.get("sources") or [],
                    }
                except Exception as e:
                    logger.warning(f"Failed to cache query '{query_text}': {e}")
                    return None

        entries = await asyncio.gather(*(one(q) for q in queries))
        return [entry for entry in entries if entry is not None]

    def _safety_manager(self):
        """The injected safety manager, else the process-wide one."""
        if self.safety_manager is None:
            from safety_enhancements import get_safety_manager

            self.safety_manager = get_safety_manager()
        return self.safety_manager

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
