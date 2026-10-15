#!/usr/bin/env python3
"""
Retrieval index identity and drift detection
=============================================
Identifies which embedding model a vector index was built with, and refuses to
search it when that model does not match the configured one.

Why this exists
---------------
The live index was built with `BAAI/bge-small-en-v1.5` at 384 dimensions while the
corpus it serves is palliative care guidance for users of eleven languages.
Measured across all eleven, it scores 1.00 hit@5 on English and **0.00 on every
Indic language**. Nothing errors. The system simply returns confident, unfounded
clinical advice to a Marathi question.

Nothing in the existing code would catch that. `get_or_create_collection` reuses
whichever collection is already on disk regardless of the embedding function it was
created with, and the auto-rebuild path triggers on corruption rather than on a
model change. So changing the configured model does not reindex, and querying
proceeds against stale vectors regardless.

This module makes the mismatch explicit and actionable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ADR 0003, chosen by measurement: mean hit@5 0.96 against 0.72 for LaBSE and 0.09
# for the previous English-only index. See evaluation/retrieval_eval/results.json.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIMENSION = 1024

# Dimensionality for models we may need to reason about. ChromaDB records nothing
# about which model built a collection, so the dimension is the only observable
# proxy available from the index itself.
KNOWN_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "google/embeddinggemma-300m": 768,
}


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Which embedding model an index was built with."""

    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("EmbeddingIdentity requires a model name")
        if self.dimension <= 0:
            raise ValueError("EmbeddingIdentity requires a positive dimension")

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "dimension": self.dimension}


def load_identity_from_config(config: Optional[Dict[str, Any]]) -> EmbeddingIdentity:
    """
    Resolve the configured embedding identity.

    A single source of truth, so the pipeline, the reindex script and the release
    record cannot disagree about which model is meant to be running. The previous
    arrangement had three sources disagreeing at once.
    """
    section = (config or {}).get("embedding") or {}
    model = str(section.get("model") or DEFAULT_EMBEDDING_MODEL)
    dimension = section.get("dimension") or KNOWN_DIMENSIONS.get(model) or DEFAULT_EMBEDDING_DIMENSION
    return EmbeddingIdentity(model=model, dimension=int(dimension))


@dataclass
class IndexDrift:
    """
    The result of comparing the configured identity against the live index.

    Drift is never merely advisory here: a mismatch means every answer is being
    retrieved through the wrong geometry.
    """

    expected: EmbeddingIdentity
    observed: Optional[EmbeddingIdentity]
    reasons: List[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.reasons)

    @property
    def remediation(self) -> str:
        if not self.reasons:
            return "No action required."
        return (
            "Reindex the corpus with the configured embedding model "
            f"({self.expected.model}, {self.expected.dimension}d) using "
            "scripts/reindex_with_configured_model.py, then restart the server. "
            "Until then every query is refused: a mismatch returns confident "
            "but ungrounded results rather than an error."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected": self.expected.to_dict(),
            "observed": self.observed.to_dict() if self.observed else None,
            "has_drift": self.has_drift,
            "reasons": list(self.reasons),
            "remediation": self.remediation,
        }

    def raise_if_drifted(self) -> None:
        """
        Refuse to continue on a mismatched index.

        Callers that search anyway produce answers that look healthy, which is the
        failure this whole module exists to prevent.
        """
        if self.has_drift:
            raise IndexMismatchError(self)


class IndexMismatchError(RuntimeError):
    """The live index was not built with the configured embedding model."""

    def __init__(self, drift: IndexDrift):
        observed = (
            f"{drift.observed.model} at {drift.observed.dimension}d"
            if drift.observed else "an unobservable index"
        )
        super().__init__(
            f"Embedding index mismatch: configured {drift.expected.model} at "
            f"{drift.expected.dimension}d, index was built with {observed}. "
            f"{drift.remediation}"
        )
        self.drift = drift


def detect_drift(
    expected: EmbeddingIdentity,
    observed: Optional[EmbeddingIdentity],
) -> IndexDrift:
    """
    Compare the configured identity against what the index actually holds.

    An unobservable index counts as drift rather than as agreement. Defaulting to
    "no news is good news" here would reintroduce exactly the silent failure this
    guards against.
    """
    reasons: List[str] = []

    if observed is None:
        reasons.append("index_unobservable")
        return IndexDrift(expected=expected, observed=None, reasons=reasons)

    if observed.dimension != expected.dimension:
        reasons.append("dimension_mismatch")

    if _normalise(observed.model) != _normalise(expected.model):
        reasons.append("model_mismatch")

    return IndexDrift(expected=expected, observed=observed, reasons=reasons)


def _normalise(model: str) -> str:
    """Compare model names tolerantly: a local path is the same model as its repo."""
    name = str(model).rstrip("/").split("/")[-1] if "/" in str(model) else str(model)
    return name.strip().lower()


def observe_index(collection: Any) -> Optional[EmbeddingIdentity]:
    """
    Read whatever identity the live index can tell us about itself.

    ChromaDB does not record which model built a collection, so the only reliable
    observation is dimensionality. Where that is ambiguous, we deliberately return
    None rather than guessing a model, and the caller treats unobservable as drift.
    """
    if collection is None:
        return None

    try:
        sample = collection.get(limit=1, include=["embeddings"])
    except Exception:
        logger.warning("Could not sample the index to observe its dimension", exc_info=True)
        return None

    embeddings = sample.get("embeddings") if isinstance(sample, dict) else None
    # ChromaDB returns a numpy array here, so truthiness is ambiguous and must not
    # be used. Length is the only safe test.
    if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
        # An empty collection is not yet drifted; it simply has nothing to say.
        return None

    dimension = len(embeddings[0])
    candidates = [m for m, d in KNOWN_DIMENSIONS.items() if d == dimension]
    model = candidates[0] if len(candidates) == 1 else f"unknown-{dimension}d"
    return EmbeddingIdentity(model=model, dimension=dimension)


def check_live_index(
    collection: Any,
    expected: EmbeddingIdentity,
) -> IndexDrift:
    """
    Compare the configured identity against a live ChromaDB collection.

    The collection's own metadata is preferred over inference from dimensionality,
    because a record written at index time is evidence rather than a guess.
    """
    observed = None
    try:
        metadata = getattr(collection, "metadata", None) or {}
        model = metadata.get("embedding_model")
        if model:
            observed = EmbeddingIdentity(
                model=str(model),
                dimension=int(metadata.get("embedding_dimension") or expected.dimension),
            )
    except Exception:
        logger.warning("Could not read index metadata", exc_info=True)

    if observed is None:
        observed = observe_index(collection)

    return detect_drift(expected, observed)


def write_index_metadata(
    collection: Any,
    identity: EmbeddingIdentity,
) -> None:
    """
    Record the identity on the collection at reindex time.

    Without this, the only available observation is dimensionality, and two models
    sharing 1024 dimensions are indistinguishable. Writing it now means a later
    mismatch is diagnosable rather than merely detectable.
    """
    collection.modify(metadata={
        "embedding_model": identity.model,
        "embedding_dimension": identity.dimension,
    })


def stamp_if_empty(collection: Any, identity: EmbeddingIdentity) -> bool:
    """
    Record `identity` on an empty collection that records none.

    An empty collection has no vectors to observe, so it would read as
    unobservable and be refused forever, even after its first upload. Whatever
    is added next is embedded by the pipeline's own function, which is the
    configured model, so recording it now is evidence rather than a guess.
    Returns True if it stamped.
    """
    try:
        metadata = getattr(collection, "metadata", None) or {}
        if metadata.get("embedding_model") or collection.count() != 0:
            return False
        write_index_metadata(collection, identity)
    except Exception:
        logger.warning("Could not record the identity on an empty index", exc_info=True)
        return False
    logger.info("Empty index: recorded %s (%dd) as its embedding model", identity.model, identity.dimension)
    return True


# What a participant reads or hears while the index is refused. It names no
# cause they could act on, and keeps the emergency route, because the refusal
# can land on a question asked in distress. The translations await clinical
# review, like the Deferral's.
INDEX_UNAVAILABLE_MESSAGE: Dict[str, str] = {
    "en": (
        "This service is temporarily unavailable and cannot answer questions right now. "
        "Please try again later. If the person is in severe distress right now — "
        "struggling to breathe, unresponsive, or in pain that cannot be settled — "
        "call 108 for an ambulance."
    ),
    "hi": (
        "यह सेवा अभी अस्थायी रूप से उपलब्ध नहीं है और अभी सवालों के जवाब नहीं दे सकती। "
        "कृपया थोड़ी देर बाद फिर से कोशिश करें। अगर मरीज़ को अभी बहुत तेज़ तकलीफ़ है — "
        "साँस लेने में दिक्कत, बेहोशी, या न ठीक होने वाला दर्द — तो 108 पर एम्बुलेंस बुलाएँ।"
    ),
    "bn": (
        "এই পরিষেবাটি এখন সাময়িকভাবে বন্ধ আছে এবং এই মুহূর্তে প্রশ্নের উত্তর দিতে পারছে না। "
        "অনুগ্রহ করে কিছুক্ষণ পরে আবার চেষ্টা করুন। রোগী এখন খুব কষ্টে থাকলে — শ্বাসকষ্ট, "
        "অচেতন, বা সামলানো যায় না এমন ব্যথা — 108 নম্বরে অ্যাম্বুলেন্স ডাকুন।"
    ),
    "ta": (
        "இந்தச் சேவை தற்காலிகமாகக் கிடைக்கவில்லை, இப்போது கேள்விகளுக்குப் பதில் அளிக்க முடியாது. "
        "சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும். நோயாளர் இப்போது மிகவும் வேதனையில் இருந்தால் — "
        "மூச்சுத் திணறல், நினைவிழப்பு, அல்லது சமாளிக்க முடியாத வலி — 108-ல் ஆம்புலன்ஸை அழைக்கவும்."
    ),
    "te": (
        "ఈ సేవ ప్రస్తుతం తాత్కాలికంగా అందుబాటులో లేదు, ఇప్పుడు ప్రశ్నలకు సమాధానం ఇవ్వలేదు. "
        "దయచేసి కొద్దిసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి. రోగి ఇప్పుడు తీవ్రమైన బాధలో ఉంటే — "
        "ఊపిరి ఆడకపోవడం, స్పృహ లేకపోవడం, లేదా తగ్గని నొప్పి — 108కి కాల్ చేసి అంబులెన్స్ పిలవండి."
    ),
    "mr": (
        "ही सेवा सध्या तात्पुरती उपलब्ध नाही आणि आत्ता प्रश्नांची उत्तरे देऊ शकत नाही. "
        "कृपया थोड्या वेळाने पुन्हा प्रयत्न करा. रुग्णाला आत्ता खूप त्रास होत असेल — "
        "श्वास घेण्यास त्रास, बेशुद्धी, किंवा कमी न होणारी वेदना — तर 108 वर रुग्णवाहिका बोलवा."
    ),
    "gu": (
        "આ સેવા હાલમાં અસ્થાયી રૂપે ઉપલબ્ધ નથી અને અત્યારે પ્રશ્નોના જવાબ આપી શકતી નથી. "
        "કૃપા કરીને થોડી વાર પછી ફરી પ્રયાસ કરો. જો દર્દીને અત્યારે ખૂબ તકલીફ હોય — "
        "શ્વાસ લેવામાં તકલીફ, બેભાન, અથવા ન મટતો દુખાવો — તો 108 પર એમ્બ્યુલન્સ બોલાવો."
    ),
    "kn": (
        "ಈ ಸೇವೆ ಸದ್ಯಕ್ಕೆ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ ಮತ್ತು ಈಗ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. "
        "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ. ರೋಗಿಗೆ ಈಗ ತೀವ್ರ ತೊಂದರೆ ಇದ್ದರೆ — "
        "ಉಸಿರಾಟದ ತೊಂದರೆ, ಪ್ರಜ್ಞೆ ಇಲ್ಲದಿರುವುದು, ಅಥವಾ ಕಡಿಮೆಯಾಗದ ನೋವು — 108ಕ್ಕೆ ಕರೆ ಮಾಡಿ ಆಂಬುಲೆನ್ಸ್ ಕರೆಸಿ."
    ),
    "ml": (
        "ഈ സേവനം ഇപ്പോൾ താൽക്കാലികമായി ലഭ്യമല്ല, ഇപ്പോൾ ചോദ്യങ്ങൾക്ക് മറുപടി നൽകാൻ കഴിയില്ല. "
        "ദയവായി കുറച്ചു സമയത്തിനു ശേഷം വീണ്ടും ശ്രമിക്കുക. രോഗിക്ക് ഇപ്പോൾ കടുത്ത ബുദ്ധിമുട്ടുണ്ടെങ്കിൽ — "
        "ശ്വാസതടസ്സം, ബോധമില്ലായ്മ, അല്ലെങ്കിൽ ശമിക്കാത്ത വേദന — 108-ൽ വിളിച്ച് ആംബുലൻസ് വരുത്തുക."
    ),
    "pa": (
        "ਇਹ ਸੇਵਾ ਇਸ ਵੇਲੇ ਆਰਜ਼ੀ ਤੌਰ 'ਤੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ ਅਤੇ ਹੁਣ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਨਹੀਂ ਦੇ ਸਕਦੀ। "
        "ਕਿਰਪਾ ਕਰਕੇ ਥੋੜ੍ਹੀ ਦੇਰ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ। ਜੇ ਮਰੀਜ਼ ਨੂੰ ਹੁਣ ਬਹੁਤ ਤਕਲੀਫ਼ ਹੈ — "
        "ਸਾਹ ਲੈਣ ਵਿੱਚ ਮੁਸ਼ਕਲ, ਬੇਹੋਸ਼ੀ, ਜਾਂ ਨਾ ਹਟਣ ਵਾਲਾ ਦਰਦ — ਤਾਂ 108 'ਤੇ ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ।"
    ),
    "or": (
        "ଏହି ସେବା ବର୍ତ୍ତମାନ ଅସ୍ଥାୟୀ ଭାବରେ ଉପଲବ୍ଧ ନାହିଁ ଏବଂ ଏବେ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଇପାରିବ ନାହିଁ। "
        "ଦୟାକରି କିଛି ସମୟ ପରେ ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ। ଯଦି ରୋଗୀଙ୍କୁ ଏବେ ବହୁତ କଷ୍ଟ ହେଉଛି — "
        "ନିଶ୍ୱାସ ନେବାରେ ଅସୁବିଧା, ଚେତାଶୂନ୍ୟ, କିମ୍ବା ନ କମୁଥିବା ଯନ୍ତ୍ରଣା — ତେବେ 108ରେ ଆମ୍ବୁଲାନ୍ସ ଡାକନ୍ତୁ।"
    ),
}


def index_unavailable_message(language: Optional[str]) -> str:
    """The refusal for `language` (any spelling of a Supported Language: "hi",
    "hi-IN", "od-IN"), in English when there is none."""
    from supported_languages import get_language

    lang = get_language(language)
    return INDEX_UNAVAILABLE_MESSAGE.get(lang.code if lang else "en", INDEX_UNAVAILABLE_MESSAGE["en"])
