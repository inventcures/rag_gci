"""
Gemini Live Service - Full Implementation

Provides real-time voice conversation capabilities using Google's Gemini Live API.

Features:
- WebSocket connection to Gemini Live API via Vertex AI
- Real-time audio streaming (send/receive)
- Session management with resumption support
- RAG context injection for grounded responses
- Multi-language support (en-IN, hi-IN, mr-IN, ta-IN)
- Smart query classification for RAG routing
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, AsyncGenerator, List, Set
from datetime import datetime
import numpy as np

from google import genai
from google.genai import types

from .config import (
    GeminiLiveConfig,
    get_config,
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    VOICE_OPTIONS,
    DEFAULT_VOICE,
    INPUT_SAMPLE_RATE,
    is_translation_model,
    model_supports_rag,
    model_uses_auto_language,
    model_uses_realtime_text,
)

# Voice safety wrapper
try:
    from voice_safety_wrapper import get_voice_safety_wrapper
    VOICE_SAFETY_AVAILABLE = True
except ImportError:
    VOICE_SAFETY_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Smart query classifier to determine if RAG should be triggered.

    Uses a hybrid approach:
    1. Skip short conversational phrases (greetings, yes/no, etc.)
    2. Use semantic similarity to health topics for longer queries
    """

    # Common conversational phrases to skip (in multiple languages)
    SKIP_PHRASES: Set[str] = {
        # English
        "yes", "no", "okay", "ok", "yeah", "yep", "nope", "sure", "thanks",
        "thank you", "hello", "hi", "hey", "bye", "goodbye", "good morning",
        "good afternoon", "good evening", "good night", "how are you",
        "i'm fine", "i am fine", "fine", "alright", "all right", "hmm", "um",
        "uh", "oh", "ah", "what", "sorry", "pardon", "excuse me", "please",
        "wait", "one moment", "just a moment", "hold on", "i see", "got it",
        "understood", "right", "correct", "wrong", "maybe", "perhaps",
        "i don't know", "i dont know", "not sure", "i think so", "i guess",
        "repeat", "say again", "come again", "what did you say", "again",
        "can you repeat", "repeat that", "please repeat",

        # Hindi
        "हां", "नहीं", "ठीक है", "अच्छा", "धन्यवाद", "शुक्रिया", "नमस्ते",
        "नमस्कार", "अलविदा", "क्षमा करें", "माफ़ कीजिए", "रुकिए", "एक मिनट",
        "समझ गया", "समझ गई", "सही", "गलत", "शायद", "पता नहीं", "हम्म",
        "जी", "जी हां", "जी नहीं", "बिल्कुल", "ज़रूर", "चलो", "अच्छा ठीक है",
        "फिर से बोलिए", "दोबारा बोलिए", "क्या बोला",

        # Marathi
        "हो", "नाही", "ठीक आहे", "चांगले", "धन्यवाद", "नमस्कार",
        "माफ करा", "थांबा", "एक मिनिट", "समजले", "बरोबर", "चुकीचे",
        "कदाचित", "माहित नाही", "हं", "होय", "परत सांगा",

        # Tamil
        "ஆம்", "இல்லை", "சரி", "நன்றி", "வணக்கம்", "மன்னிக்கவும்",
        "காத்திருங்கள்", "புரிந்தது", "சரியாக", "தவறு", "ஒருவேளை",
        "தெரியாது",
    }

    # Filler words to strip from queries (in multiple languages)
    FILLER_WORDS: Set[str] = {
        # English fillers
        "oh", "ah", "uh", "um", "er", "erm", "hmm", "hm", "mm", "mmm",
        "like", "you know", "i mean", "basically", "actually", "literally",
        "so", "well", "anyway", "anyways", "right", "okay so", "um so",
        "uh huh", "uh oh", "ooh", "aah", "ahh", "ohh", "uhh", "umm",
        "eh", "meh", "huh", "wow", "whoa", "oops", "yikes", "geez",
        "gosh", "darn", "shoot", "man", "dude", "bro", "yo",

        # Hindi fillers
        "अरे", "अच्छा", "हाँ", "हां", "ना", "तो", "बस", "यार", "भाई",
        "मतलब", "वो", "ये", "क्या", "कैसे", "ऐसे", "वैसे", "अब",
        "हम्म", "हं", "उं", "आं", "ओह", "आह", "उह", "एं", "हाय",
        "देखो", "सुनो", "बोलो", "चलो", "अरे यार", "अरे भाई",
        "असल में", "दरअसल", "वास्तव में", "सच में",

        # Hinglish fillers
        "like", "you know", "actually", "basically", "matlab",
        "toh", "na", "yaar", "bhai", "boss", "dude", "bro",
        "arrey", "arey", "haan", "nahi", "bas", "dekho", "suno",

        # Marathi fillers
        "अरे", "बरं", "हो", "ना", "म्हणजे", "तर", "आता", "मग",
        "काय", "कसं", "असं", "तसं", "बघ", "ऐक", "हं", "आं",
        "ओह", "आह", "उह", "अहो", "अगं", "अगा",

        # Tamil fillers
        "அட", "ஆமா", "இல்ல", "அப்போ", "சரி", "என்ன", "எப்படி",
        "அப்படி", "இப்படி", "பாரு", "கேளு", "ஹ்ம்", "ஆ", "ஓ",
        "உம்", "ஏய்", "டா", "டி", "மச்சான்", "நண்பா",
    }

    # Health/palliative care seed phrases for semantic similarity
    HEALTH_SEED_PHRASES: List[str] = [
        "pain management medication",
        "symptom control treatment",
        "palliative care support",
        "end of life care",
        "cancer treatment side effects",
        "morphine dosage administration",
        "nausea vomiting remedy",
        "breathing difficulty dyspnea",
        "wound care dressing",
        "nutrition feeding tube",
        "caregiver stress burnout",
        "hospice care options",
        "bedsore pressure ulcer prevention",
        "constipation laxative treatment",
        "anxiety depression management",
        "sleep problems insomnia",
        "dehydration fluid intake",
        "fever infection symptoms",
        "swallowing difficulty dysphagia",
        "fatigue weakness energy",
    ]

    # Out-of-scope keywords that indicate non-palliative queries
    OUT_OF_SCOPE_KEYWORDS: Set[str] = {
        # Programming/Tech
        "python", "javascript", "java", "code", "coding", "program", "programming",
        "script", "algorithm", "function", "variable", "loop", "array", "list",
        "linked list", "binary tree", "database", "sql", "html", "css", "api",
        "software", "developer", "debug", "compile", "github", "git",

        # Creative writing
        "poem", "poetry", "story", "write me", "compose", "song", "lyrics",
        "essay", "novel", "fiction", "creative writing",

        # Entertainment
        "movie", "film", "music", "game", "video game", "sports", "cricket",
        "football", "celebrity", "actor", "actress", "singer",

        # General knowledge unrelated to health
        "weather", "recipe", "cooking", "travel", "tourism", "holiday",
        "politics", "election", "stock market", "cryptocurrency", "bitcoin",
        "history", "geography", "mathematics", "physics", "chemistry",

        # Random requests
        "joke", "riddle", "puzzle", "trivia", "quiz", "horoscope",
        "translate", "translation", "dictionary",
    }

    # Polite decline messages in different languages
    DECLINE_MESSAGES: Dict[str, str] = {
        "en-IN": "I am Palli Sahayak - a palliative care helpline powered by AI. My current focus is on palliative care, and I may not have information to fully answer your query. Please ask me a palliative care related question and I would be happy to help you.",

        "hi-IN": "मैं पल्ली सहायक हूं - एक AI संचालित पैलिएटिव केयर हेल्पलाइन। मेरा वर्तमान फोकस पैलिएटिव केयर पर है, और मेरे पास आपके प्रश्न का पूर्ण उत्तर देने की जानकारी नहीं हो सकती। कृपया मुझसे पैलिएटिव केयर से संबंधित प्रश्न पूछें और मुझे आपकी मदद करने में खुशी होगी।",

        "mr-IN": "मी पल्ली सहायक आहे - AI द्वारे संचालित पॅलिएटिव्ह केअर हेल्पलाइन। माझे सध्याचे लक्ष पॅलिएटिव्ह केअरवर आहे, आणि तुमच्या प्रश्नाचे पूर्ण उत्तर देण्यासाठी माझ्याकडे माहिती नसू शकते। कृपया मला पॅलिएटिव्ह केअर संबंधित प्रश्न विचारा आणि मला तुमची मदत करण्यात आनंद होईल.",

        "ta-IN": "நான் பல்லி சஹாயக் - AI மூலம் இயங்கும் பேலியேட்டிவ் கேர் ஹெல்ப்லைன். எனது தற்போதைய கவனம் பேலியேட்டிவ் கேர் மீது உள்ளது, உங்கள் கேள்விக்கு முழுமையாக பதிலளிக்க என்னிடம் தகவல் இல்லாமல் இருக்கலாம். தயவுசெய்து பேலியேட்டிவ் கேர் தொடர்பான கேள்வியைக் கேளுங்கள், உங்களுக்கு உதவ மகிழ்ச்சியாக இருப்பேன்.",
    }

    # Minimum similarity threshold for health topics (0.0 - 1.0)
    SIMILARITY_THRESHOLD = 0.35

    # Palliative care similarity threshold (higher than general health)
    PALLIATIVE_THRESHOLD = 0.30

    # Minimum query length (in words) to consider for RAG
    MIN_WORDS_FOR_RAG = 3

    def __init__(self, embedding_model=None):
        """
        Initialize the query classifier.

        Args:
            embedding_model: SentenceTransformer model for embeddings.
                           If None, will attempt to load when needed.
        """
        self._embedding_model = embedding_model
        self._health_embeddings = None
        self._initialized = False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and strip
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def strip_filler_words(self, text: str) -> str:
        """
        Remove filler words from the query text.

        Args:
            text: Original query text

        Returns:
            Cleaned text with filler words removed
        """
        if not text:
            return text

        original_text = text
        text_lower = text.lower()

        # First, remove multi-word fillers (longer phrases first)
        multi_word_fillers = sorted(
            [f for f in self.FILLER_WORDS if ' ' in f],
            key=len,
            reverse=True
        )
        for filler in multi_word_fillers:
            # Use word boundary matching for multi-word fillers
            pattern = r'(?<![a-zA-Z\u0900-\u097F\u0B80-\u0BFF])' + re.escape(filler) + r'(?![a-zA-Z\u0900-\u097F\u0B80-\u0BFF])'
            text_lower = re.sub(pattern, ' ', text_lower, flags=re.IGNORECASE)
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

        # Then, remove single-word fillers
        single_word_fillers = [f for f in self.FILLER_WORDS if ' ' not in f]
        words = text.split()
        cleaned_words = []

        for word in words:
            word_lower = word.lower().strip()
            # Remove punctuation for comparison
            word_clean = re.sub(r'[^\w]', '', word_lower)

            if word_clean and word_clean not in single_word_fillers:
                cleaned_words.append(word)

        cleaned_text = ' '.join(cleaned_words)
        # Normalize whitespace
        cleaned_text = ' '.join(cleaned_text.split())

        if cleaned_text != original_text:
            logger.debug(f"Stripped fillers: '{original_text}' -> '{cleaned_text}'")

        return cleaned_text

    def _is_skip_phrase(self, text: str) -> bool:
        """Check if text matches a skip phrase."""
        normalized = self._normalize_text(text)

        # Direct match
        if normalized in self.SKIP_PHRASES:
            return True

        # Check if it's a very short phrase (1-2 words)
        words = normalized.split()
        if len(words) <= 2:
            # Check each word
            for word in words:
                if word in self.SKIP_PHRASES:
                    return True

        return False

    def _initialize_embeddings(self) -> bool:
        """Initialize health topic embeddings (lazy loading)."""
        if self._initialized:
            return self._health_embeddings is not None

        self._initialized = True

        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("QueryClassifier: Loaded fallback embedding model")
            except Exception as e:
                logger.warning(f"QueryClassifier: Could not load embedding model: {e}")
                return False

        try:
            # Compute embeddings for health seed phrases
            self._health_embeddings = self._embedding_model.encode(
                self.HEALTH_SEED_PHRASES,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.info(f"QueryClassifier: Initialized with {len(self.HEALTH_SEED_PHRASES)} health seed phrases")
            return True
        except Exception as e:
            logger.error(f"QueryClassifier: Failed to compute health embeddings: {e}")
            return False

    def _compute_health_similarity(self, text: str) -> float:
        """
        Compute semantic similarity to health topics.

        Returns:
            Similarity score (0.0 - 1.0), or -1 if unable to compute
        """
        if not self._initialize_embeddings():
            return -1.0

        try:
            # Compute embedding for query
            query_embedding = self._embedding_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

            # Compute cosine similarity with all health phrases
            similarities = np.dot(self._health_embeddings, query_embedding)

            # Return max similarity
            return float(np.max(similarities))
        except Exception as e:
            logger.error(f"QueryClassifier: Error computing similarity: {e}")
            return -1.0

    def should_query_rag(self, text: str) -> tuple[bool, str]:
        """
        Determine if RAG should be queried for this text.

        Args:
            text: User's transcribed speech

        Returns:
            Tuple of (should_query: bool, reason: str)
        """
        if not text or not text.strip():
            return False, "empty_query"

        normalized = self._normalize_text(text)
        words = normalized.split()

        # Check 1: Skip known conversational phrases
        if self._is_skip_phrase(text):
            return False, "skip_phrase"

        # Check 2: Very short queries (less than MIN_WORDS_FOR_RAG words)
        if len(words) < self.MIN_WORDS_FOR_RAG:
            return False, f"too_short ({len(words)} words)"

        # Check 3: Semantic similarity to health topics
        similarity = self._compute_health_similarity(text)

        if similarity < 0:
            # Fallback: if embedding fails, allow queries with 4+ words
            if len(words) >= 4:
                return True, "fallback_length"
            return False, "embedding_failed"

        if similarity >= self.SIMILARITY_THRESHOLD:
            return True, f"health_topic (sim={similarity:.2f})"

        # Below threshold - probably not a health query
        return False, f"low_similarity (sim={similarity:.2f})"

    def set_embedding_model(self, model) -> None:
        """Set the embedding model to use."""
        self._embedding_model = model
        self._initialized = False
        self._health_embeddings = None

    def is_out_of_scope(self, text: str) -> tuple[bool, str]:
        """
        Check if the query is out of scope for palliative care.

        Args:
            text: User's query text

        Returns:
            Tuple of (is_out_of_scope: bool, matched_keyword: str or None)
        """
        if not text:
            return False, ""

        text_lower = text.lower()
        words = text_lower.split()

        # Check for out-of-scope keywords
        for keyword in self.OUT_OF_SCOPE_KEYWORDS:
            if ' ' in keyword:
                # Multi-word keyword
                if keyword in text_lower:
                    return True, keyword
            else:
                # Single word keyword
                if keyword in words:
                    return True, keyword

        return False, ""

    def get_decline_message(self, language: str) -> str:
        """
        Get the polite decline message in the specified language.

        Args:
            language: Language code (e.g., "hi-IN", "en-IN")

        Returns:
            Decline message in the appropriate language
        """
        return self.DECLINE_MESSAGES.get(language, self.DECLINE_MESSAGES["en-IN"])

    def get_decline_instruction(self, language: str, query: str) -> str:
        """
        Get a system instruction for Gemini to politely decline out-of-scope queries.

        Args:
            language: Language code
            query: The user's out-of-scope query

        Returns:
            Instruction for Gemini to respond appropriately
        """
        decline_msg = self.get_decline_message(language)

        language_names = {
            "en-IN": "English",
            "hi-IN": "Hindi",
            "mr-IN": "Marathi",
            "ta-IN": "Tamil",
        }
        lang_name = language_names.get(language, "English")

        instruction = f"""[IMPORTANT - OUT OF SCOPE QUERY]
The user asked: "{query}"

This query is outside the scope of palliative care. Please:
1. Give a very brief, polite acknowledgment of their question (1 sentence max)
2. Then clearly state that you are a palliative care assistant
3. Use the following message as your response (respond in {lang_name}):

"{decline_msg}"

Be warm and friendly, but redirect to palliative care topics.
[END INSTRUCTION]"""

        return instruction


class GeminiLiveError(Exception):
    """Exception raised for Gemini Live API errors."""
    pass


# Live API tool through which the model retrieves RAG context BEFORE
# answering. Function calling on Live models is synchronous: the model
# does not generate its (audio) response until the tool result is sent,
# which guarantees the spoken answer is grounded in the knowledge base
# rather than world knowledge alone.
RAG_TOOL_NAME = "search_medical_knowledge"

RAG_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=RAG_TOOL_NAME,
            description=(
                "Search the verified palliative-care knowledge base for "
                "evidence-based guidance. MUST be called before answering "
                "any health, symptom, medication, or care question."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "The user's health question, rephrased as a "
                            "concise English search query"
                        ),
                    )
                },
                required=["query"],
            ),
        )
    ]
)


class GeminiLiveService:
    """
    Main service for Gemini Live API integration.

    Provides:
    - create_session(): Create new voice conversation session
    - inject_rag_context(): Add RAG context to session
    - Active session management
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
        rag_pipeline: Optional[Any] = None,
        config: Optional[GeminiLiveConfig] = None
    ):
        """
        Initialize Gemini Live Service.

        Args:
            project_id: Google Cloud project ID (default from config)
            location: Vertex AI location (default from config)
            model: Gemini model ID (default from config)
            rag_pipeline: Reference to RAG pipeline for context injection
            config: Optional pre-loaded configuration
        """
        self.config = config or get_config()

        self.project_id = project_id or self.config.project_id
        self.location = location or self.config.location or "us-central1"
        self.model = model or self.config.model
        self.rag_pipeline = rag_pipeline

        # Initialize Google GenAI client
        self.client = self._create_client()

        # Active sessions (session_id -> GeminiLiveSession)
        self.active_sessions: Dict[str, "GeminiLiveSession"] = {}

        # Query classifier for smart RAG routing
        self.query_classifier = QueryClassifier()

        # Try to use RAG pipeline's embedding model if available
        if rag_pipeline and hasattr(rag_pipeline, 'embedding_model'):
            self.query_classifier.set_embedding_model(rag_pipeline.embedding_model)
            logger.info("QueryClassifier: Using RAG pipeline's embedding model")

        logger.info(
            f"GeminiLiveService initialized - "
            f"project={self._mask_project_id()}, location={self.location}, "
            f"model={self.model}"
        )

    def _mask_project_id(self) -> str:
        """Mask project ID for logging."""
        if not self.project_id:
            return "(not set)"
        if len(self.project_id) <= 8:
            return "***"
        return f"{self.project_id[:4]}...{self.project_id[-4:]}"

    def _create_client(self) -> Optional[genai.Client]:
        """Create Google GenAI client."""
        try:
            # Check if we have credentials
            if self.config.api_key:
                # Use API key authentication
                client = genai.Client(api_key=self.config.api_key)
                logger.info("GenAI client created with API key")
                return client
            elif self.project_id and not self.project_id.startswith("$"):
                # Use Vertex AI with ADC
                client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                logger.info("GenAI client created with Vertex AI (ADC)")
                return client
            else:
                logger.warning(
                    "No valid credentials for GenAI client. "
                    "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
                )
                return None
        except Exception as e:
            logger.error(f"Failed to create GenAI client: {e}")
            return None

    def _build_system_instruction(
        self,
        language: str,
        custom_instruction: Optional[str] = None
    ) -> str:
        """
        Build the medical/palliative care system instruction.

        Args:
            language: Language code (e.g., "hi-IN")
            custom_instruction: Optional custom instruction to append

        Returns:
            Complete system instruction string
        """
        # Auto-detect mode: mirror whatever language the user speaks (English,
        # Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati,
        # Assamese, or code-mixed). Native-audio handles this natively.
        lang_instruction = (
            "Detect the language the user speaks (Indian English, Hindi, Marathi, "
            "Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati, Assamese, or "
            "code-mixed) and respond in the SAME language with a warm, empathetic, "
            "culturally-appropriate Indian tone. If the user switches language "
            "mid-conversation, follow them."
        )

        base_instruction = f"""You are a compassionate palliative care assistant helping patients and caregivers with healthcare queries.

MANDATORY KNOWLEDGE BASE GROUNDING:
- For EVERY health, symptom, medication, side-effect, or care question, you MUST first call the {RAG_TOOL_NAME} tool with a concise English version of the question, and base your answer on what it returns.
- Never answer a medical question from your own general knowledge alone. If the tool returns no relevant information, say so, give only general comfort guidance, and advise consulting their doctor.
- When the tool returns sources, mention them naturally (e.g. "according to our palliative care guidelines").
- Only greetings, thanks, and small talk may be answered without the tool.

IMPORTANT GUIDELINES:
1. Be warm, empathetic, and supportive in all interactions
2. Provide accurate medical information from the knowledge base when available
3. Always recommend consulting healthcare professionals for serious concerns
4. Use simple, clear language appropriate for patients and families
5. Be culturally sensitive to Indian healthcare contexts
6. If unsure, acknowledge uncertainty and suggest professional consultation
7. Keep responses concise and focused - this is a voice conversation

LANGUAGE INSTRUCTION: {lang_instruction}

SAFETY GUIDELINES:
- Never provide emergency medical advice
- For emergencies, direct users to call emergency services or visit the nearest hospital
- Do not diagnose conditions - only provide general health information
- Always encourage professional medical consultation for specific concerns

CONVERSATION STYLE:
- Speak naturally as in a phone conversation
- Use appropriate pauses
- Confirm understanding when needed
- Be patient with users who may be distressed
"""

        if custom_instruction:
            base_instruction += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instruction}"

        return base_instruction

    def resolve_model(self, model: Optional[str]) -> str:
        """
        Resolve a requested model ID to a usable one.

        Args:
            model: Requested model ID (or None for the service default)

        Returns:
            Validated model ID
        """
        if not model:
            return self.model

        if model != self.model and model not in SUPPORTED_MODELS:
            logger.warning(
                f"Unknown model '{model}', falling back to {self.model}"
            )
            return self.model

        return model

    async def create_session(
        self,
        session_id: str,
        language: str = "en-IN",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None,
        model: Optional[str] = None
    ) -> "GeminiLiveSession":
        """
        Create a new Gemini Live session.

        Args:
            session_id: Unique identifier for this session
            language: Language code (en-IN, hi-IN, mr-IN, ta-IN)
            voice: Voice name (Aoede, Puck, Kore, etc.)
            system_instruction: Custom system prompt to append
            model: Live model ID (default from config). Translator models
                   (e.g. gemini-3.5-live-translate-preview) translate speech
                   into `language` instead of acting as an assistant.

        Returns:
            GeminiLiveSession object

        Raises:
            GeminiLiveError: If session creation fails
        """
        if not self.client:
            raise GeminiLiveError(
                "GenAI client not initialized. "
                "Check credentials (GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT)."
            )

        model = self.resolve_model(model)

        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Unsupported language {language}, falling back to en-IN"
            )
            language = "en-IN"

        # Validate voice
        if voice not in VOICE_OPTIONS:
            logger.warning(
                f"Unknown voice {voice}, falling back to {DEFAULT_VOICE}"
            )
            voice = DEFAULT_VOICE

        if is_translation_model(model):
            config = self._build_translation_config(language)
        else:
            config = self._build_assistant_config(
                model, language, voice, system_instruction
            )

        # Add transcription if enabled
        if self.config.transcription_enabled:
            config.input_audio_transcription = types.AudioTranscriptionConfig()
            config.output_audio_transcription = types.AudioTranscriptionConfig()

        # Create session object
        session = GeminiLiveSession(
            service=self,
            session_id=session_id,
            config=config,
            language=language,
            voice=voice,
            model=model
        )

        # Store in active sessions
        self.active_sessions[session_id] = session

        logger.info(
            f"Created Gemini Live session: {session_id} "
            f"(model={model}, language={language}, voice={voice})"
        )

        return session

    def _build_assistant_config(
        self,
        model: str,
        language: str,
        voice: str,
        system_instruction: Optional[str]
    ) -> types.LiveConnectConfig:
        """Build LiveConnectConfig for conversational assistant models."""
        full_instruction = self._build_system_instruction(
            language, system_instruction
        )

        # Native-audio 2.5 models and Gemini 3.x live models auto-detect
        # language and reject several BCP-47 codes we use (e.g. 'en-IN').
        # Omit language_code for those — the system instruction already
        # steers the reply language.
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            ),
        )
        if not model_uses_auto_language(model):
            speech_config.language_code = language

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
            system_instruction=types.Content(
                parts=[types.Part(text=full_instruction)]
            ),
            # RAG grounding tool: the model must fetch knowledge-base
            # context before generating its (audio) answer
            tools=[RAG_TOOL] if (
                self.rag_pipeline and self.config.rag_context_enabled
            ) else None,
            # VAD tuning for barge-in quality and latency:
            # - LOW start sensitivity + 100ms prefix padding: speech must be
            #   clear and sustained to open the mic, so ambient noise does not
            #   interrupt the assistant mid-sentence
            # - HIGH end sensitivity + 600ms silence window: end of the user's
            #   turn is detected sooner, so responses start faster
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=600,
                )
            ),
        )

        # Gemini 3.x live models support thinking; pin to MINIMAL (the
        # lowest-latency setting) so voice replies start as fast as possible
        if model_uses_realtime_text(model):
            config.thinking_config = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            )

        return config

    def _build_translation_config(
        self,
        language: str
    ) -> types.LiveConnectConfig:
        """
        Build LiveConnectConfig for speech-to-speech translator models.

        Translator models take no system instruction or voice config; the
        source language is auto-detected and output is translated into the
        session language.
        """
        if not hasattr(types, "TranslationConfig"):
            raise GeminiLiveError(
                "This google-genai SDK version does not support translation "
                "models. Upgrade with: pip install -U google-genai"
            )

        # TranslationConfig expects a bare BCP-47 primary tag (e.g. 'hi')
        target_language = language.split("-")[0]

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            translation_config=types.TranslationConfig(
                target_language_code=target_language,
                echo_target_language=True,
            ),
        )

    async def inject_rag_context(
        self,
        session: "GeminiLiveSession",
        query_context: str
    ) -> bool:
        """
        Inject RAG-retrieved context into an active session.

        Queries the RAG pipeline and sends relevant context to the
        Gemini session as a text message for grounding.

        Args:
            session: Active GeminiLiveSession
            query_context: Query to search RAG for relevant context

        Returns:
            True if context was injected, False otherwise
        """
        if not self.rag_pipeline:
            logger.debug("No RAG pipeline configured, skipping context injection")
            return False

        if not self.config.rag_context_enabled:
            logger.debug("RAG context injection disabled")
            return False

        if not session.is_active:
            logger.warning("Cannot inject context into inactive session")
            return False

        try:
            # Query RAG for relevant documents
            result = await self.rag_pipeline.query(
                question=query_context,
                conversation_id=session.session_id,
                user_id=session.session_id,
                top_k=self.config.rag_top_k
            )

            if result.get("status") != "success":
                logger.warning(f"RAG query failed: {result.get('error')}")
                return False

            context_used = result.get("context_used", "")
            if not context_used:
                logger.debug("No relevant RAG context found")
                return False

            # Format context message
            context_message = f"""[MEDICAL KNOWLEDGE BASE CONTEXT]
The following information from verified medical documents may be relevant to the user's query:

{context_used}

Use this information to provide accurate, evidence-based responses.
When using specific information from this context, mention the source.
[END CONTEXT]"""

            # Send to session
            await session.send_text(context_message)

            logger.info(
                f"Injected RAG context into session {session.session_id} "
                f"({len(context_used)} chars)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to inject RAG context: {e}")
            return False

    async def close_session(self, session_id: str) -> None:
        """
        Close and cleanup a session.

        Args:
            session_id: Session ID to close
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            try:
                await session.disconnect()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
            finally:
                del self.active_sessions[session_id]
                logger.info(f"Closed session: {session_id}")

    def is_available(self) -> bool:
        """
        Check if Gemini Live service is available.

        Returns:
            True if service is configured and ready
        """
        return (
            self.config.enabled and
            self.client is not None
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get service status for health checks.

        Returns:
            Status dictionary
        """
        return {
            "service": "GeminiLiveService",
            "status": "ready" if self.is_available() else "not_ready",
            "enabled": self.config.enabled,
            "client_initialized": self.client is not None,
            "project_id": self._mask_project_id(),
            "model": self.model,
            "supported_models": list(SUPPORTED_MODELS.keys()),
            "active_sessions": len(self.active_sessions),
            "supported_languages": self.config.supported_languages,
            "rag_enabled": self.config.rag_context_enabled,
            "fallback_enabled": self.config.fallback_enabled,
        }


class GeminiLiveSession:
    """
    Represents an active Gemini Live session.

    Handles:
    - Audio streaming (send/receive)
    - Text messaging
    - Session lifecycle
    - Transcription capture

    Uses asyncio queues to maintain the session within async context.
    """

    # Special marker bytes for control signals
    TURN_COMPLETE = b"__TURN_COMPLETE__"
    INTERRUPTED = b"__INTERRUPTED__"

    # Explicit interruption commands, matched on whole words of the live
    # input transcription while the model is speaking. These make "stop"
    # style barge-ins deterministic instead of relying on VAD alone.
    STOP_PHRASES: Set[str] = {
        # English
        "stop", "pause", "wait", "please stop", "please pause", "stop it",
        "stop talking", "be quiet", "hold on", "one moment", "enough",
        # Hindi / Hinglish
        "ruko", "rukiye", "bas", "bas karo", "chup", "रुको", "रुकिए",
        "बस", "बस करो", "चुप", "एक मिनट", "ठहरो",
        # Marathi
        "थांबा", "थांब", "पुरे", "शांत",
        # Tamil
        "நிறுத்து", "நிறுத்துங்கள்", "பொறு", "போதும்",
        # Telugu
        "ఆపు", "ఆపండి", "చాలు",
        # Bengali
        "থামো", "থামুন", "যথেষ্ট",
        # Kannada
        "ನಿಲ್ಲಿಸಿ", "ಸಾಕು",
        # Malayalam
        "നിർത്തൂ", "മതി",
        # Gujarati
        "રોકો", "બસ",
    }

    def __init__(
        self,
        service: GeminiLiveService,
        session_id: str,
        config: types.LiveConnectConfig,
        language: str = "en-IN",
        voice: str = "Aoede",
        model: Optional[str] = None
    ):
        """
        Initialize session.

        Args:
            service: Parent GeminiLiveService
            session_id: Unique session identifier
            config: LiveConnectConfig for the session
            language: Session language
            voice: Voice name
            model: Live model ID for this session (default: service model)
        """
        self.service = service
        self.session_id = session_id
        self.config = config
        self.language = language
        self.voice = voice
        self.model = model or service.model
        self.is_translation = is_translation_model(self.model)

        # RAG grounding via Live API function calling: the model calls the
        # search tool and waits for its result before generating the answer
        self._rag_tool_enabled = bool(
            service.rag_pipeline
            and service.config.rag_context_enabled
            and not self.is_translation
        )

        # Session state
        self.is_active = False
        self.is_connected = False
        self._session = None  # Actual genai session object
        self._context_manager = None  # Store the context manager

        # Async queues for audio data
        self._audio_in_queue: asyncio.Queue = asyncio.Queue()
        self._audio_out_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._session_task = None

        # Buffers for transcription
        self.transcription_buffer: List[str] = []
        self.response_buffer: List[str] = []

        # RAG integration
        self._pending_transcription: List[str] = []  # Accumulate transcription for RAG query
        self._rag_query_in_progress = False
        self._last_rag_query = ""  # Avoid duplicate queries

        # Gemini 3.x rejects send_client_content after the first model turn;
        # mid-conversation text must then go via send_realtime_input
        self._model_turn_seen = False

        # Barge-in state: transcription fragments seen while the model is
        # actively speaking, checked against STOP_PHRASES
        self._model_speaking = False
        self._speech_window: List[str] = []

        # Metadata
        self.created_at = datetime.now()
        self.last_activity = self.created_at

        # Session resumption
        self.resumption_handle: Optional[str] = None

        logger.debug(f"GeminiLiveSession created: {session_id}")

    async def connect(self) -> None:
        """
        Establish connection to Gemini Live API.

        Starts background task that maintains the session context.

        Raises:
            GeminiLiveError: If connection fails
        """
        if self.is_connected:
            logger.warning(f"Session {self.session_id} already connected")
            return

        if not self.service.client:
            raise GeminiLiveError("Service client not initialized")

        self._running = True
        self.is_connected = True
        self.is_active = True

        # Start the session management task
        self._session_task = asyncio.create_task(self._run_session())

        # Wait briefly for connection to establish
        await asyncio.sleep(0.1)

        logger.info(f"Connected session: {self.session_id}")

    async def _run_session(self) -> None:
        """
        Main session loop - maintains the async with context.

        This runs as a background task and keeps the WebSocket alive.
        """
        try:
            async with self.service.client.aio.live.connect(
                model=self.model,
                config=self.config
            ) as session:
                self._session = session
                logger.debug(f"Session {self.session_id} context established")

                # Run send and receive concurrently
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._send_loop())
                    tg.create_task(self._receive_loop())

        except asyncio.CancelledError:
            logger.debug(f"Session {self.session_id} cancelled")
        except Exception as e:
            logger.error(f"Session {self.session_id} error: {e}")
            # Put error in output queue to notify client
            await self._audio_out_queue.put({"error": str(e)})
        finally:
            self._session = None
            self.is_connected = False
            self.is_active = False
            self._running = False

    def _is_stop_command(self, fragment: str) -> bool:
        """
        Check whether the user just spoke an explicit stop command.

        Only evaluated while the model is speaking, over a short rolling
        window of transcription fragments (fragments may split words, e.g.
        "please " + "stop"). Whole-word matching keeps ambient noise or
        words containing a phrase (e.g. "nonstop") from triggering.
        """
        if not self._model_speaking:
            return False

        self._speech_window.append(fragment)
        # A stop command is short; keep only the most recent fragments
        if len(self._speech_window) > 6:
            self._speech_window = self._speech_window[-6:]

        window = "".join(self._speech_window).lower()
        window = re.sub(r"[^\w\sऀ-෿]", " ", window)
        words = window.split()
        if not words:
            return False

        # Check the last few words as 1- and 2-word candidates
        tail = words[-4:]
        candidates = set(tail)
        candidates.update(
            f"{tail[i]} {tail[i + 1]}" for i in range(len(tail) - 1)
        )
        return bool(candidates & self.STOP_PHRASES)

    async def _send_text_now(self, text: str, turn_complete: bool = True) -> None:
        """
        Send text to Gemini using the API the model accepts.

        Gemini 3.x live models reject send_client_content after the first
        model turn; text must then go via send_realtime_input. Translator
        models do not accept text input at all.
        """
        if self.is_translation:
            logger.warning(
                f"Session {self.session_id}: translator model does not "
                f"accept text input, dropping message"
            )
            return

        if model_uses_realtime_text(self.model) and self._model_turn_seen:
            await self._session.send_realtime_input(text=text)
        else:
            await self._session.send_client_content(
                turns=[types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                )],
                turn_complete=turn_complete
            )

    async def _send_loop(self) -> None:
        """Send audio from input queue to Gemini."""
        while self._running and self._session:
            try:
                # Wait for audio with timeout to check running state
                try:
                    data = await asyncio.wait_for(
                        self._audio_in_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if data is None:  # Shutdown signal
                    break

                if isinstance(data, bytes):
                    await self._session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}"
                        )
                    )
                elif isinstance(data, dict) and "text" in data:
                    await self._send_text_now(data["text"], turn_complete=True)

                self.last_activity = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")
                break

    async def _receive_loop(self) -> None:
        """Receive audio from Gemini and put in output queue."""
        while self._running and self._session:
            try:
                async for message in self._session.receive():
                    if not self._running:
                        break

                    self.last_activity = datetime.now()

                    if message.server_content:
                        content = message.server_content

                        # Model turn (audio output)
                        if content.model_turn:
                            self._model_turn_seen = True
                            for part in content.model_turn.parts:
                                if part.inline_data:
                                    self._model_speaking = True
                                    await self._audio_out_queue.put(part.inline_data.data)

                        # Input transcription - accumulate for RAG query
                        if content.input_transcription:
                            text = content.input_transcription.text
                            if text:
                                self.transcription_buffer.append(text)
                                self._pending_transcription.append(text)
                                logger.debug(f"User transcription: {text}")
                                # Translator models never send turn_complete,
                                # so relay transcription fragments immediately
                                if self.is_translation:
                                    await self._audio_out_queue.put(
                                        {"role": "user", "text": text, "partial": True}
                                    )
                                elif self._is_stop_command(text):
                                    logger.info(
                                        f"🛑 Stop phrase detected while model "
                                        f"speaking: {text!r}"
                                    )
                                    self._model_speaking = False
                                    self._speech_window.clear()
                                    await self._audio_out_queue.put(self.INTERRUPTED)

                        # Output transcription
                        if content.output_transcription:
                            text = content.output_transcription.text
                            if text:
                                self.response_buffer.append(text)
                                if self.is_translation:
                                    await self._audio_out_queue.put(
                                        {"role": "assistant", "text": text, "partial": True}
                                    )

                        # Turn complete - trigger RAG query with accumulated transcription
                        if content.turn_complete:
                            self._model_speaking = False
                            self._speech_window.clear()
                            await self._audio_out_queue.put(self.TURN_COMPLETE)

                            # Query RAG with user's transcription
                            if self._pending_transcription and not self._rag_query_in_progress:
                                asyncio.create_task(self._query_rag_and_inject())

                        # Interrupted
                        if content.interrupted:
                            self._model_speaking = False
                            self._speech_window.clear()
                            await self._audio_out_queue.put(self.INTERRUPTED)
                            # Clear pending transcription on interrupt
                            self._pending_transcription.clear()

                    # Tool call: the model is waiting on the RAG result and
                    # will not answer until we respond
                    if message.tool_call:
                        asyncio.create_task(
                            self._handle_tool_call(message.tool_call)
                        )

                    # Handle go_away
                    if message.go_away:
                        logger.warning(f"Session {self.session_id} go_away received")
                        break

                    # Handle resumption update
                    if message.session_resumption_update:
                        update = message.session_resumption_update
                        if update.resumable and update.new_handle:
                            self.resumption_handle = update.new_handle

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                break

    async def _handle_tool_call(self, tool_call) -> None:
        """
        Serve the model's RAG tool call.

        The Live model calls search_medical_knowledge and waits for this
        response before generating its audio answer, so whatever we return
        here directly grounds the spoken reply.
        """
        for fc in tool_call.function_calls or []:
            if fc.name != RAG_TOOL_NAME:
                logger.warning(f"Unknown tool call: {fc.name}")
                continue

            query = (fc.args or {}).get("query", "")
            logger.info(f"🔧 RAG TOOL CALL: {query[:100]!r} (session {self.session_id[:30]})")

            result_text = await self._run_rag_query(query)

            await self._session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"result": result_text},
                    )
                ]
            )
            logger.info(f"✅ RAG TOOL RESPONSE sent ({len(result_text)} chars)")

    async def _run_rag_query(self, query: str) -> str:
        """Run the RAG pipeline and format the result for the model."""
        no_info = (
            "No relevant information found in the knowledge base. Give only "
            "general comfort guidance and advise consulting their doctor."
        )
        if not query or not self.service.rag_pipeline:
            return no_info

        try:
            result = await self.service.rag_pipeline.query(
                question=query,
                conversation_id=self.session_id,
                user_id=self.session_id,
                top_k=self.service.config.rag_top_k,
            )
        except Exception as e:
            logger.error(f"RAG tool query failed: {e}")
            return no_info

        if result.get("status") != "success" or not result.get("answer"):
            return no_info

        sources = ", ".join(
            s.get("filename", "Unknown")[:40] for s in result.get("sources", [])[:3]
        )
        answer = result["answer"]
        logger.info(f"📚 RAG grounding: {len(answer)} chars, sources: {sources}")
        return (
            f"Verified palliative-care knowledge base result:\n{answer}\n\n"
            f"Sources: {sources or 'knowledge base'}\n"
            "Base your spoken answer on this information and mention the "
            "sources naturally."
        )

    async def _query_rag_and_inject(self) -> None:
        """Query RAG pipeline with user transcription and inject context."""
        if self.is_translation or not model_supports_rag(self.model):
            # Translator models only translate speech; they accept no text
            # input, so RAG/safety/decline injection cannot apply
            self._pending_transcription.clear()
            return

        if not self.service.rag_pipeline:
            logger.debug("No RAG pipeline configured, skipping context injection")
            self._pending_transcription.clear()
            return

        if not self.service.config.rag_context_enabled:
            logger.debug("RAG context injection disabled")
            self._pending_transcription.clear()
            return

        self._rag_query_in_progress = True

        try:
            # Combine accumulated transcription
            raw_query = " ".join(self._pending_transcription).strip()
            self._pending_transcription.clear()

            # Skip if empty
            if not raw_query or len(raw_query) < 5:
                return

            # Strip filler words before processing
            query_text = self.service.query_classifier.strip_filler_words(raw_query)

            # Log if fillers were stripped
            if query_text != raw_query:
                logger.info(f"🧹 Stripped fillers: \"{raw_query[:50]}\" -> \"{query_text[:50]}\"")

            # Skip if query is empty after stripping fillers
            if not query_text or len(query_text) < 3:
                logger.info(f"⏭️ SKIPPING RAG - empty after filler removal: \"{raw_query[:50]}\"")
                return

            if query_text == self._last_rag_query:
                logger.debug("Skipping duplicate RAG query")
                return
            
            # =================================================================
            # VOICE SAFETY CHECK - Emergency & Handoff Detection
            # =================================================================
            try:
                from voice_safety_wrapper import get_voice_safety_wrapper
                safety_wrapper = get_voice_safety_wrapper()
                
                safety_result = await safety_wrapper.check_voice_query(
                    user_id=self.session_id,
                    transcript=query_text,
                    language=self.language,
                    call_id=self.session_id,
                    conversation_history=[{"role": "user", "content": msg} for msg in self.transcription_buffer]
                )
                
                if safety_result.should_escalate:
                    logger.warning(f"🚨 Voice safety escalation triggered: {safety_result.event_type}")
                    
                    # Inject safety message to Gemini
                    if self._session and self._running:
                        safety_message = f"""[SAFETY ALERT - IMMEDIATE RESPONSE REQUIRED]
{safety_result.safety_message}

The user may need immediate medical attention or human assistance. 
Please prioritize their safety and provide clear, calm guidance.
[END SAFETY ALERT]"""
                        
                        await self._send_text_now(safety_message, turn_complete=False)
                        
                        # Handle escalation actions
                        await safety_wrapper.handle_voice_escalation(
                            safety_result, provider="gemini_live"
                        )
                    
                    self._last_rag_query = query_text
                    return
                
                # Update query text if modified by safety check
                if safety_result.modified_transcript:
                    query_text = safety_result.modified_transcript
                    
            except Exception as e:
                logger.error(f"Voice safety check error (proceeding without): {e}")

            # Check for out-of-scope queries first
            is_out_of_scope, matched_keyword = self.service.query_classifier.is_out_of_scope(query_text)

            if is_out_of_scope:
                logger.info(f"🚫 OUT OF SCOPE query detected (keyword: '{matched_keyword}'): \"{query_text[:50]}\"")

                # Inject decline instruction to Gemini
                if self._session and self._running:
                    decline_instruction = self.service.query_classifier.get_decline_instruction(
                        self.language, query_text
                    )
                    await self._send_text_now(decline_instruction, turn_complete=False)
                    logger.info(f"📢 Injected decline instruction for out-of-scope query (language: {self.language})")
                return

            # Smart classification: check if this query should trigger RAG
            should_query, reason = self.service.query_classifier.should_query_rag(query_text)

            if not should_query:
                # Check if it's low similarity - might be out of scope for palliative care
                if "low_similarity" in reason:
                    logger.info(f"🚫 LOW RELEVANCE query (not palliative care): \"{query_text[:50]}\"")

                    # Inject gentle redirect for non-palliative queries
                    if self._session and self._running:
                        decline_instruction = self.service.query_classifier.get_decline_instruction(
                            self.language, query_text
                        )
                        await self._send_text_now(decline_instruction, turn_complete=False)
                        logger.info(f"📢 Injected redirect for low-relevance query (language: {self.language})")
                    return

                logger.info(f"⏭️ SKIPPING RAG - {reason}: \"{query_text[:50]}{'...' if len(query_text) > 50 else ''}\"")
                return

            # With tool grounding, RAG context reaches the model BEFORE it
            # answers (via search_medical_knowledge); this post-turn
            # injection would arrive after the reply and is skipped
            if self._rag_tool_enabled:
                logger.debug("RAG handled via tool call - skipping post-turn injection")
                return

            self._last_rag_query = query_text

            logger.info(f"=" * 60)
            logger.info(f"🎙️ GEMINI LIVE - RAG QUERY ({reason})")
            logger.info(f"=" * 60)
            logger.info(f"🗣️ User said: {query_text[:100]}{'...' if len(query_text) > 100 else ''}")
            logger.info(f"🌐 Language: {self.language} | Session: {self.session_id[:30]}")

            # Query RAG pipeline with cleaned text
            result = await self.service.rag_pipeline.query(
                question=query_text,
                conversation_id=self.session_id,
                user_id=self.session_id,
                top_k=self.service.config.rag_top_k
            )

            if result.get("status") != "success":
                logger.warning(f"RAG query failed: {result.get('error')}")
                return

            context_used = result.get("context_used", 0)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            if not context_used or not answer:
                logger.info("❌ No relevant RAG context found")
                return

            # Format context for injection
            source_names = ", ".join([s.get("filename", "Unknown")[:30] for s in sources[:3]])
            context_message = f"""[MEDICAL KNOWLEDGE BASE - RELEVANT INFORMATION]
Based on your question, here is relevant information from verified medical documents:

{answer}

Sources: {source_names}

Please use this information to provide an accurate, grounded response. Mention the sources if helpful.
[END KNOWLEDGE BASE CONTEXT]"""

            # Inject context into the session
            if self._session and self._running:
                await self._send_text_now(context_message, turn_complete=False)

                logger.info(f"✅ RAG CONTEXT INJECTED")
                logger.info(f"📚 Sources: {source_names}")
                logger.info(f"💬 Context length: {len(answer)} chars")
                logger.info(f"=" * 60)

        except Exception as e:
            logger.error(f"Error in RAG query/inject: {e}")
        finally:
            self._rag_query_in_progress = False

    async def disconnect(self) -> None:
        """Close the session."""
        if not self.is_connected:
            return

        self._running = False

        # Signal send loop to stop
        await self._audio_in_queue.put(None)

        # Cancel the session task
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass

        self._session = None
        self.is_connected = False
        self.is_active = False
        logger.info(f"Disconnected session: {self.session_id}")

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk to Gemini via queue.

        Args:
            audio_chunk: Raw PCM audio (16kHz, 16-bit, mono, little-endian)

        Raises:
            GeminiLiveError: If session not connected
        """
        if not self.is_active or not self._running:
            raise GeminiLiveError("Session not connected")

        await self._audio_in_queue.put(audio_chunk)

    async def send_text(self, text: str) -> None:
        """
        Send text message to Gemini via queue.

        Args:
            text: Text message to send

        Raises:
            GeminiLiveError: If session not connected
        """
        if not self.is_active or not self._running:
            raise GeminiLiveError("Session not connected")

        await self._audio_in_queue.put({"text": text})
        logger.debug(f"Queued text for session {self.session_id}: {text[:50]}...")

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Receive audio responses from Gemini via queue.

        Yields:
            Raw PCM audio chunks (24kHz, 16-bit, mono, little-endian)
            Special markers: TURN_COMPLETE, INTERRUPTED

        Raises:
            GeminiLiveError: If session not connected
        """
        if not self.is_active or not self._running:
            raise GeminiLiveError("Session not connected")

        while self._running:
            try:
                # Get from output queue with timeout
                try:
                    data = await asyncio.wait_for(
                        self._audio_out_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if data is None:
                    break

                # Check for error
                if isinstance(data, dict) and "error" in data:
                    raise GeminiLiveError(data["error"])

                yield data

            except asyncio.CancelledError:
                break

    def get_transcription(self, clear: bool = True) -> str:
        """
        Get accumulated user transcription.

        Args:
            clear: Whether to clear the buffer after reading

        Returns:
            Concatenated transcription text
        """
        text = " ".join(self.transcription_buffer)
        if clear:
            self.transcription_buffer.clear()
        return text

    def get_response_transcription(self, clear: bool = True) -> str:
        """
        Get accumulated model response transcription.

        Args:
            clear: Whether to clear the buffer after reading

        Returns:
            Concatenated response text
        """
        text = " ".join(self.response_buffer)
        if clear:
            self.response_buffer.clear()
        return text

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "voice": self.voice,
            "model": self.model,
            "is_translation": self.is_translation,
            "is_active": self.is_active,
            "is_connected": self.is_connected,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "has_resumption_handle": self.resumption_handle is not None,
            "transcription_buffer_size": len(self.transcription_buffer),
            "response_buffer_size": len(self.response_buffer),
        }
