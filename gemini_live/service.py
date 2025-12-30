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
    VOICE_OPTIONS,
    DEFAULT_VOICE,
    INPUT_SAMPLE_RATE,
)

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
        language_instructions = {
            "en-IN": "Respond in Indian English with a warm, empathetic tone.",
            "hi-IN": "हिंदी में जवाब दें। गर्मजोशी और सहानुभूति के साथ बात करें।",
            "mr-IN": "मराठीत उत्तर द्या. सहानुभूती आणि काळजी घेणारा स्वर वापरा.",
            "ta-IN": "தமிழில் பதிலளிக்கவும். அன்பான மற்றும் பரிவான தொனியில் பேசுங்கள்."
        }

        lang_instruction = language_instructions.get(
            language,
            language_instructions["en-IN"]
        )

        base_instruction = f"""You are a compassionate palliative care assistant helping patients and caregivers with healthcare queries.

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

    async def create_session(
        self,
        session_id: str,
        language: str = "en-IN",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None
    ) -> "GeminiLiveSession":
        """
        Create a new Gemini Live session.

        Args:
            session_id: Unique identifier for this session
            language: Language code (en-IN, hi-IN, mr-IN, ta-IN)
            voice: Voice name (Aoede, Puck, Kore, etc.)
            system_instruction: Custom system prompt to append

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

        # Build system instruction
        full_instruction = self._build_system_instruction(
            language, system_instruction
        )

        # Build configuration
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                ),
                language_code=language
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=full_instruction)]
            ),
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
            voice=voice
        )

        # Store in active sessions
        self.active_sessions[session_id] = session

        logger.info(
            f"Created Gemini Live session: {session_id} "
            f"(language={language}, voice={voice})"
        )

        return session

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

    def __init__(
        self,
        service: GeminiLiveService,
        session_id: str,
        config: types.LiveConnectConfig,
        language: str = "en-IN",
        voice: str = "Aoede"
    ):
        """
        Initialize session.

        Args:
            service: Parent GeminiLiveService
            session_id: Unique session identifier
            config: LiveConnectConfig for the session
            language: Session language
            voice: Voice name
        """
        self.service = service
        self.session_id = session_id
        self.config = config
        self.language = language
        self.voice = voice

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
                model=self.service.model,
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
                    await self._session.send_client_content(
                        turns=[types.Content(
                            role="user",
                            parts=[types.Part(text=data["text"])]
                        )],
                        turn_complete=True
                    )

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
                            for part in content.model_turn.parts:
                                if part.inline_data:
                                    await self._audio_out_queue.put(part.inline_data.data)

                        # Input transcription - accumulate for RAG query
                        if content.input_transcription:
                            text = content.input_transcription.text
                            if text:
                                self.transcription_buffer.append(text)
                                self._pending_transcription.append(text)
                                logger.debug(f"User transcription: {text}")

                        # Output transcription
                        if content.output_transcription:
                            text = content.output_transcription.text
                            if text:
                                self.response_buffer.append(text)

                        # Turn complete - trigger RAG query with accumulated transcription
                        if content.turn_complete:
                            await self._audio_out_queue.put(self.TURN_COMPLETE)

                            # Query RAG with user's transcription
                            if self._pending_transcription and not self._rag_query_in_progress:
                                asyncio.create_task(self._query_rag_and_inject())

                        # Interrupted
                        if content.interrupted:
                            await self._audio_out_queue.put(self.INTERRUPTED)
                            # Clear pending transcription on interrupt
                            self._pending_transcription.clear()

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

    async def _query_rag_and_inject(self) -> None:
        """Query RAG pipeline with user transcription and inject context."""
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

            # Check for out-of-scope queries first
            is_out_of_scope, matched_keyword = self.service.query_classifier.is_out_of_scope(query_text)

            if is_out_of_scope:
                logger.info(f"🚫 OUT OF SCOPE query detected (keyword: '{matched_keyword}'): \"{query_text[:50]}\"")

                # Inject decline instruction to Gemini
                if self._session and self._running:
                    decline_instruction = self.service.query_classifier.get_decline_instruction(
                        self.language, query_text
                    )
                    await self._session.send_client_content(
                        turns=[types.Content(
                            role="user",
                            parts=[types.Part(text=decline_instruction)]
                        )],
                        turn_complete=False
                    )
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
                        await self._session.send_client_content(
                            turns=[types.Content(
                                role="user",
                                parts=[types.Part(text=decline_instruction)]
                            )],
                            turn_complete=False
                        )
                        logger.info(f"📢 Injected redirect for low-relevance query (language: {self.language})")
                    return

                logger.info(f"⏭️ SKIPPING RAG - {reason}: \"{query_text[:50]}{'...' if len(query_text) > 50 else ''}\"")
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
                await self._session.send_client_content(
                    turns=[types.Content(
                        role="user",
                        parts=[types.Part(text=context_message)]
                    )],
                    turn_complete=False  # Don't mark as complete - let model continue
                )

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
            "is_active": self.is_active,
            "is_connected": self.is_connected,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "has_resumption_handle": self.resumption_handle is not None,
            "transcription_buffer_size": len(self.transcription_buffer),
            "response_buffer_size": len(self.response_buffer),
        }
