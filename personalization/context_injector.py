"""
Context Injector

Integrates longitudinal patient context into RAG queries for personalized,
compassionate responses. Ensures continuity of care across sessions.

This module bridges the gap between the longitudinal memory system and
the RAG query pipeline, injecting relevant patient context into LLM prompts
while maintaining empathetic, multi-language communication.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .longitudinal_memory import (
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    TemporalTrend,
    SeverityLevel,
)
from .user_profile import UserProfileManager, UserProfile, CommunicationStyle
from .context_memory import ContextMemory, PatientContext

logger = logging.getLogger(__name__)


# ============================================================================
# COMPASSIONATE LANGUAGE TEMPLATES
# ============================================================================

COMPASSIONATE_TEMPLATES = {
    "en-IN": {
        "welcome_back": [
            "Welcome back. I remember speaking with you before.",
            "Good to hear from you again.",
            "I'm glad you reached out again."
        ],
        "continuity": [
            "As we discussed before,",
            "Following up on our previous conversation,",
            "You mentioned earlier that"
        ],
        "empathy_worsening": [
            "I'm sorry to hear that things have been more difficult lately.",
            "I understand that this has been a challenging time.",
            "I appreciate you sharing how things have been."
        ],
        "empathy_improving": [
            "It's good to hear that things have been a bit better.",
            "I'm glad to see some improvement.",
            "That's encouraging to hear."
        ],
        "empathy_stable": [
            "I see things have been relatively stable.",
            "Glad to know things have been steady."
        ],
        "transition_phrases": [
            "Let's see how we can help with that.",
            "Here's what might be helpful for you.",
            "Based on what you've shared, here's some guidance."
        ],
        "emotional_support": {
            "anxiety": "I understand feeling anxious can be difficult.",
            "depression": "I hear that you've been feeling low.",
            "fear": "It's natural to feel fearful sometimes.",
            "pain": "I know pain can be very challenging to cope with."
        },
        "symptom_trend": {
            "improving": "{symptom} has been improving",
            "stable": "{symptom} has been stable",
            "worsening": "{symptom} has been getting worse",
            "fluctuating": "{symptom} has been up and down"
        },
        "medication_current": "Currently taking {medications}",
        "alert_concern": "I'm concerned about",
        "care_continuity": "As part of your ongoing care,"
    },
    "hi-IN": {
        "welcome_back": [
            "वापसी पर स्वागत है। मुझे याद है कि हमने पहले बात की थी।",
            "फिर से बात करके अच्छा लगा।",
            "आपसे फिर जुड़ना अच्छा लग रहा है।"
        ],
        "continuity": [
            "जैसा हमने पहले बात की थी,",
            "पिछली बातचीत के अनुसार,",
            "आपने पहले बताया था कि"
        ],
        "empathy_worsening": [
            "मुझे खेद है कि चीजें पहले से ज्यादा कठिन हो गई हैं।",
            "मैं समझता हूं कि यह समय मुश्किल रहा होगा।",
            "आपने अपने साथ शेयर किया, इसके लिए धन्यवाद।"
        ],
        "empathy_improving": [
            "सुनकर अच्छा लगा कि चीजें कुछ बेहतर हुई हैं।",
            "कुछ सुधार देखकर खुशी हुई।",
            "यह सुनकर राहत मिली।"
        ],
        "empathy_stable": [
            "चीजें स्थिर हैं, यह जानकर अच्छा लगा।",
            "स्थिति स्थिर है, यह अच्छा है।"
        ],
        "transition_phrases": [
            "देखते हैं कि हम इसमें कैसे मदद कर सकते हैं।",
            "यहाँ कुछ मार्गदर्शन दिया गया है।",
            "आपने जो शेयर किया, उसके आधार पर यह मददगार हो सकता है।"
        ],
        "emotional_support": {
            "anxiety": "मैं समझता हूं कि चिंता होना मुश्किल होता है।",
            "depression": "मैं सुन रहा हूं कि आप उदास महसूस कर रहे हैं।",
            "fear": "डर अनुभव करना स्वाभाविक है।",
            "pain": "मुझे पता है कि दर्द का सामना करना बहुत कठिन होता है।"
        },
        "symptom_trend": {
            "improving": "{symptom} में सुधार हो रहा है",
            "stable": "{symptom} स्थिर है",
            "worsening": "{symptom} बिगड़ रहा है",
            "fluctuating": "{symptom} ऊपर-नीचे हो रहा है"
        },
        "medication_current": "वर्तमान में {medications} ले रहे हैं",
        "alert_concern": "मुझे चिंता है कि",
        "care_continuity": "आपकी निरंतर देखभाल के रूप में,"
    },
    "bn-IN": {  # Bengali
        "welcome_back": "ফিরে আপনাকে স্বাগতম। আমি মনে করতে পারি আমরা আগে কথা বলেছিলাম।",
        "empathy_worsening": "আমি দুঃখিত যে জিনিসগুলো আরও কঠিন হয়ে গেছে।",
        "symptom_trend": {
            "improving": "{symptom} উন্নতি হচ্ছে",
            "stable": "{symptom} স্থিতিশীল",
            "worsening": "{symptom} খারাপ হচ্ছে"
        }
    },
    "ta-IN": {  # Tamil
        "welcome_back": "மீண்டும் வந்ததற்கு வரவேற்புகள். நாம் முன்பு பேசியதை நான் நினைவுகூருகிறேன்.",
        "empathy_worsening": "நிலைமைகள் மோசமாக மாறியதற்கு என் ஆழ்ச்சி.",
        "symptom_trend": {
            "improving": "{symptom} மேம்படுத்தப்படுகிறது",
            "stable": "{symptom} நிலையானது",
            "worsening": "{symptom} மோசமாகிறது"
        }
    },
    "te-IN": {  # Telugu
        "welcome_back": "మళ్ళీ రావడంతో స్వాగతం. మేము ముందు మాట్లాడినట్లు నేను గుర్తుంచుకుంటున్నాను.",
        "empathy_worsening": "పరిస్థితులు మరింత కష్టంగా మారాయినందున నాకు బాధగా ఉంది.",
        "symptom_trend": {
            "improving": "{symptom} మెరుగుపరచబడుతోంది",
            "stable": "{symptom} స్థిరంగా ఉంది",
            "worsening": "{symptom} దిగువకు దూసుకుంటోంది"
        }
    },
    "mr-IN": {  # Marathi
        "welcome_back": "पुन्हा आल्याबद्दल आपले स्वागत आहे. मला आठवते की आपण आधी बोललो होतात.",
        "empathy_worsening": "मला दुःख आहे की परिस्थिती अधिक कठीण झाली आहे.",
        "symptom_trend": {
            "improving": "{symptom} सुधारत आहे",
            "stable": "{symptom} स्थिर आहे",
            "worsening": "{symptom} वाईट होत आहे"
        }
    },
    "gu-IN": {  # Gujarati
        "welcome_back": "પાછા આવવા પર સ્વાગત છે. મને યાદ છે કે અમે પહેલા વાત કરી હતી.",
        "empathy_worsening": "મને દુઃખ છે કે બાબતો વધુ મુશ્કેલ બની ગઈ છે.",
        "symptom_trend": {
            "improving": "{symptom} સુધરી રહ્યું છે",
            "stable": "{symptom} સ્થિર છે",
            "worsening": "{symptom} બગડી રહ્યું છે"
        }
    },
    "kn-IN": {  # Kannada
        "welcome_back": "ಮರಳಿ ಬಂದಿರು ಸ್ವಾಗತ. ನಾನು ನೆನಪಿಸಿಕೊಳ್ಳುತ್ತೇನೆ ನಾವು ಹಿಂದೆ ಮಾತನಾಡಿದ್ದೆವು.",
        "empathy_worsening": "ಪರಿಸ್ಥಿತಿಗಳು ಹೆಚ್ಚು ಕಷ್ಟಕರವಾಗಿವೆ ಎಂದು ನನಗೆ ದುಃಖವಾಗುತ್ತಿದೆ.",
        "symptom_trend": {
            "improving": "{symptom} ಸುಧಾರಿಸುತ್ತಿದೆ",
            "stable": "{symptom} ಸ್ಥಿರವಾಗಿದೆ",
            "worsening": "{symptom} ಕೆಟ್ಟುತ್ತಿದೆ"
        }
    },
    "ml-IN": {  # Malayalam
        "welcome_back": "വീണ്ടും വന്നതിന് സ്വാഗതം. ഞാൻ ഓർക്കുന്നു ഞങ്ങൾ മുമ്പ് സംസാരിച്ചത്.",
        "empathy_worsening": "കാര്യങ്ങൾ കൂടുതൽ ബുദ്ധിമുട്ടായതിൽ എനിക്ക് ഖേദിപ്പെടുന്നു.",
        "symptom_trend": {
            "improving": "{symptom} മെച്ചപ്പെടുന്നു",
            "stable": "{symptom} സ്ഥിരമാണ്",
            "worsening": "{symptom} മോശമായി"
        }
    }
}


# Symptom name translations for multi-language support
SYMPTOM_TRANSLATIONS = {
    "pain": {
        "hi-IN": "दर्द",
        "bn-IN": "ব্যথা",
        "ta-IN": "வலி",
        "te-IN": "నొప్పి",
        "mr-IN": "दुखणे",
        "gu-IN": "પીડા",
        "kn-IN": "ನೋವು",
        "ml-IN": "വേദന"
    },
    "breathlessness": {
        "hi-IN": "सांस फूलना",
        "bn-IN": "শ্বাসকষ্ট",
        "ta-IN": "மூச்சம்",
        "te-IN": "శ్వాస తీవ్రత",
        "mr-IN": "श्वास घोट",
        "gu-IN": "શ્વાસ લેવું",
        "kn-IN": "ಉಸಿರುವಾಸ",
        "ml-IN": "ശ്വാസതടസ്സം"
    },
    "nausea": {
        "hi-IN": "मतली",
        "bn-IN": "বমি বমি ভাব",
        "ta-IN": "மயக்கம்",
        "te-IN": "వికారం",
        "mr-IN": "उलटी",
        "gu-IN": "ઉબકાઈ",
        "kn-IN": "ವಾಕರಿಕೆ",
        "ml-IN": "ഓക്കാനലി"
    },
    "fatigue": {
        "hi-IN": "थकान",
        "bn-IN": "ক্লান্তি",
        "ta-IN": "சோர்வு",
        "te-IN": "అలసత",
        "mr-IN": "थकवा",
        "gu-IN": "થાક",
        "kn-IN": "ಆಯಾಸ",
        "ml-IN": "ക്ഷീണം"
    },
    "constipation": {
        "hi-IN": "कब्ज",
        "bn-IN": "কোষ্ঠকাঠিন্য",
        "ta-IN": "மலச்சிக்கல்",
        "te-IN": "కన్స్టిపేషన్",
        "mr-IN": "कब्जीय",
        "gu-IN": "કબજિયા",
        "kn-IN": "ಅಜಿರ್ಣ",
        "ml-IN": "മലബന്ധം"
    },
    "anxiety": {
        "hi-IN": "चिंता",
        "bn-IN": "উদ্বেগ",
        "ta-IN": "பதற்றம்",
        "te-IN": "ఆందోళన",
        "mr-IN": "चिंता",
        "gu-IN": "ચિંતા",
        "kn-IN": "ಆತಂಕ",
        "ml-IN": "ഉത്കണ്ഠ"
    }
}


# ============================================================================
# CONTEXT INJECTOR
# ============================================================================

class ContextInjector:
    """
    Injects compassionate, personalized patient context into RAG queries.

    This module bridges the longitudinal memory system with the query pipeline,
    ensuring that responses are:
    - Personalized to the patient's history
    - Compassionate and empathetic in tone
    - Multi-language appropriate
    - Contextually aware of symptom trends, medications, and care team
    """

    def __init__(
        self,
        longitudinal_manager: LongitudinalMemoryManager,
        user_profile_manager: UserProfileManager,
        context_memory: ContextMemory
    ):
        """
        Initialize the context injector.

        Args:
            longitudinal_manager: Manager for 1-5 year records
            user_profile_manager: Manager for user profiles
            context_memory: Manager for 90-day current context
        """
        self.longitudinal = longitudinal_manager
        self.profiles = user_profile_manager
        self.context_memory = context_memory

        logger.info("ContextInjector initialized")

    async def inject_context(
        self,
        user_id: str,
        question: str = "",
        max_length: int = 2000
    ) -> str:
        """
        Generate compassionate, personalized context string for LLM prompt.

        Args:
            user_id: User/patient identifier
            question: The current question (for context relevance)
            max_length: Maximum length of context string

        Returns:
            Context string for injection into prompt, or empty string if no context
        """
        try:
            # Get user profile for language and preferences
            profile = await self.profiles.get_profile(user_id)
            if not profile:
                logger.debug(f"No profile found for {user_id}")
                return ""

            language = profile.preferences.language
            lang_code = language.split("-")[0] if "-" in language else language

            # Get current context (90-day)
            current_context = await self.context_memory.get_or_create_context(user_id)

            # Get longitudinal summary (30-day default for query context)
            longitudinal_summary = await self.longitudinal.get_longitudinal_summary(
                user_id,
                days=30
            )

            # Build compassionate context
            context_parts = []

            # 1. Welcome back for returning users (continuity)
            if profile.total_sessions > 1:
                welcome = self._get_template("welcome_back", language)
                if welcome:
                    context_parts.append(welcome)

            # 2. Primary condition
            if current_context.primary_condition:
                condition_text = self._format_condition(
                    current_context.primary_condition,
                    language
                )
                context_parts.append(condition_text)

            # 3. Recent symptom trends with empathy
            symptom_context = self._format_symptom_trends(
                longitudinal_summary,
                language,
                empathetic=True
            )
            if symptom_context:
                context_parts.append(symptom_context)

            # 4. Current medications (if any and recent discussion)
            if current_context.medications:
                med_context = self._format_medications(
                    current_context.medications,
                    language
                )
                if med_context:
                    context_parts.append(med_context)

            # 5. Active alerts (carefully phrased)
            active_alerts = longitudinal_summary.get("active_alerts", [])
            if active_alerts:
                alert_context = self._format_alerts_compassionately(
                    active_alerts,
                    language
                )
                if alert_context:
                    context_parts.append(alert_context)

            # 6. Care context (caregiver presence, location)
            care_context = self._format_care_context(
                current_context,
                language
            )
            if care_context:
                context_parts.append(care_context)

            # Combine and trim if needed
            full_context = " ".join(context_parts)

            if len(full_context) > max_length:
                # Keep most important parts, trim less critical
                full_context = self._trim_context_intelligently(
                    full_context,
                    max_length,
                    language
                )

            return full_context

        except Exception as e:
            logger.error(f"Error injecting context for {user_id}: {e}")
            return ""

    def _get_template(
        self,
        key: str,
        language: str,
        default: Optional[str] = None
    ) -> Optional[str]:
        """Get a language template, with fallback."""
        if language in COMPASSIONATE_TEMPLATES:
            templates = COMPASSIONATE_TEMPLATES[language]
        else:
            # Try to get from base language code
            lang_code = language.split("-")[0]
            templates = COMPASSIONATE_TEMPLATES.get(
                f"{lang_code}-IN",
                COMPASSIONATE_TEMPLATES["en-IN"]
            )

        # Navigate nested structure
        if "." in key:
            parts = key.split(".")
            value = templates
            for part in parts:
                value = value.get(part, {})
                if not value or isinstance(value, str):
                    break
            if isinstance(value, str):
                return value
        else:
            value = templates.get(key)
            if isinstance(value, list):
                # Pick first item from list, or could randomize
                return value[0] if value else None
            return value

        return default

    def _format_condition(self, condition: str, language: str) -> str:
        """Format primary condition."""
        templates = {
            "en-IN": f"Managing {condition}",
            "hi-IN": f"{condition} का प्रबंधन कर रहे हैं",
            "bn-IN": f"{condition} পরিচালনা করছেন"
        }

        if language in templates:
            return templates[language]
        elif "-" in language:
            lang_base = language.split("-")[0]
            key = f"{lang_base}-IN"
            if key in templates:
                return templates[key]

        return templates["en-IN"]

    def _format_symptom_trends(
        self,
        summary: Dict[str, Any],
        language: str,
        empathetic: bool = True
    ) -> Optional[str]:
        """Format symptom trends with appropriate tone."""
        trend_summaries = summary.get("summaries", {})
        active_symptoms = summary.get("active_symptoms", [])

        if not trend_summaries and not active_symptoms:
            return None

        context_parts = []

        # Process trending symptoms
        for key, trend_data in trend_summaries.items():
            if "symptom" not in key:
                continue

            entity_name = trend_data.get("entity_name", "")
            trend = trend_data.get("trend", "unknown")

            # Get localized symptom name
            localized_name = self._localize_symptom(entity_name, language)

            # Format trend based on direction
            if empathetic and trend == "worsening":
                empathy_phrase = self._get_template("empathy_worsening", language, "")
                if empathy_phrase:
                    context_parts.append(empathy_phrase)

            # Add trend information
            trend_template = self._get_template("symptom_trend", language)
            if trend_template:
                if isinstance(trend_template, dict):
                    trend_phrase = trend_template.get(trend, trend_template.get("unknown", ""))
                else:
                    trend_phrase = trend_template

                # Use localized symptom name
                context_parts.append(trend_template.get(trend, "").format(symptom=localized_name))

        # Also mention active symptoms without trend data
        for symptom_data in active_symptoms:
            name = symptom_data.get("name", "")
            if name and not any(f"symptom:{name}" in key for key in trend_summaries.keys()):
                localized_name = self._localize_symptom(name, language)
                severity = symptom_data.get("severity", 2)
                if severity >= 3:  # Only mention severe symptoms
                    if language == "hi-IN":
                        context_parts.append(f"{localized_name} की समस्या है")
                    else:
                        context_parts.append(f"experiencing {localized_name}")

        if context_parts:
            lang_base = language.split("-")[0]
            if lang_base == "hi":
                return " ".join(context_parts)
            return ". ".join(context_parts) + "."

        return None

    def _format_medications(
        self,
        medications: List[Any],
        language: str
    ) -> Optional[str]:
        """Format current medications."""
        if not medications:
            return None

        med_names = [m.name if hasattr(m, 'name') else str(m) for m in medications[:3]]

        if language == "hi-IN":
            return f"वर्तमान में {', '.join(med_names)} ले रहे हैं"
        elif language.startswith("hi"):
            return f"वर्तमान में {', '.join(med_names)} ले रहे हैं"
        else:
            return f"Currently taking {', '.join(med_names)}"

    def _format_alerts_compassionately(
        self,
        alerts: List[Dict[str, Any]],
        language: str
    ) -> Optional[str]:
        """Format alerts with compassionate delivery."""
        if not alerts:
            return None

        # Only include HIGH and URGENT alerts
        urgent_alerts = [
            a for a in alerts
            if a.get("priority") in ["high", "urgent"]
        ]

        if not urgent_alerts:
            return None

        # Get the most urgent alert
        alert = urgent_alerts[0]
        title = alert.get("title", "")
        description = alert.get("description", "")

        # For alerts, we want to be direct but kind
        if language == "hi-IN":
            return f"ध्यान दें: {description[:100]}"
        else:
            return f"Please note: {description[:150]}"

    def _format_care_context(
        self,
        context: Any,
        language: str
    ) -> Optional[str]:
        """Format care context (caregiver, location)."""
        parts = []

        if context.has_caregiver:
            if language == "hi-IN":
                parts.append("आपके साथ आपके परिवार के सदस्य भी हैं")
            else:
                parts.append("has family support")

        if context.care_location and context.care_location != "home":
            location_map = {
                "hospital": "in hospital",
                "hospice": "in hospice care",
                "clinic": "at clinic"
            }
            location_text = location_map.get(context.care_location, context.care_location)
            parts.append(location_text)

        if parts:
            if language == "hi-IN":
                return " ".join(parts)
            return " and ".join(parts)

        return None

    def _localize_symptom(self, symptom: str, language: str) -> str:
        """Get localized symptom name."""
        symptom_lower = symptom.lower()

        if language in SYMPTOM_TRANSLATIONS:
            for key, translation in SYMPTOM_TRANSLATIONS[language].items():
                if key in symptom_lower:
                    return translation

        # Fallback: check if symptom contains a known key
        for key, translations in SYMPTOM_TRANSLATIONS.items():
            if key in symptom_lower:
                return translations.get(language, symptom)

        return symptom

    def _trim_context_intelligently(
        self,
        context: str,
        max_length: int,
        language: str
    ) -> str:
        """Intelligently trim context to fit length limit."""
        if len(context) <= max_length:
            return context

        # Priority: keep alert info > symptom trends > medications > general
        # This is a simplified version - could be more sophisticated

        # Split by sentences and prioritize
        sentences = []
        if language.startswith("hi"):
            # For Hindi, split by vowel-based sentence boundaries
            import re
            sentences = re.split(r'[।॥.]\s*', context)
        else:
            import re
            sentences = re.split(r'[.!?]\s+', context)

        # Keep first few sentences (usually most important)
        result = ""
        for sent in sentences:
            if len(result) + len(sent) + 2 <= max_length:
                result += sent + (". " if language != "hi-IN" else "। ")
            else:
                break

        return result.strip()

    async def get_context_for_monitoring(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed context for monitoring/alerting purposes.

        Returns more detailed information than query context.
        """
        try:
            longitudinal_summary = await self.longitudinal.get_longitudinal_summary(
                user_id,
                days=90
            )

            current_context = await self.context_memory.get_or_create_context(user_id)
            profile = await self.profiles.get_profile(user_id)

            return {
                "user_id": user_id,
                "profile": {
                    "role": profile.role.value if profile else "unknown",
                    "language": profile.preferences.language if profile else "en-IN",
                    "total_sessions": profile.total_sessions if profile else 0,
                    "last_interaction": profile.last_interaction.isoformat() if profile else None
                },
                "current_condition": {
                    "primary": current_context.primary_condition,
                    "stage": current_context.condition_stage,
                    "symptoms": [s.name for s in current_context.symptoms],
                    "medications": [m.name for m in current_context.medications]
                },
                "longitudinal": longitudinal_summary
            }
        except Exception as e:
            logger.error(f"Error getting monitoring context for {user_id}: {e}")
            return {}


# ============================================================================
# CONTEXT BUILDER FOR LLM PROMPTS
# ============================================================================

class PromptContextBuilder:
    """
    Builds complete LLM prompts with patient context.

    This class handles the integration of patient context into the actual
    prompt sent to the LLM, ensuring appropriate tone and structure.
    """

    def __init__(self, context_injector: ContextInjector):
        """
        Initialize the prompt context builder.

        Args:
            context_injector: The context injector instance
        """
        self.injector = context_injector

    async def build_prompt_with_context(
        self,
        base_system_prompt: str,
        user_id: str,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build a complete system prompt with patient context.

        Args:
            base_system_prompt: The base system prompt
            user_id: Patient identifier
            question: The current question
            conversation_history: Recent conversation turns

        Returns:
            Enhanced system prompt with patient context
        """
        # Get patient context
        patient_context = await self.injector.inject_context(user_id, question)

        # Build enhanced prompt
        if not patient_context:
            return base_system_prompt

        # Insert context into system prompt
        enhanced_prompt = f"""{base_system_prompt}

## Patient Context
{patient_context}

Remember to:
- Respond with compassion and empathy
- Acknowledge the patient's history and current situation
- Provide practical, actionable advice
- Offer emotional support when needed
- If something seems concerning, suggest consulting their healthcare team
"""

        return enhanced_prompt

    async def build_user_context_summary(
        self,
        user_id: str
    ) -> str:
        """
        Build a concise summary of user context for display purposes.

        Useful for dashboards, caregiver views, etc.
        """
        context = await self.injector.get_context_for_monitoring(user_id)

        profile = context.get("profile", {})
        current = context.get("current_condition", {})
        longitudinal = context.get("longitudinal", {})

        summary_parts = []

        # Basic info
        total_sessions = profile.get("total_sessions", 0)
        if total_sessions > 5:
            summary_parts.append(f"Returning patient ({total_sessions} sessions)")

        # Condition
        if current.get("primary"):
            summary_parts.append(f"Condition: {current['primary']}")
            if current.get("stage"):
                summary_parts.append(f"Stage: {current['stage']}")

        # Recent symptoms
        active_symptoms = longitudinal.get("active_symptoms", [])
        if active_symptoms:
            symptom_names = [s["name"] for s in active_symptoms[:3]]
            summary_parts.append(f"Active symptoms: {', '.join(symptom_names)}")

        # Alerts
        alerts = longitudinal.get("active_alerts", [])
        if alerts:
            summary_parts.append(f"{len(alerts)} active alert(s)")

        return " | ".join(summary_parts) if summary_parts else "New patient"
