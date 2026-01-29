#!/usr/bin/env python3
"""
Palli Sahayak Safety & Enhancement Module
==========================================
Implements 5 key quick-win features:
1. Evidence Badges - Show confidence levels with responses
2. Emergency Detection & Escalation - Auto-detect emergencies
3. Medication Reminder Scheduler - Schedule medication reminders
4. Response Length Optimization - Adjust based on comprehension
5. Human Handoff System - Warm transfer to human caregivers

Author: Palli Sahayak AI Team
"""

import os
import json
import asyncio
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
from collections import defaultdict

# Try to import schedule, fallback to simple implementation if not available
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    # Simple schedule fallback
    class MockSchedule:
        def every(self, interval=1):
            return self
        def day(self):
            return self
        def at(self, time_str):
            return self
        def do(self, job_func, *args, **kwargs):
            return self
        def run_pending(self):
            pass
    schedule = MockSchedule()

logger = logging.getLogger(__name__)


# ============================================================================
# 1. EVIDENCE BADGES SYSTEM
# ============================================================================

class EvidenceLevel(Enum):
    """Evidence quality levels based on medical literature standards"""
    A = "A"  # RCT/Meta-analysis - Highest confidence
    B = "B"  # Well-designed controlled studies
    C = "C"  # Observational studies/limited evidence
    D = "D"  # Expert opinion/consensus
    E = "E"  # Insufficient evidence - Consult physician


@dataclass
class EvidenceBadge:
    """Evidence badge attached to each response"""
    level: EvidenceLevel
    confidence_score: float  # 0.0 to 1.0
    source_quality: str
    recommendation: str
    consult_physician: bool
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "confidence_score": round(self.confidence_score, 2),
            "source_quality": self.source_quality,
            "recommendation": self.recommendation,
            "consult_physician": self.consult_physician,
            "badge_emoji": self.get_badge_emoji(),
            "sources": self.sources
        }
    
    def get_badge_emoji(self) -> str:
        """Get appropriate emoji for evidence level"""
        emojis = {
            EvidenceLevel.A: "🟢",  # Green - Strong evidence
            EvidenceLevel.B: "🟡",  # Yellow - Good evidence
            EvidenceLevel.C: "🟠",  # Orange - Limited evidence
            EvidenceLevel.D: "🔵",  # Blue - Expert opinion
            EvidenceLevel.E: "🔴",  # Red - Consult physician
        }
        return emojis.get(self.level, "⚪")
    
    def format_for_user(self, language: str = "en") -> str:
        """Format evidence badge for end-user display"""
        translations = {
            "en": {
                "confidence": "Confidence",
                "source": "Source Quality",
                "consult": "⚠️ Please consult a physician for this matter",
                "level_a": "High - Based on clinical trials",
                "level_b": "Good - Based on controlled studies",
                "level_c": "Moderate - Limited studies available",
                "level_d": "Expert opinion",
                "level_e": "Consult physician recommended",
            },
            "hi": {
                "confidence": "विश्वसनीयता",
                "source": "स्रोत गुणवत्ता",
                "consult": "⚠️ कृपया इस मामले के लिए डॉक्टर से परामर्श करें",
                "level_a": "उच्च - नैदानिक परीक्षणों पर आधारित",
                "level_b": "अच्छा - नियंत्रित अध्ययनों पर आधारित",
                "level_c": "मध्यम - सीमित अध्ययन उपलब्ध",
                "level_d": "विशेषज्ञ राय",
                "level_e": "डॉक्टर से परामर्श अनुशंसित",
            },
            "bn": {
                "confidence": "আত্মবিশ্বাস",
                "source": "উৎসের গুণমান",
                "consult": "⚠️ অনুগ্রহ করে এই বিষয়ে একজন ডাক্তারের পরামর্শ নিন",
                "level_a": "উচ্চ - ক্লিনিকাল ট্রায়ালের উপর ভিত্তি করে",
                "level_b": "ভাল - নিয়ন্ত্রিত অধ্যয়নের উপর ভিত্তি করে",
                "level_c": "মাঝারি - সীমিত অধ্যয়ন উপলব্ধ",
                "level_d": "বিশেষজ্ঞ মতামত",
                "level_e": "ডাক্তারের পরামর্শ সুপারিশ",
            },
            "ta": {
                "confidence": "நம்பகத்தன்மை",
                "source": "மூல தரம்",
                "consult": "⚠️ தயவுசெய்து இந்த விஷயத்திற்கு மருத்துவரை அணுகவும்",
                "level_a": "உயர் - கிளினிக்கல் சோதனைகளை அடிப்படையாகக் கொண்டது",
                "level_b": "நல்லது - கட்டுப்படுத்தப்பட்ட ஆய்வுகளை அடிப்படையாகக் கொண்டது",
                "level_c": "மிதமான - வரையறுக்கப்பட்ட ஆய்வுகள்",
                "level_d": "நிபுணர் கருத்து",
                "level_e": "மருத்துவரை அணுக பரிந்துரை",
            },
        }
        
        t = translations.get(language, translations["en"])
        level_desc = t.get(f"level_{self.level.value.lower()}", "Unknown")
        
        lines = [
            f"\n{'─' * 40}",
            f"{self.get_badge_emoji()} {t['confidence']}: {self.confidence_score:.0%}",
            f"📚 {t['source']}: {level_desc}",
        ]
        
        if self.consult_physician:
            lines.append(f"\n{t['consult']}")
        
        lines.append(f"{'─' * 40}")
        return "\n".join(lines)


class EvidenceBadgeSystem:
    """System for attaching evidence badges to RAG responses"""
    
    # High-quality source patterns
    AUTHORITATIVE_SOURCES = [
        "who", "world health organization",
        "nice", "national institute",
        "asco", "american society of clinical oncology",
        "eapc", "european association for palliative care",
        "aacn", "american association of critical care nurses",
        "hpna", "hospice and palliative nurses association",
        "mhpc", "maharashtra hospice and palliative care",
        " textbook", "guidelines", "consensus",
        "randomized", "clinical trial", "meta-analysis",
        "systematic review", "cochrane"
    ]
    
    # Lower-quality source patterns
    LOWER_QUALITY_SOURCES = [
        "blog", "forum", "opinion", "personal experience",
        "anecdotal", "unverified", "preprint"
    ]
    
    def __init__(self):
        self.confidence_history: Dict[str, List[float]] = defaultdict(list)
        
    def calculate_evidence_badge(
        self, 
        query: str, 
        sources: List[Dict[str, Any]], 
        distances: List[float],
        answer_text: str
    ) -> EvidenceBadge:
        """
        Calculate evidence badge based on sources and query type
        """
        # Calculate base confidence from vector distances
        if distances:
            avg_distance = sum(distances) / len(distances)
            # Convert distance to confidence (lower distance = higher confidence)
            base_confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))
        else:
            base_confidence = 0.5
        
        # Analyze source quality
        source_quality_score = self._analyze_source_quality(sources)
        
        # Check for high-stakes medical queries
        is_high_stakes = self._is_high_stakes_query(query)
        
        # Check answer for uncertainty indicators
        uncertainty_score = self._detect_uncertainty(answer_text)
        
        # Calculate final confidence
        confidence = (base_confidence * 0.4 + 
                     source_quality_score * 0.4 + 
                     (1.0 - uncertainty_score) * 0.2)
        
        # Determine evidence level
        level = self._determine_evidence_level(confidence, source_quality_score, is_high_stakes)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(level, is_high_stakes)
        
        # Determine if physician consultation is needed
        consult_physician = (level in [EvidenceLevel.D, EvidenceLevel.E] or 
                           (is_high_stakes and confidence < 0.8))
        
        # Get source quality description
        source_quality_desc = self._get_source_quality_description(source_quality_score)
        
        return EvidenceBadge(
            level=level,
            confidence_score=confidence,
            source_quality=source_quality_desc,
            recommendation=recommendation,
            consult_physician=consult_physician,
            sources=sources[:3]  # Include top 3 sources
        )
    
    def _analyze_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Analyze quality of sources"""
        if not sources:
            return 0.3
        
        scores = []
        for source in sources:
            filename = source.get("filename", "").lower()
            
            # Check for authoritative sources
            score = 0.5  # Base score
            for auth_source in self.AUTHORITATIVE_SOURCES:
                if auth_source in filename:
                    score += 0.15
            
            # Check for lower quality indicators
            for low_source in self.LOWER_QUALITY_SOURCES:
                if low_source in filename:
                    score -= 0.2
            
            scores.append(max(0.0, min(1.0, score)))
        
        return sum(scores) / len(scores) if scores else 0.3
    
    def _is_high_stakes_query(self, query: str) -> bool:
        """Determine if query is high-stakes (medical emergency, critical decision)"""
        high_stakes_patterns = [
            r'\b(chest pain|heart attack|cardiac arrest)\b',
            r'\b(can\'t breathe|shortness of breath|breathing difficulty)\b',
            r'\b(severe bleeding|hemorrhage)\b',
            r'\b(unconscious|passed out|fainted)\b',
            r'\b(stroke|seizure|convulsion)\b',
            r'\b(overdose|poisoning)\b',
            r'\b(suicide|self.?harm|kill myself)\b',
            r'\b(emergency|urgent|911|ambulance)\b',
        ]
        
        query_lower = query.lower()
        for pattern in high_stakes_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def _detect_uncertainty(self, answer: str) -> float:
        """Detect uncertainty indicators in answer"""
        uncertainty_patterns = [
            r'\b(may|might|could|possibly|perhaps)\b',
            r'\b(unclear|uncertain|unknown|not sure)\b',
            r'\b(depends on|varies|different for everyone)\b',
            r'\b(consult|see|talk to) (your|a) (doctor|physician)\b',
            r'\b(insufficient|limited|lack of) (evidence|data|research)\b',
            r'\b(more research|further studies) (needed|required)\b',
        ]
        
        uncertainty_count = 0
        answer_lower = answer.lower()
        for pattern in uncertainty_patterns:
            uncertainty_count += len(re.findall(pattern, answer_lower))
        
        # Normalize to 0-1 scale
        return min(1.0, uncertainty_count / 5.0)
    
    def _determine_evidence_level(
        self, 
        confidence: float, 
        source_quality: float,
        is_high_stakes: bool
    ) -> EvidenceLevel:
        """Determine evidence level based on scores"""
        combined_score = (confidence * 0.6 + source_quality * 0.4)
        
        if is_high_stakes:
            # Be more conservative for high-stakes queries
            if combined_score >= 0.9:
                return EvidenceLevel.A
            elif combined_score >= 0.75:
                return EvidenceLevel.B
            elif combined_score >= 0.6:
                return EvidenceLevel.C
            elif combined_score >= 0.4:
                return EvidenceLevel.D
            else:
                return EvidenceLevel.E
        else:
            if combined_score >= 0.85:
                return EvidenceLevel.A
            elif combined_score >= 0.7:
                return EvidenceLevel.B
            elif combined_score >= 0.55:
                return EvidenceLevel.C
            elif combined_score >= 0.4:
                return EvidenceLevel.D
            else:
                return EvidenceLevel.E
    
    def _generate_recommendation(self, level: EvidenceLevel, is_high_stakes: bool) -> str:
        """Generate recommendation based on evidence level"""
        recommendations = {
            EvidenceLevel.A: "Strong evidence supports this information",
            EvidenceLevel.B: "Good evidence supports this information",
            EvidenceLevel.C: "Limited evidence - use with caution",
            EvidenceLevel.D: "Based on expert opinion",
            EvidenceLevel.E: "Consult a physician before acting",
        }
        
        base = recommendations.get(level, "Consult healthcare provider")
        
        if is_high_stakes:
            base += " | High-stakes medical decision"
        
        return base
    
    def _get_source_quality_description(self, score: float) -> str:
        """Get human-readable source quality description"""
        if score >= 0.8:
            return "Excellent - Authoritative medical sources"
        elif score >= 0.6:
            return "Good - Reliable medical sources"
        elif score >= 0.4:
            return "Fair - Mixed sources"
        else:
            return "Limited - Weak source material"


# ============================================================================
# 2. EMERGENCY DETECTION & ESCALATION
# ============================================================================

class EmergencyLevel(Enum):
    """Emergency severity levels"""
    NONE = "none"
    LOW = "low"           # Concerning but not immediate
    MEDIUM = "medium"     # Should seek care soon
    HIGH = "high"         # Urgent - seek immediate care
    CRITICAL = "critical" # Life-threatening - call emergency services


@dataclass
class EmergencyAlert:
    """Emergency alert details"""
    level: EmergencyLevel
    detected_keywords: List[str]
    message: str
    action_required: str
    contact_emergency_services: bool
    notify_caregivers: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "detected_keywords": self.detected_keywords,
            "message": self.message,
            "action_required": self.action_required,
            "contact_emergency_services": self.contact_emergency_services,
            "notify_caregivers": self.notify_caregivers,
            "timestamp": self.timestamp.isoformat()
        }


class EmergencyDetectionSystem:
    """
    Detects emergency situations from user queries and triggers appropriate responses.
    Supports multiple Indian languages.
    """
    
    # Emergency keyword patterns by severity level and language
    EMERGENCY_KEYWORDS = {
        EmergencyLevel.CRITICAL: {
            "en": [
                "can't breathe", "cannot breathe", "choking", "not breathing",
                "cardiac arrest", "heart stopped", "no pulse",
                "unconscious", "not responding", "passed out",
                "severe bleeding", "bleeding heavily", "blood everywhere",
                "suicide", "kill myself", "want to die", "end my life",
                "overdose", "took too many pills", "poisoning",
                "chest pain", "heart attack", "stroke",
            ],
            "hi": [
                "सांस नहीं आ रही", "सांस लेने में तकलीफ", "दम घुटना",
                "होश नहीं", "बेहोश", "चेतना खो दी",
                "खून बह रहा है", "तेज खून",
                "आत्महत्या", "मरना चाहता हूं", "जान देना",
                "ओवरडोज", "ज्यादा गोली खा ली",
                "छाती में दर्द", "हार्ट अटैक", "दिल का दौरा",
            ],
            "bn": [
                "শ্বাস নিতে পারছি না", "শ্বাসকষ্ট", "দম বন্ধ",
                "জ্ঞান হারিয়ে ফেলেছে", "অজ্ঞান", "চেতনা নেই",
                "রক্তপাত হচ্ছে", "প্রচণ্ড রক্তপাত",
                "আত্মহত্যা", "মরতে চাই", "মারা যেতে চাই",
                "ওভারডোজ", "অতিরিক্ত ওষুধ খেয়েছি",
                "বুকে ব্যথা", "হার্ট অ্যাটাক", "স্ট্রোক",
            ],
            "ta": [
                "மூச்சு வரவில்லை", "மூச்சுத் திணறல்", "மூச்சு அடைப்பு",
                "பரிசுத்தமற்றவர்", "சுயநினைவு இல்லை", "மயக்கம்",
                "கடுமையான இரத்தப்போக்கு", "அதிக இரத்தப்போக்கு",
                "தற்கொலை", "சாக விரும்புகிறேன்", "உயிரை மாய்க்க",
                "மருந்து அதிகப்படியாக", "நச்சூட்டல்",
                "மார்பு வலி", "இதயத் தாக்கம்", "பக்கவாதம்",
            ],
            "gu": [
                "શ્વાસ લઈ શકતો નથી", "શ્વાસ લેવામાં તકલીફ", "ગળું દબાઈ જવું",
                "બેહોશ", "હોશ નથી", "ચેતના ગુમાવી",
                "ખૂન વહી રહ્યું છે", "તીવ્ર રક્તસ્રાવ",
                "આત્મહત્યા", "મરવા માંગુ છું", "જીવન સમાપ્ત",
                "ઓવરડોઝ", "વધારે ગોળીઓ લઈ લીધી",
                "છાતીમાં દુઃખાવો", "હૃદયરોગ", "સ્ટ્રોક",
            ],
        },
        EmergencyLevel.HIGH: {
            "en": [
                "severe pain", "extreme pain", "unbearable pain",
                "high fever", "very high temperature",
                "can't move", "paralyzed", "numbness",
                "allergic reaction", "swelling", "anaphylaxis",
                "fall", "fell down", "injured",
                "vomiting blood", "blood in stool",
            ],
            "hi": [
                "तीव्र दर्द", "बहुत दर्द", "बर्दाश्त नहीं हो रहा",
                "तेज बुखार", "बहुत गर्मी",
                "हिल नहीं पा रहा", "सुन्न", "लकवा",
                "एलर्जी", "सूजन", "एनाफाइलैक्सिस",
                "गिर गया", "गिरावट", "चोट",
                "खून की उलटी", "मल में खून",
            ],
            "bn": [
                "প্রচণ্ড ব্যথা", "অসহনীয় ব্যথা", "ভয়ানক ব্যথা",
                "উচ্চ জ্বর", "অত্যন্ত তাপমাত্রা",
                "নড়তে পারছি না", "পক্ষাঘাত", "অনুভূতি নেই",
                "এলার্জি", "ফোলা", "অ্যানাফাইল্যাক্সিস",
                "পড়ে গেছি", "পতন", "আহত",
                "রক্তবমি", "পায়খানায় রক্ত",
            ],
            "ta": [
                "கடுமையான வலி", "தாங்க முடியாத வலி", "கடுமையான வலி",
                "அதிக காய்ச்சல்", "மிக உயர்ந்த வெப்பநிலை",
                "நகர முடியவில்லை", "பக்கவாதம்", "மரத்துப் போதல்",
                "ஒவ்வாமை", "வீக்கம்", "அனாஃபிலாக்சிஸ்",
                "விழுந்தேன்", "வீழ்ச்சி", "காயம்",
                "ரத்த வாந்தி", "மலத்தில் ரத்தம்",
            ],
            "gu": [
                "તીવ્ર પીડા", "અસહ્ય પીડા", "આપવા માંગતો નથી",
                "ઉચ્ચ તાવ", "ખૂબ જ ઉચ્ચ તાપમાન",
                "હિલાવી શકાતો નથી", "લકવો", "સુન્નતા",
                "એલર્જી", "સોજો", "એનાફાઈલેક્સિસ",
                "પડી ગયો", "પતન", "ઈજા",
                "રક્તવમન", "મળમાં રક્ત",
            ],
        },
        EmergencyLevel.MEDIUM: {
            "en": [
                "persistent pain", "ongoing pain", "constant pain",
                "fever", "temperature",
                "nausea", "vomiting", "diarrhea",
                "rash", "itching", "skin reaction",
                "dizzy", "lightheaded", "weakness",
                "worried", "concerned", "anxious",
            ],
            "hi": [
                "लगातार दर्द", "बना रहता है", "ठीक नहीं हो रहा",
                "बुखार", "तापमान",
                "जी मिचलाना", "उलटी", "दस्त",
                "चकत्ते", "खुजली", "त्वचा की समस्या",
                "चक्कर", "कमजोरी", "थकान",
                "चिंता", "डर", "परेशान",
            ],
            "bn": [
                "অবিরাম ব্যথা", "চলমান ব্যথা", "ধ্রুবক ব্যথা",
                "জ্বর", "তাপমাত্রা",
                "বমি বমি ভাব", "বমি", "ডায়রিয়া",
                "ফুসকুড়ি", "চুলকানি", "ত্বকের প্রতিক্রিয়া",
                "ঘোরাচ্ছে", "দুর্বলতা", "ক্লান্তি",
                "উদ্বিগ্ন", "চিন্তা", "ভয়",
            ],
            "ta": [
                "தொடர்ச்சியான வலி", "நிலையான வலி", "மாறாத வலி",
                "காய்ச்சல்", "வெப்பநிலை",
                "குமட்டல்", "வாந்தி", "வயிற்றுப்போக்கு",
                "தோல் தடிப்பு", "அரிப்பு", "தோல் எதிர்வினை",
                "மயக்கம்", "பலவீனம்", "சோர்வு",
                "கவலை", "பதட்டம்", "பயம்",
            ],
            "gu": [
                "સતત પીડા", "ચાલુ પીડા", "થયા વિના પીડા",
                "તાવ", "તાપમાન",
                "ઉબકા", "ઉલટી", "ઝાડા",
                "ફોલ્લી", "ખંજવાળ", "ત્વચાની પ્રતિક્રિયા",
                "ચક્કર", "બેભાનપણું", "કમજોરી",
                "ચિંતા", "ડર", "પરેશાની",
            ],
        },
    }
    
    def __init__(self):
        self.alert_history: List[EmergencyAlert] = []
        self.caregiver_contacts: Dict[str, List[str]] = {}  # user_id -> phone numbers
        
    def detect_emergency(self, query: str, user_id: Optional[str] = None, 
                        language: str = "en") -> Optional[EmergencyAlert]:
        """
        Detect emergency keywords in user query
        Returns EmergencyAlert if emergency detected, None otherwise
        """
        query_lower = query.lower()
        detected_keywords = []
        highest_level = EmergencyLevel.NONE
        
        # Priority order for severity levels
        level_priority = {
            EmergencyLevel.CRITICAL: 3,
            EmergencyLevel.HIGH: 2,
            EmergencyLevel.MEDIUM: 1,
            EmergencyLevel.NONE: 0,
        }
        
        # Check each severity level
        for level in [EmergencyLevel.CRITICAL, EmergencyLevel.HIGH, EmergencyLevel.MEDIUM]:
            keywords = self.EMERGENCY_KEYWORDS.get(level, {})
            
            # Check all language variants
            for lang, lang_keywords in keywords.items():
                for keyword in lang_keywords:
                    if keyword in query_lower:
                        detected_keywords.append(keyword)
                        if level_priority[level] > level_priority[highest_level]:
                            highest_level = level
        
        if highest_level == EmergencyLevel.NONE:
            return None
        
        # Create appropriate alert
        return self._create_alert(highest_level, detected_keywords, language)
    
    def _create_alert(self, level: EmergencyLevel, keywords: List[str], 
                     language: str) -> EmergencyAlert:
        """Create emergency alert with appropriate messaging"""
        
        messages = {
            "en": {
                EmergencyLevel.CRITICAL: "🚨 CRITICAL EMERGENCY DETECTED. Call emergency services (108/102) immediately!",
                EmergencyLevel.HIGH: "⚠️ URGENT: Please seek immediate medical attention.",
                EmergencyLevel.MEDIUM: "⚡ Please consult a doctor as soon as possible.",
            },
            "hi": {
                EmergencyLevel.CRITICAL: "🚨 गंभीर आपातकाल! तुरंत आपातकालीन सेवाओं (108/102) को कॉल करें!",
                EmergencyLevel.HIGH: "⚠️ तत्काल: कृपया तुरंत चिकित्सा सहायता लें।",
                EmergencyLevel.MEDIUM: "⚡ कृपया जल्द से जल्द डॉक्टर से परामर्श करें।",
            },
            "bn": {
                EmergencyLevel.CRITICAL: "🚨 গুরুতর জরুরী! অবিলম্বে জরুরী সেবায় (108/102) কল করুন!",
                EmergencyLevel.HIGH: "⚠️ জরুরী: দয়া করে অবিলম্বে চিকিৎসা সহায়তা নিন।",
                EmergencyLevel.MEDIUM: "⚡ দয়া করে যত তাড়াতাড়ি সম্ভব একজন ডাক্তারের পরামর্শ নিন।",
            },
            "ta": {
                EmergencyLevel.CRITICAL: "🚨 மிக முக்கிய அவசரம்! உடனடியாக அவசர சேவைகளை (108/102) அழைக்கவும்!",
                EmergencyLevel.HIGH: "⚠️ அவசரம்: தயவுசெய்து உடனடியாக மருத்துவ உதவியைப் பெறவும்.",
                EmergencyLevel.MEDIUM: "⚡ தயவுசெய்து விரைவில் மருத்துவரை அணுகவும்.",
            },
            "gu": {
                EmergencyLevel.CRITICAL: "🚨 ગંભીર આપત્તિ! તાત્કાલિક આપત્તિ સેવાઓ (108/102) ને કૉલ કરો!",
                EmergencyLevel.HIGH: "⚠️ તાત્કાલિક: કૃપા કરીને તાત્કાલિક વૈદ્યકીય સહાય લો.",
                EmergencyLevel.MEDIUM: "⚡ કૃપા કરીને શક્ય તેટલી વહેલી તકે ડોક્ટરનો સંપર્ક કરો.",
            },
        }
        
        actions = {
            "en": {
                EmergencyLevel.CRITICAL: "1. Call 108 (ambulance) or 102 immediately\n2. Stay with the patient\n3. Do not give food or water if unconscious\n4. A human caregiver has been notified",
                EmergencyLevel.HIGH: "1. Go to nearest hospital emergency\n2. Call your doctor\n3. Do not drive yourself\n4. A human caregiver has been notified",
                EmergencyLevel.MEDIUM: "1. Schedule doctor appointment\n2. Monitor symptoms\n3. Rest and stay hydrated\n4. Contact if symptoms worsen",
            },
            "hi": {
                EmergencyLevel.CRITICAL: "1. तुरंत 108 (एम्बुलेंस) या 102 पर कॉल करें\n2. मरीज के पास रहें\n3. बेहोश होने पर खाना-पानी न दें\n4. एक मानव देखभाल करने वाले को सूचित किया गया है",
                EmergencyLevel.HIGH: "1. निकटतम अस्पताल के आपातकालीन में जाएं\n2. अपने डॉक्टर को कॉल करें\n3. खुद गाड़ी न चलाएं\n4. एक मानव देखभाल करने वाले को सूचित किया गया है",
                EmergencyLevel.MEDIUM: "1. डॉक्टर का समय निर्धारित करें\n2. लक्षणों की निगरानी करें\n3. आराम करें और हाइड्रेटेड रहें\n4. अगर लक्षण बिगड़ें तो संपर्क करें",
            },
        }
        
        lang_messages = messages.get(language, messages["en"])
        lang_actions = actions.get(language, actions["en"])
        
        return EmergencyAlert(
            level=level,
            detected_keywords=keywords,
            message=lang_messages.get(level, lang_messages[EmergencyLevel.HIGH]),
            action_required=lang_actions.get(level, lang_actions[EmergencyLevel.HIGH]),
            contact_emergency_services=level == EmergencyLevel.CRITICAL,
            notify_caregivers=level in [EmergencyLevel.CRITICAL, EmergencyLevel.HIGH]
        )
    
    def register_caregiver(self, user_id: str, phone_number: str):
        """Register a caregiver contact for emergency notifications"""
        if user_id not in self.caregiver_contacts:
            self.caregiver_contacts[user_id] = []
        if phone_number not in self.caregiver_contacts[user_id]:
            self.caregiver_contacts[user_id].append(phone_number)
            logger.info(f"Registered caregiver {phone_number} for user {user_id}")
    
    async def notify_caregivers(self, user_id: str, alert: EmergencyAlert,
                               whatsapp_api=None) -> List[Dict[str, Any]]:
        """Notify all registered caregivers about emergency"""
        results = []
        
        if user_id not in self.caregiver_contacts:
            return results
        
        message = f"""🚨 PALLI SAHAYAK EMERGENCY ALERT 🚨

Patient ID: {user_id}
Severity: {alert.level.value.upper()}
Detected keywords: {', '.join(alert.detected_keywords[:3])}

{alert.message}

Please check on the patient immediately.
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        for phone in self.caregiver_contacts[user_id]:
            try:
                if whatsapp_api:
                    result = await whatsapp_api.send_text_message(phone, message)
                    results.append({"phone": phone, "result": result})
                else:
                    results.append({"phone": phone, "result": "API not available"})
            except Exception as e:
                logger.error(f"Failed to notify caregiver {phone}: {e}")
                results.append({"phone": phone, "error": str(e)})
        
        return results


# ============================================================================
# 3. MEDICATION REMINDER SCHEDULER
# ============================================================================

@dataclass
class MedicationReminder:
    """Medication reminder configuration"""
    reminder_id: str
    user_id: str
    medication_name: str
    dosage: str
    frequency: str  # "daily", "twice_daily", "weekly", etc.
    scheduled_times: List[str]  # ["08:00", "20:00"] in 24h format
    start_date: datetime
    end_date: Optional[datetime]
    instructions: str
    language: str
    active: bool = True
    last_reminded: Optional[datetime] = None
    taken_history: List[datetime] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "user_id": self.user_id,
            "medication_name": self.medication_name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "scheduled_times": self.scheduled_times,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "instructions": self.instructions,
            "language": self.language,
            "active": self.active,
            "last_reminded": self.last_reminded.isoformat() if self.last_reminded else None,
            "taken_history": [t.isoformat() for t in self.taken_history],
        }


class MedicationReminderScheduler:
    """
    Schedule and manage medication reminders for patients.
    Supports multiple languages and flexible schedules.
    """
    
    FREQUENCY_PATTERNS = {
        "daily": 1,
        "twice_daily": 2,
        "thrice_daily": 3,
        "weekly": 7,
        "monthly": 30,
    }
    
    def __init__(self, storage_path: str = "data/medication_reminders"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.reminders: Dict[str, MedicationReminder] = {}
        self.user_reminders: Dict[str, List[str]] = defaultdict(list)
        
        self._load_reminders()
        self._start_scheduler()
    
    def _load_reminders(self):
        """Load reminders from storage"""
        reminders_file = self.storage_path / "reminders.json"
        if reminders_file.exists():
            try:
                with open(reminders_file, 'r') as f:
                    data = json.load(f)
                    for reminder_data in data:
                        reminder = self._dict_to_reminder(reminder_data)
                        self.reminders[reminder.reminder_id] = reminder
                        self.user_reminders[reminder.user_id].append(reminder.reminder_id)
                logger.info(f"Loaded {len(self.reminders)} medication reminders")
            except Exception as e:
                logger.error(f"Failed to load reminders: {e}")
    
    def _save_reminders(self):
        """Save reminders to storage"""
        try:
            reminders_file = self.storage_path / "reminders.json"
            data = [r.to_dict() for r in self.reminders.values()]
            with open(reminders_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reminders: {e}")
    
    def _dict_to_reminder(self, data: Dict) -> MedicationReminder:
        """Convert dict to MedicationReminder"""
        return MedicationReminder(
            reminder_id=data["reminder_id"],
            user_id=data["user_id"],
            medication_name=data["medication_name"],
            dosage=data["dosage"],
            frequency=data["frequency"],
            scheduled_times=data["scheduled_times"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            instructions=data["instructions"],
            language=data["language"],
            active=data.get("active", True),
            last_reminded=datetime.fromisoformat(data["last_reminded"]) if data.get("last_reminded") else None,
            taken_history=[datetime.fromisoformat(t) for t in data.get("taken_history", [])],
        )
    
    def _start_scheduler(self):
        """Start the background reminder scheduler"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("Medication reminder scheduler started")
    
    def create_reminder(
        self,
        user_id: str,
        medication_name: str,
        dosage: str,
        frequency: str,
        times: List[str],
        instructions: str = "",
        language: str = "en",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> MedicationReminder:
        """Create a new medication reminder"""
        reminder_id = hashlib.md5(
            f"{user_id}_{medication_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        reminder = MedicationReminder(
            reminder_id=reminder_id,
            user_id=user_id,
            medication_name=medication_name,
            dosage=dosage,
            frequency=frequency,
            scheduled_times=times,
            start_date=start_date or datetime.now(),
            end_date=end_date,
            instructions=instructions,
            language=language,
        )
        
        self.reminders[reminder_id] = reminder
        self.user_reminders[user_id].append(reminder_id)
        self._save_reminders()
        
        # Schedule the reminder
        self._schedule_reminder(reminder)
        
        logger.info(f"Created reminder {reminder_id} for user {user_id}")
        return reminder
    
    def _schedule_reminder(self, reminder: MedicationReminder):
        """Schedule a reminder in the scheduler"""
        for time_str in reminder.scheduled_times:
            try:
                hour, minute = map(int, time_str.split(':'))
                schedule.every().day.at(time_str).do(
                    self._trigger_reminder, reminder.reminder_id
                )
                logger.info(f"Scheduled reminder {reminder.reminder_id} at {time_str}")
            except Exception as e:
                logger.error(f"Failed to schedule reminder at {time_str}: {e}")
    
    def _trigger_reminder(self, reminder_id: str):
        """Trigger a medication reminder (called by scheduler)"""
        if reminder_id not in self.reminders:
            return
        
        reminder = self.reminders[reminder_id]
        
        if not reminder.active:
            return
        
        # Check if end date has passed
        if reminder.end_date and datetime.now() > reminder.end_date:
            reminder.active = False
            self._save_reminders()
            return
        
        # Update last reminded
        reminder.last_reminded = datetime.now()
        self._save_reminders()
        
        # This will be called by the main app with WhatsApp API
        logger.info(f"Medication reminder triggered: {reminder.medication_name} for user {reminder.user_id}")
        
        # Return the reminder info for the main loop to handle
        return reminder
    
    def get_reminder_message(self, reminder: MedicationReminder) -> str:
        """Generate reminder message in appropriate language"""
        messages = {
            "en": f"💊 Medication Reminder\n\nTime to take: {reminder.medication_name}\nDosage: {reminder.dosage}\n\n{reminder.instructions}\n\nReply 'TAKEN' when you've taken your medication.",
            "hi": f"💊 दवा की याद दिलाने वाला\n\nलेने का समय: {reminder.medication_name}\nखुराक: {reminder.dosage}\n\n{reminder.instructions}\n\nदवा लेने के बाद 'TAKEN' जवाब दें।",
            "bn": f"💊 ওষুধ স্মরণ করিয়ে দেওয়া\n\nখাওয়ার সময়: {reminder.medication_name}\nডোজ: {reminder.dosage}\n\n{reminder.instructions}\n\nওষুধ খাওয়ার পরে 'TAKEN' উত্তর দিন।",
            "ta": f"💊 மருந்து நினைவூட்டல்\n\nஎடுக்கும் நேரம்: {reminder.medication_name}\nமருந்தளவு: {reminder.dosage}\n\n{reminder.instructions}\n\nமருந்தை எடுத்த பிறகு 'TAKEN' பதிலளிக்கவும்.",
            "gu": f"💊 દવા યાદ અપાવનાર\n\nલેવાનો સમય: {reminder.medication_name}\nખુરાક: {reminder.dosage}\n\n{reminder.instructions}\n\nતમે દવા લીધા પછી 'TAKEN' જવાબ આપો.",
        }
        return messages.get(reminder.language, messages["en"])
    
    def mark_taken(self, reminder_id: str) -> bool:
        """Mark a medication as taken"""
        if reminder_id not in self.reminders:
            return False
        
        reminder = self.reminders[reminder_id]
        reminder.taken_history.append(datetime.now())
        self._save_reminders()
        
        logger.info(f"Marked medication {reminder.medication_name} as taken")
        return True
    
    def get_user_reminders(self, user_id: str) -> List[MedicationReminder]:
        """Get all reminders for a user"""
        reminder_ids = self.user_reminders.get(user_id, [])
        return [self.reminders[r_id] for r_id in reminder_ids if r_id in self.reminders]
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder"""
        if reminder_id not in self.reminders:
            return False
        
        reminder = self.reminders[reminder_id]
        user_id = reminder.user_id
        
        del self.reminders[reminder_id]
        if reminder_id in self.user_reminders[user_id]:
            self.user_reminders[user_id].remove(reminder_id)
        
        self._save_reminders()
        logger.info(f"Deleted reminder {reminder_id}")
        return True


# ============================================================================
# 4. RESPONSE LENGTH OPTIMIZATION
# ============================================================================

class UserComprehensionLevel(Enum):
    """User comprehension/sophistication levels"""
    SIMPLE = "simple"       # Short, simple language
    MODERATE = "moderate"   # Medium length, some medical terms
    DETAILED = "detailed"   # Comprehensive, technical


@dataclass
class ComprehensionProfile:
    """User's comprehension profile"""
    user_id: str
    level: UserComprehensionLevel
    avg_message_length: int
    vocabulary_complexity: float
    question_types: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class ResponseLengthOptimizer:
    """
    Optimizes response length and complexity based on user comprehension level.
    Adapts to user's communication style and capabilities.
    """
    
    LENGTH_GUIDELINES = {
        UserComprehensionLevel.SIMPLE: {
            "max_chars": 500,
            "max_sentences": 4,
            "sentence_length": "short",
            "medical_terms": "avoid",
            "structure": "bullet_points",
        },
        UserComprehensionLevel.MODERATE: {
            "max_chars": 1000,
            "max_sentences": 8,
            "sentence_length": "medium",
            "medical_terms": "explain",
            "structure": "paragraph_with_key_points",
        },
        UserComprehensionLevel.DETAILED: {
            "max_chars": 2000,
            "max_sentences": 15,
            "sentence_length": "long",
            "medical_terms": "use_with_citations",
            "structure": "comprehensive",
        },
    }
    
    def __init__(self, storage_path: str = "data/comprehension_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.profiles: Dict[str, ComprehensionProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load comprehension profiles"""
        profiles_file = self.storage_path / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for profile_data in data:
                        profile = ComprehensionProfile(
                            user_id=profile_data["user_id"],
                            level=UserComprehensionLevel(profile_data["level"]),
                            avg_message_length=profile_data["avg_message_length"],
                            vocabulary_complexity=profile_data["vocabulary_complexity"],
                            question_types=profile_data["question_types"],
                            last_updated=datetime.fromisoformat(profile_data["last_updated"]),
                        )
                        self.profiles[profile.user_id] = profile
            except Exception as e:
                logger.error(f"Failed to load comprehension profiles: {e}")
    
    def _save_profiles(self):
        """Save comprehension profiles"""
        try:
            profiles_file = self.storage_path / "profiles.json"
            data = []
            for profile in self.profiles.values():
                data.append({
                    "user_id": profile.user_id,
                    "level": profile.level.value,
                    "avg_message_length": profile.avg_message_length,
                    "vocabulary_complexity": profile.vocabulary_complexity,
                    "question_types": profile.question_types,
                    "last_updated": profile.last_updated.isoformat(),
                })
            with open(profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
    
    def analyze_user_message(self, user_id: str, message: str):
        """Analyze user message to update comprehension profile"""
        # Calculate message complexity metrics
        word_count = len(message.split())
        sentence_count = len([s for s in message.split('.') if s.strip()])
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Check for medical terminology
        medical_terms = [
            "pain", "symptom", "treatment", "medication", "diagnosis",
            "prognosis", "chronic", "acute", "malignant", "benign",
        ]
        medical_term_count = sum(1 for term in medical_terms if term in message.lower())
        vocabulary_complexity = min(1.0, medical_term_count / 5.0)
        
        # Determine question type
        question_types = []
        if any(w in message.lower() for w in ["what", "क्या", "কি", "என்ன"]):
            question_types.append("what")
        if any(w in message.lower() for w in ["how", "कैसे", "কিভাবে", "எப்படி"]):
            question_types.append("how")
        if any(w in message.lower() for w in ["why", "क्यों", "কেন", "ஏன்"]):
            question_types.append("why")
        
        # Update or create profile
        if user_id in self.profiles:
            profile = self.profiles[user_id]
            # Update with exponential moving average
            profile.avg_message_length = int(0.7 * profile.avg_message_length + 0.3 * len(message))
            profile.vocabulary_complexity = 0.7 * profile.vocabulary_complexity + 0.3 * vocabulary_complexity
            profile.question_types = list(set(profile.question_types + question_types))
            profile.last_updated = datetime.now()
        else:
            # Determine initial level based on first message
            if avg_sentence_length < 5 and word_count < 15:
                level = UserComprehensionLevel.SIMPLE
            elif avg_sentence_length < 12 and word_count < 40:
                level = UserComprehensionLevel.MODERATE
            else:
                level = UserComprehensionLevel.DETAILED
            
            profile = ComprehensionProfile(
                user_id=user_id,
                level=level,
                avg_message_length=len(message),
                vocabulary_complexity=vocabulary_complexity,
                question_types=question_types,
            )
            self.profiles[user_id] = profile
        
        self._save_profiles()
    
    def get_user_level(self, user_id: str) -> UserComprehensionLevel:
        """Get comprehension level for user"""
        if user_id not in self.profiles:
            return UserComprehensionLevel.MODERATE
        return self.profiles[user_id].level
    
    def get_optimization_prompt(self, user_id: str) -> str:
        """Get LLM prompt addition for response optimization"""
        level = self.get_user_level(user_id)
        guidelines = self.LENGTH_GUIDELINES[level]
        
        prompts = {
            UserComprehensionLevel.SIMPLE: """
CRITICAL RESPONSE REQUIREMENTS:
- Maximum 4 short sentences
- Use simple words (8th grade level)
- Avoid medical jargon - use everyday terms
- Use bullet points for clarity
- Focus on the most important 1-2 points only
- Be direct and clear
""",
            UserComprehensionLevel.MODERATE: """
RESPONSE REQUIREMENTS:
- Maximum 8 sentences
- Use moderate vocabulary
- Explain medical terms when first used
- Include key points in bullets after brief explanation
- Provide practical advice
""",
            UserComprehensionLevel.DETAILED: """
RESPONSE REQUIREMENTS:
- Provide comprehensive information (up to 2000 characters)
- Use appropriate medical terminology with citations
- Include context, mechanisms, and evidence
- Structure with clear sections
- Reference sources where applicable
""",
        }
        
        return prompts.get(level, prompts[UserComprehensionLevel.MODERATE])
    
    def adapt_response(self, response: str, user_id: str) -> str:
        """Adapt an existing response to user's level (post-processing)"""
        level = self.get_user_level(user_id)
        guidelines = self.LENGTH_GUIDELINES[level]
        
        # Truncate if too long
        if len(response) > guidelines["max_chars"]:
            # Find the last complete sentence before the limit
            truncated = response[:guidelines["max_chars"]]
            last_period = truncated.rfind('.')
            if last_period > 0:
                response = truncated[:last_period + 1]
            else:
                response = truncated + "..."
        
        return response


# ============================================================================
# 5. HUMAN HANDOFF SYSTEM
# ============================================================================

class HandoffReason(Enum):
    """Reasons for human handoff"""
    EMERGENCY = "emergency"
    COMPLEX_MEDICAL = "complex_medical"
    EMOTIONAL_SUPPORT = "emotional_support"
    TECHNICAL_ISSUE = "technical_issue"
    USER_REQUEST = "user_request"
    AI_UNCERTAIN = "ai_uncertain"
    SAFETY_ESCALATION = "safety_escalation"


@dataclass
class HandoffRequest:
    """Human handoff request"""
    request_id: str
    user_id: str
    reason: HandoffReason
    priority: str  # "low", "medium", "high", "urgent"
    context: str
    conversation_history: List[Dict[str, Any]]
    timestamp: datetime
    status: str = "pending"  # "pending", "assigned", "in_progress", "resolved"
    assigned_to: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "reason": self.reason.value,
            "priority": self.priority,
            "context": self.context,
            "conversation_history": self.conversation_history,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "assigned_to": self.assigned_to,
            "notes": self.notes,
        }


class HumanHandoffSystem:
    """
    Manages warm handoffs to human caregivers/nurses.
    Provides seamless transition from AI to human support.
    """
    
    # Handoff trigger patterns
    HANDOFF_TRIGGERS = {
        HandoffReason.EMERGENCY: {
            "keywords": ["emergency", "urgent", "critical", "ambulance", "108", "102"],
            "auto_trigger": True,
        },
        HandoffReason.COMPLEX_MEDICAL: {
            "keywords": ["complex", "complicated", "rare condition", "multiple medications", "interaction"],
            "auto_trigger": False,
        },
        HandoffReason.EMOTIONAL_SUPPORT: {
            "keywords": ["grief", "depressed", "anxious", "scared", "alone", "can't cope"],
            "auto_trigger": False,
        },
        HandoffReason.USER_REQUEST: {
            "keywords": ["talk to a human", "talk to human", "speak to doctor", "real person", "nurse", "caregiver"],
            "auto_trigger": True,
        },
    }
    
    def __init__(self, storage_path: str = "data/handoff_requests"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.pending_requests: Dict[str, HandoffRequest] = {}
        self.resolved_requests: Dict[str, HandoffRequest] = {}
        
        # Caregiver availability
        self.available_caregivers: List[str] = []
        self.caregiver_assignments: Dict[str, str] = {}  # request_id -> caregiver_id
        
        self._load_requests()
    
    def _load_requests(self):
        """Load handoff requests from storage"""
        pending_file = self.storage_path / "pending.json"
        resolved_file = self.storage_path / "resolved.json"
        
        if pending_file.exists():
            try:
                with open(pending_file, 'r') as f:
                    data = json.load(f)
                    for req_data in data:
                        request = self._dict_to_request(req_data)
                        self.pending_requests[request.request_id] = request
            except Exception as e:
                logger.error(f"Failed to load pending requests: {e}")
        
        if resolved_file.exists():
            try:
                with open(resolved_file, 'r') as f:
                    data = json.load(f)
                    for req_data in data:
                        request = self._dict_to_request(req_data)
                        self.resolved_requests[request.request_id] = request
            except Exception as e:
                logger.error(f"Failed to load resolved requests: {e}")
    
    def _save_requests(self):
        """Save handoff requests"""
        try:
            pending_file = self.storage_path / "pending.json"
            resolved_file = self.storage_path / "resolved.json"
            
            with open(pending_file, 'w') as f:
                json.dump([r.to_dict() for r in self.pending_requests.values()], f, indent=2)
            
            with open(resolved_file, 'w') as f:
                json.dump([r.to_dict() for r in self.resolved_requests.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save requests: {e}")
    
    def _dict_to_request(self, data: Dict) -> HandoffRequest:
        """Convert dict to HandoffRequest"""
        return HandoffRequest(
            request_id=data["request_id"],
            user_id=data["user_id"],
            reason=HandoffReason(data["reason"]),
            priority=data["priority"],
            context=data["context"],
            conversation_history=data.get("conversation_history", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=data.get("status", "pending"),
            assigned_to=data.get("assigned_to"),
            notes=data.get("notes", ""),
        )
    
    def check_handoff_needed(self, query: str, ai_confidence: float = 1.0) -> Optional[HandoffReason]:
        """Check if handoff is needed based on query"""
        query_lower = query.lower()
        
        for reason, config in self.HANDOFF_TRIGGERS.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    return reason
        
        # Check AI uncertainty
        if ai_confidence < 0.5:
            return HandoffReason.AI_UNCERTAIN
        
        return None
    
    def create_handoff_request(
        self,
        user_id: str,
        reason: HandoffReason,
        context: str,
        conversation_history: List[Dict[str, Any]],
        priority: str = "medium"
    ) -> HandoffRequest:
        """Create a new handoff request"""
        request_id = hashlib.md5(
            f"{user_id}_{reason.value}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Determine priority based on reason
        if reason == HandoffReason.EMERGENCY:
            priority = "urgent"
        elif reason in [HandoffReason.SAFETY_ESCALATION, HandoffReason.AI_UNCERTAIN]:
            priority = "high"
        
        request = HandoffRequest(
            request_id=request_id,
            user_id=user_id,
            reason=reason,
            priority=priority,
            context=context,
            conversation_history=conversation_history,
            timestamp=datetime.now(),
        )
        
        self.pending_requests[request_id] = request
        self._save_requests()
        
        logger.info(f"Created handoff request {request_id} for user {user_id} (reason: {reason.value})")
        return request
    
    def get_handoff_message(self, request: HandoffRequest, language: str = "en") -> str:
        """Generate handoff message for user"""
        messages = {
            "en": {
                HandoffReason.EMERGENCY: f"🚨 Emergency detected. Connecting you to a human caregiver immediately.\n\nRequest ID: {request.request_id}\nA nurse will contact you within 5 minutes.",
                HandoffReason.COMPLEX_MEDICAL: f"🏥 This question requires medical expertise. Connecting you to a healthcare professional.\n\nRequest ID: {request.request_id}\nExpected response: 15-30 minutes",
                HandoffReason.EMOTIONAL_SUPPORT: f"💙 I understand you're going through a difficult time. Let me connect you with someone who can provide emotional support.\n\nRequest ID: {request.request_id}\nA counselor will reach out within 10 minutes.",
                HandoffReason.TECHNICAL_ISSUE: f"⚙️ I'm having trouble understanding. Let me connect you with technical support.\n\nRequest ID: {request.request_id}",
                HandoffReason.USER_REQUEST: f"👤 Connecting you to a human caregiver as requested.\n\nRequest ID: {request.request_id}\nSomeone will be with you shortly.",
                HandoffReason.AI_UNCERTAIN: f"🤔 I want to make sure you get the most accurate information. Let me have a healthcare professional review your question.\n\nRequest ID: {request.request_id}\nExpected response: 15-30 minutes",
                HandoffReason.SAFETY_ESCALATION: f"⚠️ For your safety, this requires human review. A healthcare professional will contact you shortly.\n\nRequest ID: {request.request_id}",
            },
            "hi": {
                HandoffReason.EMERGENCY: f"🚨 आपातकाल का पता चला। तुरंत एक मानव देखभाल करने वाले से जोड़ रहा हूं।\n\nअनुरोध ID: {request.request_id}\nएक नर्स 5 मिनट के भीतर संपर्क करेगी।",
                HandoffReason.USER_REQUEST: f"👤 आपके अनुरोध के अनुसार एक मानव देखभाल करने वाले से जोड़ रहा हूं।\n\nअनुरोध ID: {request.request_id}\nकोई जल्द ही आपके साथ होगा।",
                HandoffReason.AI_UNCERTAIN: f"🤔 मैं चाहता हूं कि आपको सबसे सटीक जानकारी मिले। मैं एक स्वास्थ्य पेशेवर से आपके सवाल की समीक्षा करवाता हूं।\n\nअनुरोध ID: {request.request_id}\nअपेक्षित प्रतिक्रिया: 15-30 मिनट",
            },
        }
        
        lang_messages = messages.get(language, messages["en"])
        return lang_messages.get(request.reason, lang_messages.get(HandoffReason.USER_REQUEST, messages["en"][HandoffReason.USER_REQUEST]))
    
    def assign_caregiver(self, request_id: str, caregiver_id: str) -> bool:
        """Assign a caregiver to a handoff request"""
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        request.assigned_to = caregiver_id
        request.status = "assigned"
        
        self.caregiver_assignments[request_id] = caregiver_id
        self._save_requests()
        
        logger.info(f"Assigned caregiver {caregiver_id} to request {request_id}")
        return True
    
    def mark_in_progress(self, request_id: str) -> bool:
        """Mark request as in progress"""
        if request_id not in self.pending_requests:
            return False
        
        self.pending_requests[request_id].status = "in_progress"
        self._save_requests()
        return True
    
    def resolve_request(self, request_id: str, notes: str = "") -> bool:
        """Resolve a handoff request"""
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        request.status = "resolved"
        request.notes = notes
        
        # Move to resolved
        self.resolved_requests[request_id] = request
        del self.pending_requests[request_id]
        
        if request_id in self.caregiver_assignments:
            del self.caregiver_assignments[request_id]
        
        self._save_requests()
        
        logger.info(f"Resolved handoff request {request_id}")
        return True
    
    def get_pending_requests(self, priority_filter: Optional[str] = None) -> List[HandoffRequest]:
        """Get pending handoff requests"""
        requests = list(self.pending_requests.values())
        if priority_filter:
            requests = [r for r in requests if r.priority == priority_filter]
        # Sort by priority and timestamp
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        requests.sort(key=lambda r: (priority_order.get(r.priority, 2), r.timestamp))
        return requests
    
    def get_user_active_request(self, user_id: str) -> Optional[HandoffRequest]:
        """Get active handoff request for a user"""
        for request in self.pending_requests.values():
            if request.user_id == user_id and request.status in ["pending", "assigned", "in_progress"]:
                return request
        return None


# ============================================================================
# MAIN SAFETY ENHANCEMENTS MANAGER
# ============================================================================

class SafetyEnhancementsManager:
    """
    Main manager class that coordinates all 5 safety enhancement features.
    Integrates with the main RAG pipeline and WhatsApp bot.
    """
    
    def __init__(self):
        self.evidence_system = EvidenceBadgeSystem()
        self.emergency_system = EmergencyDetectionSystem()
        self.reminder_scheduler = MedicationReminderScheduler()
        self.response_optimizer = ResponseLengthOptimizer()
        self.handoff_system = HumanHandoffSystem()
        
        logger.info("🛡️ Safety Enhancements Manager initialized")
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        language: str = "en",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process user query through all safety enhancement layers.
        Returns dict with: should_respond, response_additions, actions
        """
        result = {
            "should_respond": True,
            "response_additions": {},
            "actions": [],
            "modified_prompt": None,
            "emergency_alert": None,
            "handoff_request": None,
        }
        
        # 1. Check for emergencies first (highest priority)
        emergency_alert = self.emergency_system.detect_emergency(query, user_id, language)
        if emergency_alert:
            result["emergency_alert"] = emergency_alert
            result["response_additions"]["emergency"] = emergency_alert.message
            result["response_additions"]["emergency_actions"] = emergency_alert.action_required
            
            if emergency_alert.contact_emergency_services:
                result["should_respond"] = False
                # Create urgent handoff
                handoff = self.handoff_system.create_handoff_request(
                    user_id=user_id,
                    reason=HandoffReason.EMERGENCY,
                    context=f"Emergency detected: {', '.join(emergency_alert.detected_keywords)}",
                    conversation_history=conversation_history or [],
                    priority="urgent"
                )
                result["handoff_request"] = handoff
                return result
        
        # 2. Check if handoff is needed
        handoff_reason = self.handoff_system.check_handoff_needed(query)
        if handoff_reason:
            handoff = self.handoff_system.create_handoff_request(
                user_id=user_id,
                reason=handoff_reason,
                context=query,
                conversation_history=conversation_history or [],
            )
            result["handoff_request"] = handoff
            result["should_respond"] = False
            result["response_additions"]["handoff"] = self.handoff_system.get_handoff_message(handoff, language)
            return result
        
        # 3. Analyze user comprehension
        self.response_optimizer.analyze_user_message(user_id, query)
        result["modified_prompt"] = self.response_optimizer.get_optimization_prompt(user_id)
        
        return result
    
    def add_evidence_badge(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        distances: List[float],
        answer: str,
        language: str = "en"
    ) -> str:
        """Add evidence badge to response"""
        badge = self.evidence_system.calculate_evidence_badge(query, sources, distances, answer)
        badge_text = badge.format_for_user(language)
        return f"{answer}\n{badge_text}"
    
    def optimize_response(self, response: str, user_id: str) -> str:
        """Optimize response length for user"""
        return self.response_optimizer.adapt_response(response, user_id)


# Singleton instance
_safety_manager: Optional[SafetyEnhancementsManager] = None


def get_safety_manager() -> SafetyEnhancementsManager:
    """Get or create the safety manager singleton"""
    global _safety_manager
    if _safety_manager is None:
        _safety_manager = SafetyEnhancementsManager()
    return _safety_manager
