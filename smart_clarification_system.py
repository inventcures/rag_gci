"""
Smart Clarification System for Healthcare Queries
Intelligently determines when clarifying questions are needed and limits them to 2-3 essential questions
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ClarityLevel(Enum):
    CLEAR = "clear"          # Query is clear, no clarification needed
    NEEDS_CONTEXT = "needs_context"    # Needs 1-2 context questions
    AMBIGUOUS = "ambiguous"   # Needs 2-3 clarifying questions
    UNCLEAR = "unclear"       # Too vague, needs significant clarification

@dataclass
class ClarificationState:
    user_id: str
    original_query: str
    clarity_level: ClarityLevel
    questions_asked: List[str]
    answers_received: List[str]
    max_questions: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SmartClarificationSystem:
    """Intelligent system to determine when and what to ask for clarification"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.user_states: Dict[str, ClarificationState] = {}
        
        # Medical context patterns that often need clarification
        self.ambiguous_patterns = [
            r'\b(pain|दर्द|ব্যথা|வலி|દુ:ખ)\b',  # Pain without location/type
            r'\b(medicine|medication|दवा|ঔষধ|மருந்து|દવા)\b',  # Medicine without specifics
            r'\b(treatment|इलाज|চিকিত্সা|சிकित्सा|સારવાર)\b',  # Treatment without condition
            r'\b(problem|समस्या|সমস্যা|समस्या|સમસ્યા)\b',  # Problem without details
            r'\b(how|कैसे|কীভাবে|எப்படி|કેવી)\s*(to|से)?\b',  # How questions without context
        ]
        
        # Clear queries that typically don't need clarification
        self.clear_patterns = [
            r'what\s+is\s+\w+',  # "What is diabetes"
            r'how\s+to\s+(take|use)\s+\w+',  # "How to take medicine"
            r'side\s+effects?\s+of\s+\w+',  # "Side effects of drug"
            r'symptoms?\s+of\s+\w+',  # "Symptoms of disease"
            r'dosage\s+of\s+\w+',  # "Dosage of medicine"
        ]
        
        # Healthcare-specific question templates
        self.question_templates = {
            "pain": [
                "Where exactly is the pain located?",
                "How would you describe the pain (sharp, dull, burning)?",
                "When did the pain start?"
            ],
            "medicine": [
                "What specific medicine are you asking about?",
                "What condition is this medicine for?",
                "Are you currently taking any other medications?"
            ],
            "treatment": [
                "What condition do you need treatment for?",
                "Have you been diagnosed by a doctor?",
                "What symptoms are you experiencing?"
            ],
            "general": [
                "Can you provide more details about your concern?",
                "What specific symptoms are you experiencing?",
                "How long have you been experiencing this?"
            ]
        }
    
    async def analyze_query_clarity(self, query: str, user_id: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Analyze if a query needs clarification
        Returns: {
            "needs_clarification": bool,
            "clarity_level": ClarityLevel,
            "confidence": float,
            "reasoning": str,
            "suggested_questions": List[str]
        }
        """
        try:
            # Clean and normalize query
            clean_query = query.strip().lower()
            
            # Check if user is currently in clarification flow
            if user_id in self.user_states:
                state = self.user_states[user_id]
                # If it's been more than 10 minutes, reset the state
                if datetime.now() - state.created_at > timedelta(minutes=10):
                    del self.user_states[user_id]
                else:
                    return await self._handle_clarification_response(query, user_id)
            
            # Rule-based initial assessment
            rule_based_result = self._rule_based_clarity_check(clean_query)
            
            # If rule-based is clear, trust it (avoid unnecessary LLM calls)
            if rule_based_result["clarity_level"] == ClarityLevel.CLEAR:
                return rule_based_result
            
            # For potentially unclear queries, use LLM for deeper analysis
            llm_result = await self._llm_clarity_analysis(query, conversation_history)
            
            # Combine rule-based and LLM results
            final_result = self._combine_analysis_results(rule_based_result, llm_result)
            
            # Only initiate clarification if really needed
            if final_result["needs_clarification"] and final_result["confidence"] > 0.7:
                await self._initiate_clarification_flow(query, user_id, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error analyzing query clarity: {e}")
            # Default to no clarification on error
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 0.5,
                "reasoning": "Error in analysis, proceeding with query",
                "suggested_questions": []
            }
    
    def _rule_based_clarity_check(self, query: str) -> Dict[str, Any]:
        """Fast rule-based clarity assessment"""
        
        # Check for clear patterns
        for pattern in self.clear_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "needs_clarification": False,
                    "clarity_level": ClarityLevel.CLEAR,
                    "confidence": 0.8,
                    "reasoning": "Query matches clear pattern",
                    "suggested_questions": []
                }
        
        # Count ambiguous patterns
        ambiguous_count = 0
        ambiguous_types = []
        
        for pattern in self.ambiguous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                ambiguous_count += 1
                if "pain" in pattern:
                    ambiguous_types.append("pain")
                elif "medicine" in pattern:
                    ambiguous_types.append("medicine")
                elif "treatment" in pattern:
                    ambiguous_types.append("treatment")
                else:
                    ambiguous_types.append("general")
        
        # Determine clarity level based on patterns
        if ambiguous_count == 0:
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 0.7,
                "reasoning": "No ambiguous patterns detected",
                "suggested_questions": []
            }
        elif ambiguous_count == 1:
            return {
                "needs_clarification": True,
                "clarity_level": ClarityLevel.NEEDS_CONTEXT,
                "confidence": 0.6,
                "reasoning": f"Single ambiguous pattern detected: {ambiguous_types[0]}",
                "suggested_questions": self._get_questions_for_type(ambiguous_types[0])[:2]
            }
        else:
            return {
                "needs_clarification": True,
                "clarity_level": ClarityLevel.AMBIGUOUS,
                "confidence": 0.8,
                "reasoning": f"Multiple ambiguous patterns: {', '.join(ambiguous_types)}",
                "suggested_questions": self._get_questions_for_type("general")[:3]
            }
    
    async def _llm_clarity_analysis(self, query: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Use LLM to analyze query clarity for complex cases"""
        try:
            # Create context-aware prompt
            context = ""
            if conversation_history:
                context = "Previous conversation:\n" + "\n".join(conversation_history[-3:]) + "\n\n"
            
            prompt = f"""You are a medical assistant AI analyzing whether a healthcare query needs clarification.

{context}Current query: "{query}"

Analyze this query and respond with ONLY a JSON object:
{{
    "needs_clarification": boolean,
    "clarity_level": "clear|needs_context|ambiguous|unclear",
    "confidence": float (0.0-1.0),
    "reasoning": "brief explanation",
    "missing_info": ["list", "of", "missing", "key", "information"],
    "suggested_questions": ["max 2-3 focused questions"]
}}

Guidelines:
- Only suggest clarification if ESSENTIAL information is missing
- Limit to 2-3 most important questions
- Focus on patient safety and accurate diagnosis
- Consider that the user may have limited medical knowledge

Examples of CLEAR queries (no clarification needed):
- "What are symptoms of diabetes?"
- "Side effects of paracetamol"
- "How to take blood pressure medication"

Examples that NEED clarification:
- "I have pain" (where? what type? how long?)
- "My medicine isn't working" (which medicine? what condition? how long taking?)
- "I feel sick" (what symptoms? how long? severity?)"""

            # Call Groq API
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Extract JSON from response
                try:
                    # Find JSON block
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        parsed = json.loads(json_str)
                        
                        # Convert clarity_level string to enum
                        clarity_map = {
                            "clear": ClarityLevel.CLEAR,
                            "needs_context": ClarityLevel.NEEDS_CONTEXT,
                            "ambiguous": ClarityLevel.AMBIGUOUS,
                            "unclear": ClarityLevel.UNCLEAR
                        }
                        parsed["clarity_level"] = clarity_map.get(parsed.get("clarity_level", "clear"), ClarityLevel.CLEAR)
                        
                        return parsed
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM JSON response: {e}")
            
            # Fallback if LLM call fails
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 0.5,
                "reasoning": "LLM analysis failed, proceeding with query",
                "suggested_questions": []
            }
            
        except Exception as e:
            logger.error(f"LLM clarity analysis error: {e}")
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 0.5,
                "reasoning": "Analysis error, proceeding with query",
                "suggested_questions": []
            }
    
    def _combine_analysis_results(self, rule_result: Dict, llm_result: Dict) -> Dict:
        """Intelligently combine rule-based and LLM analysis"""
        
        # If rule-based is confident it's clear, trust it
        if rule_result["clarity_level"] == ClarityLevel.CLEAR and rule_result["confidence"] > 0.7:
            return rule_result
        
        # If LLM is confident, use its result
        if llm_result.get("confidence", 0) > 0.8:
            return llm_result
        
        # Otherwise, be conservative - prefer fewer clarification questions
        if rule_result["confidence"] >= llm_result.get("confidence", 0):
            return rule_result
        else:
            return llm_result
    
    def _get_questions_for_type(self, question_type: str) -> List[str]:
        """Get appropriate questions for the ambiguity type"""
        return self.question_templates.get(question_type, self.question_templates["general"])
    
    async def _initiate_clarification_flow(self, query: str, user_id: str, analysis_result: Dict):
        """Start the clarification conversation flow"""
        
        clarity_level = analysis_result["clarity_level"]
        suggested_questions = analysis_result["suggested_questions"]
        
        # Limit questions based on clarity level
        if clarity_level == ClarityLevel.NEEDS_CONTEXT:
            max_questions = 2
        elif clarity_level == ClarityLevel.AMBIGUOUS:
            max_questions = 3
        else:
            max_questions = 2
        
        # Limit to maximum suggested questions
        questions_to_ask = suggested_questions[:max_questions]
        
        # Create clarification state
        self.user_states[user_id] = ClarificationState(
            user_id=user_id,
            original_query=query,
            clarity_level=clarity_level,
            questions_asked=questions_to_ask,
            answers_received=[],
            max_questions=max_questions
        )
        
        logger.info(f"Initiated clarification flow for user {user_id}: {len(questions_to_ask)} questions")
    
    async def _handle_clarification_response(self, response: str, user_id: str) -> Dict[str, Any]:
        """Handle user's response to clarification question"""
        
        if user_id not in self.user_states:
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 1.0,
                "reasoning": "No active clarification session",
                "suggested_questions": []
            }
        
        state = self.user_states[user_id]
        
        # Add user's response
        state.answers_received.append(response)
        
        # Check if we have enough information or reached max questions
        if len(state.answers_received) >= len(state.questions_asked) or len(state.answers_received) >= state.max_questions:
            # Clarification complete - construct enhanced query
            enhanced_query = self._construct_enhanced_query(state)
            
            # Clean up state
            del self.user_states[user_id]
            
            return {
                "needs_clarification": False,
                "clarity_level": ClarityLevel.CLEAR,
                "confidence": 1.0,
                "reasoning": "Clarification completed",
                "enhanced_query": enhanced_query,
                "original_query": state.original_query,
                "clarification_complete": True
            }
        
        # More clarification needed
        next_question_idx = len(state.answers_received)
        if next_question_idx < len(state.questions_asked):
            return {
                "needs_clarification": True,
                "clarity_level": state.clarity_level,
                "confidence": 1.0,
                "reasoning": "Continuing clarification flow",
                "next_question": state.questions_asked[next_question_idx],
                "questions_remaining": len(state.questions_asked) - next_question_idx - 1
            }
        
        # Shouldn't reach here, but handle gracefully
        del self.user_states[user_id]
        return {
            "needs_clarification": False,
            "clarity_level": ClarityLevel.CLEAR,
            "confidence": 1.0,
            "reasoning": "Clarification flow completed unexpectedly",
            "suggested_questions": []
        }
    
    def _construct_enhanced_query(self, state: ClarificationState) -> str:
        """Construct an enhanced query from original query and clarification responses"""
        
        enhanced_parts = [f"Original question: {state.original_query}"]
        
        for i, (question, answer) in enumerate(zip(state.questions_asked, state.answers_received)):
            if answer.strip():
                enhanced_parts.append(f"Additional context {i+1}: {answer}")
        
        return " | ".join(enhanced_parts)
    
    def get_clarification_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current clarification status for a user"""
        if user_id not in self.user_states:
            return None
        
        state = self.user_states[user_id]
        return {
            "in_clarification": True,
            "original_query": state.original_query,
            "questions_asked": len(state.questions_asked),
            "questions_answered": len(state.answers_received),
            "remaining_questions": len(state.questions_asked) - len(state.answers_received),
            "current_question": state.questions_asked[len(state.answers_received)] if len(state.answers_received) < len(state.questions_asked) else None
        }
    
    def clear_user_state(self, user_id: str):
        """Clear clarification state for a user (useful for reset/cancel)"""
        if user_id in self.user_states:
            del self.user_states[user_id]