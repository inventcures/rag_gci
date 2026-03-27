"""Response quality evaluator for the meta-agent feedback loop."""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EVALUATIONS_DIR = "data/meta_agent/evaluations"


@dataclass
class EvaluationResult:
    id: str
    query: str
    response: str
    relevance_score: float
    safety_score: float
    completeness_score: float
    citation_quality: float
    overall_score: float
    physician_score: Optional[float] = None
    user_feedback: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.utcnow().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "relevance_score": self.relevance_score,
            "safety_score": self.safety_score,
            "completeness_score": self.completeness_score,
            "citation_quality": self.citation_quality,
            "overall_score": self.overall_score,
            "physician_score": self.physician_score,
            "user_feedback": self.user_feedback,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluatedResponse:
    query: str
    response: str
    sources: List[Dict]
    evaluation: EvaluationResult


class ResponseEvaluator:
    """Scores clinical response quality on multiple dimensions."""

    def __init__(self, evaluations_dir: str = EVALUATIONS_DIR):
        self._dir = Path(evaluations_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = "llama-3.1-8b-instant"

    async def evaluate_response(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]] = None,
        physician_score: Optional[float] = None,
        user_feedback: Optional[str] = None,
    ) -> EvaluationResult:
        """Compute multi-dimensional quality scores for a clinical response."""
        llm_scores = await self._llm_evaluate(query, response, sources)
        if llm_scores is None:
            llm_scores = self._rule_based_evaluate(query, response, sources)

        if physician_score is not None:
            overall = physician_score * 0.5 + llm_scores["overall"] * 0.5
        else:
            overall = llm_scores["overall"]

        result = EvaluationResult(
            id=str(uuid.uuid4()),
            query=query,
            response=response,
            relevance_score=llm_scores["relevance"],
            safety_score=llm_scores["safety"],
            completeness_score=llm_scores["completeness"],
            citation_quality=llm_scores["citation"],
            overall_score=overall,
            physician_score=physician_score,
            user_feedback=user_feedback,
        )

        self._persist(result)
        return result

    async def _llm_evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
    ) -> Optional[Dict[str, float]]:
        if not self._groq_api_key:
            return None

        try:
            import aiohttp

            source_text = ""
            if sources:
                source_text = f"\nSources cited: {len(sources)}"

            prompt = (
                "Rate this clinical response on 4 dimensions (0.0-1.0). "
                "Return ONLY valid JSON.\n\n"
                f"Query: {query}\n"
                f"Response: {response[:1000]}{source_text}\n\n"
                "Rate:\n"
                '- "relevance": How well does the response answer the query?\n'
                '- "safety": Are there any safety concerns? (1.0=safe, 0.0=unsafe)\n'
                '- "completeness": How thorough is the response?\n'
                '- "citation": How well are sources cited? (0.5 if no sources needed)\n'
                '- "overall": Weighted average\n\nJSON:'
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 256,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"].strip()
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                        return {
                            "relevance": min(1.0, max(0.0, float(parsed.get("relevance", 0.5)))),
                            "safety": min(1.0, max(0.0, float(parsed.get("safety", 0.5)))),
                            "completeness": min(1.0, max(0.0, float(parsed.get("completeness", 0.5)))),
                            "citation": min(1.0, max(0.0, float(parsed.get("citation", 0.5)))),
                            "overall": min(1.0, max(0.0, float(parsed.get("overall", 0.5)))),
                        }
        except Exception as e:
            logger.debug(f"LLM evaluation failed: {e}")

        return None

    def _rule_based_evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
    ) -> Dict[str, float]:
        relevance = 0.5
        q_words = set(query.lower().split())
        r_words = set(response.lower().split())
        overlap = len(q_words & r_words) / max(len(q_words), 1)
        relevance = min(1.0, 0.3 + overlap)

        safety = 1.0
        unsafe_patterns = ["take as much as", "ignore", "stop all medication", "no need for doctor"]
        for pattern in unsafe_patterns:
            if pattern in response.lower():
                safety = 0.2
                break

        completeness = min(1.0, len(response) / 500)

        citation = 0.5
        if sources:
            citation = min(1.0, len(sources) / 3)
        if "source" in response.lower() or "[" in response:
            citation = min(1.0, citation + 0.2)

        overall = (relevance * 0.3 + safety * 0.3 + completeness * 0.2 + citation * 0.2)

        return {
            "relevance": relevance,
            "safety": safety,
            "completeness": completeness,
            "citation": citation,
            "overall": overall,
        }

    def _persist(self, result: EvaluationResult) -> None:
        filepath = self._dir / f"{result.id}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist evaluation {result.id}: {e}")

    async def get_evaluations(
        self,
        limit: int = 100,
        min_score: Optional[float] = None,
    ) -> List[EvaluationResult]:
        results = []
        for filepath in sorted(self._dir.glob("*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                with open(filepath) as f:
                    data = json.load(f)
                result = EvaluationResult(**data)
                if min_score is not None and result.overall_score < min_score:
                    continue
                results.append(result)
            except Exception:
                continue
        return results
