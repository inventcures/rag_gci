"""HyperAgent-inspired clinical meta-agent for self-improving response quality."""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from meta_agent.safety_constraints import SafetyConstraintChecker, SAFETY_CONSTRAINTS
from meta_agent.evaluator import ResponseEvaluator, EvaluationResult, EvaluatedResponse
from meta_agent.improvement_archive import ImprovementArchive, Improvement
from meta_agent.sandbox import ImprovementSandbox, SandboxResult, TestCase

logger = logging.getLogger(__name__)


@dataclass
class MetaImprovement:
    """Improvement to the improvement procedure itself."""
    strategy_weights: Dict[str, float]
    reason: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.utcnow().timestamp()


class ClinicalMetaAgent:
    """
    Self-improving clinical quality agent.

    Evaluates batches of physician-scored responses, identifies patterns
    in high/low scoring answers, generates prompt and retrieval modifications,
    tests them in a sandbox, and archives successful improvements.

    Safety constraints are inviolable: emergency thresholds only increase,
    dosage validation is read-only, handoff triggers are append-only.
    """

    def __init__(
        self,
        evaluator: Optional[ResponseEvaluator] = None,
        archive: Optional[ImprovementArchive] = None,
        sandbox: Optional[ImprovementSandbox] = None,
        safety_checker: Optional[SafetyConstraintChecker] = None,
        improvement_threshold: float = 0.05,
    ):
        self._evaluator = evaluator or ResponseEvaluator()
        self._archive = archive or ImprovementArchive()
        self._sandbox = sandbox or ImprovementSandbox(min_improvement_threshold=improvement_threshold)
        self._safety_checker = safety_checker or SafetyConstraintChecker()
        self._threshold = improvement_threshold
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = "llama-3.1-8b-instant"

        self._strategy_weights = {
            "prompt_modification": 0.4,
            "retrieval_weight_adjustment": 0.3,
            "context_length_tuning": 0.2,
            "safety_threshold_tightening": 0.1,
        }

    async def evaluate_and_improve(
        self,
        batch: List[EvaluatedResponse],
    ) -> List[Improvement]:
        """
        Analyze a batch of evaluated responses, identify patterns,
        generate improvements, test in sandbox, and archive successes.
        """
        if not batch:
            return []

        high_scoring = [b for b in batch if b.evaluation.overall_score >= 0.7]
        low_scoring = [b for b in batch if b.evaluation.overall_score < 0.5]

        if not low_scoring:
            logger.info("No low-scoring responses to improve")
            return []

        patterns = await self._identify_patterns(high_scoring, low_scoring)
        candidates = await self._generate_improvements(patterns, low_scoring)

        archived = []
        for candidate in candidates:
            test_cases = self._build_test_cases(batch)
            result = await self._sandbox.test_improvement(
                improvement_id=candidate.id,
                test_cases=test_cases,
                before_scorer=self._baseline_scorer,
                after_scorer=self._baseline_scorer,
            )

            if result.passed:
                await self._archive.archive_improvement(candidate)
                archived.append(candidate)
                logger.info(
                    f"Improvement {candidate.id} archived: "
                    f"delta={result.improvement_delta:.4f}"
                )
            else:
                logger.info(
                    f"Improvement {candidate.id} rejected: "
                    f"delta={result.improvement_delta:.4f} "
                    f"violations={result.safety_violations}"
                )

        return archived

    async def meta_improve(self) -> MetaImprovement:
        """
        Improve the improvement procedure itself.
        Analyzes which improvement strategies produce the best results
        and adjusts strategy weights accordingly.
        """
        improvements = await self._archive.get_improvements(limit=50)
        if not improvements:
            return MetaImprovement(
                strategy_weights=dict(self._strategy_weights),
                reason="No improvements in archive yet",
            )

        strategy_deltas: Dict[str, List[float]] = {}
        for imp in improvements:
            ct = imp.change_type
            if ct not in strategy_deltas:
                strategy_deltas[ct] = []
            strategy_deltas[ct].append(imp.improvement_delta)

        new_weights = dict(self._strategy_weights)
        for strategy, deltas in strategy_deltas.items():
            if strategy in new_weights and deltas:
                avg_delta = sum(deltas) / len(deltas)
                if avg_delta > self._threshold:
                    new_weights[strategy] = min(0.6, new_weights[strategy] + 0.05)
                elif avg_delta < 0:
                    new_weights[strategy] = max(0.05, new_weights[strategy] - 0.05)

        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        old_weights = dict(self._strategy_weights)
        self._strategy_weights = new_weights

        reason = (
            f"Adjusted strategy weights based on {len(improvements)} archived improvements. "
            f"Changes: {self._diff_weights(old_weights, new_weights)}"
        )

        logger.info(f"Meta-improvement: {reason}")
        return MetaImprovement(strategy_weights=new_weights, reason=reason)

    async def _identify_patterns(
        self,
        high: List[EvaluatedResponse],
        low: List[EvaluatedResponse],
    ) -> Dict[str, Any]:
        """Identify patterns differentiating high and low scoring responses."""
        patterns: Dict[str, Any] = {
            "low_safety": [],
            "low_relevance": [],
            "low_citation": [],
            "low_completeness": [],
        }

        for resp in low:
            e = resp.evaluation
            if e.safety_score < 0.5:
                patterns["low_safety"].append(resp.query)
            if e.relevance_score < 0.5:
                patterns["low_relevance"].append(resp.query)
            if e.citation_quality < 0.3:
                patterns["low_citation"].append(resp.query)
            if e.completeness_score < 0.4:
                patterns["low_completeness"].append(resp.query)

        avg_high = sum(r.evaluation.overall_score for r in high) / max(len(high), 1)
        avg_low = sum(r.evaluation.overall_score for r in low) / max(len(low), 1)
        patterns["score_gap"] = avg_high - avg_low

        return patterns

    async def _generate_improvements(
        self,
        patterns: Dict[str, Any],
        low_scoring: List[EvaluatedResponse],
    ) -> List[Improvement]:
        if not self._groq_api_key:
            return self._rule_based_improvements(patterns)

        try:
            import aiohttp

            prompt = (
                "As a clinical AI improvement agent, suggest improvements "
                "to address these patterns in low-scoring responses. "
                "Return ONLY a JSON array of improvements:\n"
                '[{"domain": "...", "change_type": "prompt_modification|retrieval_weight_adjustment|...", '
                '"description": "what to change and why"}]\n\n'
                f"Patterns: {json.dumps(patterns, default=str)[:1000]}\n\n"
                f"Example low queries: {[r.query[:100] for r in low_scoring[:3]]}\n\nJSON:"
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
                        "temperature": 0.3,
                        "max_tokens": 512,
                    },
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        return self._rule_based_improvements(patterns)

                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"].strip()
                    start = raw.find("[")
                    end = raw.rfind("]") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                        return [
                            Improvement(
                                id=str(uuid.uuid4()),
                                domain=item.get("domain", "general"),
                                change_type=item.get("change_type", "prompt_modification"),
                                description=item.get("description", ""),
                                before_metrics={},
                                after_metrics={},
                                improvement_delta=0.0,
                            )
                            for item in parsed
                        ]
        except Exception as e:
            logger.debug(f"LLM improvement generation failed: {e}")

        return self._rule_based_improvements(patterns)

    def _rule_based_improvements(
        self,
        patterns: Dict[str, Any],
    ) -> List[Improvement]:
        improvements = []

        if patterns.get("low_relevance"):
            improvements.append(Improvement(
                id=str(uuid.uuid4()),
                domain="retrieval",
                change_type="retrieval_weight_adjustment",
                description=(
                    f"Increase retrieval top-k or lower similarity threshold "
                    f"for {len(patterns['low_relevance'])} low-relevance queries"
                ),
                before_metrics={},
                after_metrics={},
                improvement_delta=0.0,
            ))

        if patterns.get("low_citation"):
            improvements.append(Improvement(
                id=str(uuid.uuid4()),
                domain="prompt",
                change_type="prompt_modification",
                description=(
                    "Add explicit citation instructions to the prompt template"
                ),
                before_metrics={},
                after_metrics={},
                improvement_delta=0.0,
            ))

        if patterns.get("low_completeness"):
            improvements.append(Improvement(
                id=str(uuid.uuid4()),
                domain="prompt",
                change_type="context_length_tuning",
                description=(
                    "Increase max_tokens or context window for more complete responses"
                ),
                before_metrics={},
                after_metrics={},
                improvement_delta=0.0,
            ))

        return improvements

    def _build_test_cases(
        self,
        batch: List[EvaluatedResponse],
    ) -> List[TestCase]:
        return [
            TestCase(
                query=resp.query,
                expected_min_score=max(0.3, resp.evaluation.overall_score - 0.1),
                physician_score=resp.evaluation.physician_score,
            )
            for resp in batch[:10]
        ]

    async def _baseline_scorer(self, query: str) -> float:
        """Placeholder scorer that returns a fixed baseline."""
        return 0.5

    def _diff_weights(
        self,
        old: Dict[str, float],
        new: Dict[str, float],
    ) -> str:
        changes = []
        for k in sorted(set(old) | set(new)):
            o = old.get(k, 0)
            n = new.get(k, 0)
            if abs(o - n) > 0.001:
                changes.append(f"{k}: {o:.2f}->{n:.2f}")
        return ", ".join(changes) if changes else "no changes"
