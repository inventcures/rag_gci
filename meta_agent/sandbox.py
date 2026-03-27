"""Safe sandbox for testing proposed improvements before deployment."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Awaitable

from meta_agent.safety_constraints import SafetyConstraintChecker, Modification

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    query: str
    expected_min_score: float
    reference_response: Optional[str] = None
    physician_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxResult:
    improvement_id: str
    passed: bool
    before_avg_score: float
    after_avg_score: float
    improvement_delta: float
    test_cases_run: int
    test_cases_passed: int
    safety_violations: List[str]
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "improvement_id": self.improvement_id,
            "passed": self.passed,
            "before_avg_score": self.before_avg_score,
            "after_avg_score": self.after_avg_score,
            "improvement_delta": self.improvement_delta,
            "test_cases_run": self.test_cases_run,
            "test_cases_passed": self.test_cases_passed,
            "safety_violations": self.safety_violations,
        }


class ImprovementSandbox:
    """
    Tests proposed improvements against held-out examples in isolation.
    Compares before/after metrics and checks safety constraints.
    """

    def __init__(
        self,
        safety_checker: Optional[SafetyConstraintChecker] = None,
        min_improvement_threshold: float = 0.05,
    ):
        self._safety_checker = safety_checker or SafetyConstraintChecker()
        self._threshold = min_improvement_threshold

    async def test_improvement(
        self,
        improvement_id: str,
        test_cases: List[TestCase],
        before_scorer: Callable[[str], Awaitable[float]],
        after_scorer: Callable[[str], Awaitable[float]],
        modifications: Optional[List[Modification]] = None,
    ) -> SandboxResult:
        """
        Run modified prompt/retrieval against test cases and compare metrics.
        before_scorer: scores a query with current configuration.
        after_scorer: scores a query with the proposed modification applied.
        """
        safety_violations = []
        if modifications:
            for mod in modifications:
                if not self._safety_checker.validate_modification(mod):
                    safety_violations.append(
                        f"Constraint violation: {mod.field} ({mod.old_value} -> {mod.new_value})"
                    )

        if safety_violations:
            return SandboxResult(
                improvement_id=improvement_id,
                passed=False,
                before_avg_score=0.0,
                after_avg_score=0.0,
                improvement_delta=0.0,
                test_cases_run=0,
                test_cases_passed=0,
                safety_violations=safety_violations,
            )

        before_scores = []
        after_scores = []
        details = []
        cases_passed = 0

        for tc in test_cases:
            try:
                before_score = await before_scorer(tc.query)
                after_score = await after_scorer(tc.query)

                before_scores.append(before_score)
                after_scores.append(after_score)

                passed = after_score >= tc.expected_min_score
                if passed:
                    cases_passed += 1

                details.append({
                    "query": tc.query[:100],
                    "before": before_score,
                    "after": after_score,
                    "delta": after_score - before_score,
                    "passed": passed,
                })
            except Exception as e:
                logger.warning(f"Sandbox test case failed: {e}")
                details.append({
                    "query": tc.query[:100],
                    "error": str(e),
                    "passed": False,
                })

        if not before_scores:
            return SandboxResult(
                improvement_id=improvement_id,
                passed=False,
                before_avg_score=0.0,
                after_avg_score=0.0,
                improvement_delta=0.0,
                test_cases_run=len(test_cases),
                test_cases_passed=0,
                safety_violations=[],
                details=details,
            )

        before_avg = sum(before_scores) / len(before_scores)
        after_avg = sum(after_scores) / len(after_scores)
        delta = after_avg - before_avg

        overall_pass = (
            delta >= self._threshold
            and cases_passed >= len(test_cases) * 0.6
            and not safety_violations
        )

        logger.info(
            f"Sandbox result for {improvement_id}: "
            f"before={before_avg:.3f} after={after_avg:.3f} "
            f"delta={delta:.3f} passed={overall_pass}"
        )

        return SandboxResult(
            improvement_id=improvement_id,
            passed=overall_pass,
            before_avg_score=before_avg,
            after_avg_score=after_avg,
            improvement_delta=delta,
            test_cases_run=len(test_cases),
            test_cases_passed=cases_passed,
            safety_violations=safety_violations,
            details=details,
        )
