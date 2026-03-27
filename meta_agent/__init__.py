"""
Meta Agent Module for Palli Sahayak.
HyperAgent-inspired self-improving clinical quality system.
Evaluates response quality, generates improvements, tests in sandbox,
and archives successful stepping stones.
"""

from meta_agent.safety_constraints import SafetyConstraintChecker, SAFETY_CONSTRAINTS
from meta_agent.evaluator import ResponseEvaluator, EvaluationResult
from meta_agent.improvement_archive import ImprovementArchive, Improvement
from meta_agent.sandbox import ImprovementSandbox, SandboxResult
from meta_agent.clinical_meta_agent import ClinicalMetaAgent

__all__ = [
    "SafetyConstraintChecker",
    "SAFETY_CONSTRAINTS",
    "ResponseEvaluator",
    "EvaluationResult",
    "ImprovementArchive",
    "Improvement",
    "ImprovementSandbox",
    "SandboxResult",
    "ClinicalMetaAgent",
]

__version__ = "1.0.0"
