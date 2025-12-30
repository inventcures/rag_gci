"""
Clinical Validation Pipeline for Palli Sahayak

Provides multi-layer validation for medical responses:
1. Automated validation (medical entities, dosages, contraindications)
2. Expert review sampling (5% random sampling for human review)
3. User feedback integration (helpfulness ratings, issue reporting)
4. Metrics tracking (accuracy, hallucination rate, expert agreement)
"""

from .validator import ClinicalValidator, ValidationResult
from .expert_sampling import ExpertSampler, SampleRecord
from .feedback import FeedbackCollector, UserFeedback
from .metrics import ValidationMetrics

__all__ = [
    "ClinicalValidator",
    "ValidationResult",
    "ExpertSampler",
    "SampleRecord",
    "FeedbackCollector",
    "UserFeedback",
    "ValidationMetrics",
]
