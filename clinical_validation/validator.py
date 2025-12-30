"""
Clinical Validator - Automated validation for medical responses.

Performs multi-layer validation:
1. Medical entity verification
2. Dosage range validation
3. Contraindication detection
4. Hallucination detection (response grounded in sources)
5. Safety check (emergency indicators, harmful advice)
"""

import re
import logging
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    MEDICAL_ENTITY = "medical_entity"
    DOSAGE = "dosage"
    CONTRAINDICATION = "contraindication"
    HALLUCINATION = "hallucination"
    SAFETY = "safety"
    CITATION = "citation"
    SCOPE = "scope"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    category: ValidationCategory
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a response."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.level == ValidationLevel.CRITICAL for i in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(i.level == ValidationLevel.ERROR for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "confidence_score": self.confidence_score,
            "issues": [
                {
                    "category": i.category.value,
                    "level": i.level.value,
                    "message": i.message,
                    "details": i.details,
                    "suggestion": i.suggestion
                }
                for i in self.issues
            ],
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "validation_time_ms": self.validation_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


class ClinicalValidator:
    """
    Multi-layer clinical validation for palliative care responses.

    Validates:
    - Medical terminology accuracy
    - Medication dosages within safe ranges
    - Contraindication warnings present
    - Response grounded in provided sources
    - No harmful or dangerous advice
    """

    # Common palliative care medications with safe dosage ranges
    MEDICATION_DOSAGES: Dict[str, Dict[str, Any]] = {
        "morphine": {
            "oral_mg_per_dose": (2.5, 200),  # min, max
            "iv_mg_per_dose": (1, 50),
            "unit": "mg",
            "requires_prescription": True,
            "controlled": True
        },
        "oxycodone": {
            "oral_mg_per_dose": (2.5, 80),
            "unit": "mg",
            "requires_prescription": True,
            "controlled": True
        },
        "fentanyl": {
            "patch_mcg_per_hour": (12, 100),
            "unit": "mcg",
            "requires_prescription": True,
            "controlled": True
        },
        "paracetamol": {
            "oral_mg_per_dose": (325, 1000),
            "max_daily_mg": 4000,
            "unit": "mg",
            "requires_prescription": False
        },
        "acetaminophen": {
            "oral_mg_per_dose": (325, 1000),
            "max_daily_mg": 4000,
            "unit": "mg",
            "requires_prescription": False
        },
        "ibuprofen": {
            "oral_mg_per_dose": (200, 800),
            "max_daily_mg": 3200,
            "unit": "mg",
            "requires_prescription": False
        },
        "ondansetron": {
            "oral_mg_per_dose": (4, 8),
            "iv_mg_per_dose": (4, 8),
            "unit": "mg"
        },
        "metoclopramide": {
            "oral_mg_per_dose": (5, 20),
            "unit": "mg"
        },
        "haloperidol": {
            "oral_mg_per_dose": (0.5, 5),
            "unit": "mg"
        },
        "lorazepam": {
            "oral_mg_per_dose": (0.5, 4),
            "unit": "mg",
            "controlled": True
        },
        "dexamethasone": {
            "oral_mg_per_dose": (0.5, 16),
            "unit": "mg"
        },
        "bisacodyl": {
            "oral_mg_per_dose": (5, 15),
            "unit": "mg"
        },
        "lactulose": {
            "oral_ml_per_dose": (15, 45),
            "unit": "ml"
        }
    }

    # Medical conditions relevant to palliative care
    PALLIATIVE_CONDITIONS: Set[str] = {
        "cancer", "tumor", "malignancy", "metastasis", "metastatic",
        "terminal", "end-stage", "advanced", "chronic",
        "copd", "heart failure", "renal failure", "liver failure",
        "als", "motor neuron disease", "dementia", "alzheimer",
        "parkinson", "huntington", "multiple sclerosis", "ms",
        "hiv", "aids", "stroke", "paralysis"
    }

    # Symptoms commonly addressed in palliative care
    PALLIATIVE_SYMPTOMS: Set[str] = {
        "pain", "nausea", "vomiting", "constipation", "diarrhea",
        "dyspnea", "breathlessness", "shortness of breath",
        "fatigue", "weakness", "anorexia", "cachexia",
        "anxiety", "depression", "insomnia", "delirium",
        "edema", "swelling", "ascites", "bleeding",
        "cough", "hiccups", "itching", "pruritus",
        "mouth sores", "mucositis", "dry mouth", "xerostomia",
        "bedsore", "pressure ulcer", "wound"
    }

    # Dangerous phrases that should trigger safety alerts
    DANGEROUS_PHRASES: List[str] = [
        "take as much as you want",
        "no limit",
        "cannot overdose",
        "completely safe",
        "guaranteed cure",
        "stop all medications",
        "ignore doctor",
        "don't need a doctor",
        "self-prescribe",
        "buy online without prescription"
    ]

    # Required disclaimers for medical advice
    REQUIRED_DISCLAIMER_PATTERNS: List[str] = [
        r"consult.*doctor",
        r"consult.*physician",
        r"consult.*healthcare",
        r"medical.*advice",
        r"professional.*guidance",
        r"speak.*doctor",
        r"see.*doctor"
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        require_citations: bool = True,
        require_disclaimer: bool = True
    ):
        """
        Initialize the clinical validator.

        Args:
            strict_mode: If True, any warning becomes an error
            require_citations: Require source citations for medical claims
            require_disclaimer: Require medical disclaimer in responses
        """
        self.strict_mode = strict_mode
        self.require_citations = require_citations
        self.require_disclaimer = require_disclaimer

        logger.info(
            f"ClinicalValidator initialized - strict={strict_mode}, "
            f"citations={require_citations}, disclaimer={require_disclaimer}"
        )

    def validate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a response against clinical safety standards.

        Args:
            query: Original user query
            response: Generated response to validate
            sources: Source documents used for the response
            context: Additional context from RAG retrieval

        Returns:
            ValidationResult with validation outcome and issues
        """
        import time
        start_time = time.time()

        issues: List[ValidationIssue] = []
        checks_passed: List[str] = []
        checks_failed: List[str] = []

        # Run all validation checks
        checks = [
            ("safety_check", self._check_safety),
            ("dosage_check", self._check_dosages),
            ("hallucination_check", self._check_hallucination),
            ("citation_check", self._check_citations),
            ("disclaimer_check", self._check_disclaimer),
            ("scope_check", self._check_scope),
        ]

        for check_name, check_func in checks:
            try:
                check_issues = check_func(query, response, sources, context)
                if check_issues:
                    issues.extend(check_issues)
                    checks_failed.append(check_name)
                else:
                    checks_passed.append(check_name)
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {e}")
                issues.append(ValidationIssue(
                    category=ValidationCategory.SAFETY,
                    level=ValidationLevel.WARNING,
                    message=f"Validation check {check_name} could not complete",
                    details={"error": str(e)}
                ))

        # Calculate confidence score
        confidence = self._calculate_confidence(issues, checks_passed, checks_failed)

        # Determine validity
        is_valid = not any(
            i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for i in issues
        )

        if self.strict_mode and any(i.level == ValidationLevel.WARNING for i in issues):
            is_valid = False

        validation_time = (time.time() - start_time) * 1000

        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence,
            issues=issues,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            validation_time_ms=validation_time
        )

        logger.info(
            f"Validation complete - valid={is_valid}, confidence={confidence:.2f}, "
            f"issues={len(issues)}, time={validation_time:.1f}ms"
        )

        return result

    def _check_safety(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Check for dangerous or harmful content."""
        issues = []
        response_lower = response.lower()

        for phrase in self.DANGEROUS_PHRASES:
            if phrase in response_lower:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SAFETY,
                    level=ValidationLevel.CRITICAL,
                    message=f"Potentially dangerous phrase detected: '{phrase}'",
                    suggestion="Remove or rephrase this advice to ensure patient safety"
                ))

        # Check for unsupervised medication changes
        if re.search(r"(stop|discontinue|quit).*medication.*without", response_lower):
            issues.append(ValidationIssue(
                category=ValidationCategory.SAFETY,
                level=ValidationLevel.ERROR,
                message="Response suggests stopping medication without supervision",
                suggestion="Always recommend consulting healthcare provider before medication changes"
            ))

        return issues

    def _check_dosages(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Validate medication dosages are within safe ranges."""
        issues = []
        response_lower = response.lower()

        # Extract dosage mentions
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g)\s+(?:of\s+)?(\w+)'
        matches = re.findall(dosage_pattern, response_lower)

        for dose_str, unit, med_name in matches:
            dose = float(dose_str)

            # Check against known medications
            for med_key, med_info in self.MEDICATION_DOSAGES.items():
                if med_key in med_name or med_name in med_key:
                    # Check oral dosage
                    oral_key = f"oral_{unit}_per_dose"
                    if oral_key in med_info:
                        min_dose, max_dose = med_info[oral_key]
                        if dose < min_dose:
                            issues.append(ValidationIssue(
                                category=ValidationCategory.DOSAGE,
                                level=ValidationLevel.WARNING,
                                message=f"Low dose of {med_key}: {dose}{unit} (typical: {min_dose}-{max_dose}{unit})",
                                details={"medication": med_key, "dose": dose, "unit": unit}
                            ))
                        elif dose > max_dose:
                            issues.append(ValidationIssue(
                                category=ValidationCategory.DOSAGE,
                                level=ValidationLevel.ERROR,
                                message=f"High dose of {med_key}: {dose}{unit} exceeds typical max of {max_dose}{unit}",
                                details={"medication": med_key, "dose": dose, "unit": unit},
                                suggestion=f"Verify dosage - typical range is {min_dose}-{max_dose}{unit}"
                            ))

                    # Check if controlled substance mentioned without prescription warning
                    if med_info.get("controlled") and "prescription" not in response_lower:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.SAFETY,
                            level=ValidationLevel.WARNING,
                            message=f"Controlled substance {med_key} mentioned without prescription requirement note",
                            suggestion="Add note that this medication requires a prescription"
                        ))
                    break

        return issues

    def _check_hallucination(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Check if response is grounded in provided sources."""
        issues = []

        if not sources and not context:
            # No sources to verify against
            return issues

        # Build source content for verification
        source_text = ""
        if sources:
            source_text = " ".join([
                s.get("content", "") or s.get("text", "")
                for s in sources
            ]).lower()
        if context:
            source_text += " " + context.lower()

        if not source_text.strip():
            return issues

        # Check for specific medical claims not in sources
        response_lower = response.lower()

        # Extract key medical terms from response
        medical_terms = set()
        for term in self.PALLIATIVE_CONDITIONS | self.PALLIATIVE_SYMPTOMS:
            if term in response_lower:
                medical_terms.add(term)

        # Check medication mentions
        med_pattern = r'\b(' + '|'.join(self.MEDICATION_DOSAGES.keys()) + r')\b'
        meds_in_response = set(re.findall(med_pattern, response_lower))
        meds_in_sources = set(re.findall(med_pattern, source_text))

        unsupported_meds = meds_in_response - meds_in_sources
        if unsupported_meds:
            issues.append(ValidationIssue(
                category=ValidationCategory.HALLUCINATION,
                level=ValidationLevel.WARNING,
                message=f"Medications mentioned not found in sources: {', '.join(unsupported_meds)}",
                details={"unsupported_medications": list(unsupported_meds)},
                suggestion="Verify these medications are appropriate or add source citations"
            ))

        # Check for percentage/statistic claims
        stat_pattern = r'(\d+(?:\.\d+)?)\s*%'
        stats_in_response = re.findall(stat_pattern, response)
        if stats_in_response:
            # Check if any statistics are in sources
            stats_in_sources = re.findall(stat_pattern, source_text)
            for stat in stats_in_response:
                if stat not in stats_in_sources:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.HALLUCINATION,
                        level=ValidationLevel.WARNING,
                        message=f"Statistic '{stat}%' not found in sources",
                        suggestion="Verify this statistic or remove if not from sources"
                    ))

        return issues

    def _check_citations(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Check for proper source citations."""
        issues = []

        if not self.require_citations:
            return issues

        # Check for citation patterns
        citation_patterns = [
            r'\[source[s]?:',
            r'\{retrieved from:',
            r'according to',
            r'based on',
            r'from the.*document',
            r'the.*guideline.*states'
        ]

        has_citation = any(
            re.search(pattern, response.lower())
            for pattern in citation_patterns
        )

        if not has_citation and sources:
            issues.append(ValidationIssue(
                category=ValidationCategory.CITATION,
                level=ValidationLevel.INFO,
                message="Response does not explicitly cite sources",
                suggestion="Consider adding source references for credibility"
            ))

        return issues

    def _check_disclaimer(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Check for required medical disclaimers."""
        issues = []

        if not self.require_disclaimer:
            return issues

        response_lower = response.lower()

        has_disclaimer = any(
            re.search(pattern, response_lower)
            for pattern in self.REQUIRED_DISCLAIMER_PATTERNS
        )

        if not has_disclaimer:
            # Check if response contains medical advice
            contains_advice = any([
                "should" in response_lower,
                "recommend" in response_lower,
                "take" in response_lower and any(
                    med in response_lower for med in self.MEDICATION_DOSAGES
                ),
                "treatment" in response_lower,
                "therapy" in response_lower
            ])

            if contains_advice:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SAFETY,
                    level=ValidationLevel.WARNING,
                    message="Medical advice given without consultation disclaimer",
                    suggestion="Add 'Please consult your healthcare provider for personalized guidance'"
                ))

        return issues

    def _check_scope(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict]],
        context: Optional[str]
    ) -> List[ValidationIssue]:
        """Check if response stays within palliative care scope."""
        issues = []
        response_lower = response.lower()

        # Check for out-of-scope content
        out_of_scope_indicators = [
            ("diagnostic", "providing diagnosis"),
            ("prognosis", "predicting outcomes"),
            ("cure", "claiming cures"),
            ("surgery", "surgical recommendations")
        ]

        for indicator, description in out_of_scope_indicators:
            if indicator in response_lower:
                # Allow if properly qualified
                qualified = any([
                    "may" in response_lower,
                    "might" in response_lower,
                    "could" in response_lower,
                    "consult" in response_lower,
                    "doctor" in response_lower
                ])

                if not qualified:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SCOPE,
                        level=ValidationLevel.INFO,
                        message=f"Response may be {description} without proper qualification",
                        suggestion="Ensure claims are appropriately qualified"
                    ))

        return issues

    def _calculate_confidence(
        self,
        issues: List[ValidationIssue],
        checks_passed: List[str],
        checks_failed: List[str]
    ) -> float:
        """Calculate confidence score based on validation results."""
        base_score = 1.0

        # Deduct for issues based on severity
        severity_weights = {
            ValidationLevel.INFO: 0.02,
            ValidationLevel.WARNING: 0.1,
            ValidationLevel.ERROR: 0.25,
            ValidationLevel.CRITICAL: 0.5
        }

        for issue in issues:
            base_score -= severity_weights.get(issue.level, 0.05)

        # Bonus for passed checks
        check_bonus = len(checks_passed) * 0.02
        base_score += min(check_bonus, 0.1)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_score))

    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a human-readable summary of validation result."""
        if result.is_valid and result.confidence_score >= 0.9:
            status = "PASSED"
        elif result.is_valid:
            status = "PASSED with warnings"
        elif result.has_critical_issues:
            status = "FAILED (critical issues)"
        else:
            status = "FAILED"

        summary = f"Validation {status} (confidence: {result.confidence_score:.0%})"

        if result.issues:
            summary += f"\nIssues ({len(result.issues)}):"
            for issue in result.issues[:5]:  # Show first 5
                summary += f"\n  [{issue.level.value.upper()}] {issue.message}"

        return summary
