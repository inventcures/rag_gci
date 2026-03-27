"""Query complexity classifier for routing simple vs multi-hop queries."""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

MULTI_HOP_INDICATORS = [
    r"alternative",
    r"instead of",
    r"if.*then",
    r"while also",
    r"contraindicated",
    r"interaction with",
    r"given that",
    r"but also",
    r"combined with",
    r"switch from",
]

MEDICAL_ENTITY_PATTERNS = [
    r"\b(morphine|fentanyl|oxycodone|hydromorphone|methadone|tramadol|codeine)\b",
    r"\b(paracetamol|acetaminophen|ibuprofen|diclofenac|naproxen|aspirin)\b",
    r"\b(gabapentin|pregabalin|amitriptyline|duloxetine|carbamazepine)\b",
    r"\b(dexamethasone|prednisolone|methylprednisolone)\b",
    r"\b(ondansetron|metoclopramide|domperidone|haloperidol)\b",
    r"\b(laxative|bisacodyl|lactulose|senna|docusate)\b",
    r"\b(pain|nausea|vomiting|constipation|dyspnea|breathlessness)\b",
    r"\b(anxiety|depression|insomnia|delirium|agitation)\b",
    r"\b(liver|renal|kidney|hepatic|cardiac|pulmonary)\b",
    r"\b(cancer|malignant|tumor|metastatic|palliative)\b",
    r"\b(impairment|insufficiency|failure|obstruction)\b",
]

CONSTRAINT_PHRASES = [
    r"who (?:is|has|already)",
    r"patient with",
    r"in (?:a )?patient",
    r"history of",
    r"currently (?:on|taking)",
    r"cannot (?:take|tolerate)",
    r"allergic to",
    r"renal (?:impairment|failure|insufficiency)",
    r"hepatic (?:impairment|failure|insufficiency)",
    r"liver (?:impairment|failure|disease)",
    r"bowel obstruction",
    r"elderly patient",
]


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"


@dataclass
class ClassificationResult:
    complexity: QueryComplexity
    indicator_count: int
    entity_count: int
    constraint_count: int
    matched_indicators: List[str]


class QueryComplexityClassifier:
    """Routes queries to simple (current pipeline) or multi-hop (Context-1)."""

    def __init__(
        self,
        multi_hop_threshold: int = 2,
        entity_threshold: int = 2,
    ):
        self._multi_hop_threshold = multi_hop_threshold
        self._entity_threshold = entity_threshold
        self._indicator_patterns = [
            re.compile(p, re.IGNORECASE) for p in MULTI_HOP_INDICATORS
        ]
        self._entity_patterns = [
            re.compile(p, re.IGNORECASE) for p in MEDICAL_ENTITY_PATTERNS
        ]
        self._constraint_patterns = [
            re.compile(p, re.IGNORECASE) for p in CONSTRAINT_PHRASES
        ]

    def classify(self, query: str) -> QueryComplexity:
        result = self.classify_detailed(query)
        return result.complexity

    def classify_detailed(self, query: str) -> ClassificationResult:
        matched_indicators = []
        for pattern in self._indicator_patterns:
            if pattern.search(query):
                matched_indicators.append(pattern.pattern)

        entity_matches = set()
        for pattern in self._entity_patterns:
            for match in pattern.finditer(query):
                entity_matches.add(match.group(0).lower())

        constraint_count = sum(
            1 for p in self._constraint_patterns if p.search(query)
        )

        indicator_count = len(matched_indicators)
        entity_count = len(entity_matches)

        is_multi_hop = (
            indicator_count >= self._multi_hop_threshold
            or (entity_count >= self._entity_threshold and constraint_count >= 1)
            or (indicator_count >= 1 and entity_count >= self._entity_threshold)
        )

        complexity = QueryComplexity.MULTI_HOP if is_multi_hop else QueryComplexity.SIMPLE

        logger.info(
            f"Query classified as {complexity.value}: "
            f"indicators={indicator_count}, entities={entity_count}, "
            f"constraints={constraint_count}"
        )

        return ClassificationResult(
            complexity=complexity,
            indicator_count=indicator_count,
            entity_count=entity_count,
            constraint_count=constraint_count,
            matched_indicators=matched_indicators,
        )
