"""
Knowledge-graph cross-reference check (EVAH protocol item K26).

After an answer is drafted from retrieved passages, the clinical claims in it are
checked against the Neo4j knowledge graph:

    draft answer -> sentences -> entity mentions -> claims -> graph lookup -> verdicts

A claim links a medication to a symptom, side effect or condition with one of
three relations:

    treats          "Senna helps relieve constipation."
    side_effect     "Morphine commonly causes constipation."
    contraindicated "NSAIDs should be avoided in kidney failure."

and may be negated ("Morphine does not cause diarrhoea").

Verdicts:

    supported     the graph has the edge the claim asserts
    contradicted  the graph has an edge that says the opposite, e.g. the answer
                  says morphine relieves constipation but the graph records
                  constipation as a side effect of morphine
    unsupported   the graph has no edge either way, or does not know an entity

Extraction is deterministic pattern matching built from the same palliative-care
lexicon `knowledge_graph.EntityExtractor` uses. No LLM is called, so the drafted
answer never leaves the deployment for this check, and the cost is one graph
round trip. A relation is only read from the words between the two mentions, so
"Morphine can cause constipation, so give senna" does not become a claim that
senna causes constipation.

Policy (settings.kg_check_mode):

    off     not run (default; the study release is unchanged)
    shadow  run and log, answer unchanged
    flag    answer unchanged, a caution line appended when a claim is
            contradicted or unsupported
    block   sentences with contradicted claims (and unsupported ones when
            block_unsupported is set) are removed, with the caution line

If the graph is down or slow the check reports `unavailable` or `timeout`, the
answer goes out as drafted, and the result is logged. It never fails a turn.
"""

import asyncio
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retrieval.graph_store import GraphUnavailable

logger = logging.getLogger(__name__)

MAX_CLAIMS = 25
# Mentions further apart than this are not read as one claim.
MAX_SPAN_CHARS = 80
MIN_REMAINING_CHARS = 40

CAUTION_CONTRADICTED = (
    "Note: part of this answer conflicted with the clinical knowledge base and "
    "was removed. Please check with your doctor or palliative care team before "
    "acting on it."
)
CAUTION_FLAGGED_CONTRADICTED = (
    "Note: part of this answer conflicts with the clinical knowledge base. "
    "Please check with your doctor or palliative care team before acting on it."
)
CAUTION_UNSUPPORTED = (
    "Note: some statements in this answer could not be confirmed against the "
    "clinical knowledge base. Please check with your doctor or palliative care "
    "team before acting on them."
)
SAFE_FALLBACK = (
    "I could not confirm this information against the clinical knowledge base. "
    "Please consult your doctor or palliative care team."
)

_SIDE_EFFECT_CUE = re.compile(
    r"side[\s-]?effects?|adverse|\bcaus(?:e|es|ed|ing)\b|can make|\blead(?:s)? to\b|"
    r"\bresults? in\b|induced by|\bproduc(?:e|es)\b"
)
_CONTRA_CUE = re.compile(
    r"contraindicat|should not be (?:given|used|taken)|must not be|\bavoid(?:ed)?\b|"
    r"\bunsafe\b|not safe|not (?:be )?(?:given|used) (?:in|for|with)"
)
_TREATS_CUE = re.compile(
    r"\btreat|\breliev|\bhelp|\beas(?:e|es|ing)\b|\bmanag|\bcontrol|\breduc|"
    r"\balleviat|\bprevent|\bused for\b|first[\s-]line|\beffective\b|\bfor\b"
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|doesn't|does not|don't|do not|cannot|can't|isn't|won't|without|neither|nor)\b"
)
_TRAILING_INDUCED = re.compile(r"\b[\w]+-induced\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

TREATS_RELS = {"TREATS", "ALLEVIATES", "MANAGES", "PREVENTS"}


@dataclass
class Mention:
    name: str
    type: str  # Medication | Symptom | Condition
    start: int
    end: int


@dataclass
class Claim:
    kind: str  # treats | side_effect | contraindicated
    medication: str
    target: str
    target_type: str
    negated: bool
    sentence_index: int
    sentence: str
    verdict: str = "unchecked"  # supported | contradicted | unsupported | unchecked
    reason: str = ""

    def describe(self) -> str:
        verb = {
            "treats": "treats",
            "side_effect": "causes",
            "contraindicated": "is contraindicated in",
        }[self.kind]
        neg = "NOT " if self.negated else ""
        return f"{self.medication} {neg}{verb} {self.target}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # The sentence is answer text, which the study log scrubs separately;
        # the claim record keeps only entity names and verdicts.
        d.pop("sentence", None)
        d["claim"] = self.describe()
        return d


@dataclass
class KGCheckResult:
    mode: str
    status: str = "ok"  # ok | skipped | unavailable | timeout | error
    action: str = "none"  # none | logged | flagged | blocked
    claims: List[Claim] = field(default_factory=list)
    kg_version: str = ""
    latency_ms: float = 0.0
    removed_sentences: int = 0
    detail: str = ""

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.claims if c.verdict == verdict)

    @property
    def contradicted(self) -> int:
        return self.count("contradicted")

    @property
    def unsupported(self) -> int:
        return self.count("unsupported")

    @property
    def supported(self) -> int:
        return self.count("supported")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "action": self.action,
            "kg_version": self.kg_version,
            "latency_ms": round(self.latency_ms, 1),
            "claims_checked": len(self.claims),
            "supported": self.supported,
            "contradicted": self.contradicted,
            "unsupported": self.unsupported,
            "removed_sentences": self.removed_sentences,
            "detail": self.detail,
            "claims": [c.to_dict() for c in self.claims],
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """Finds medication-to-symptom/condition claims in English answer text."""

    def __init__(self, lexicon: Optional[Dict[str, List[Tuple[str, str]]]] = None):
        self.lexicon = lexicon or self._default_lexicon()
        self._compiled = {
            etype: [(re.compile(p, re.IGNORECASE), name) for p, name in patterns]
            for etype, patterns in self.lexicon.items()
        }

    @staticmethod
    def _default_lexicon() -> Dict[str, List[Tuple[str, str]]]:
        from knowledge_graph.entity_extractor import EntityExtractor

        # llm_provider="none": construct the pattern tables only, no client.
        ex = EntityExtractor(llm_provider="none", use_patterns=True)

        def flatten(groups: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
            return [pair for patterns in groups.values() for pair in patterns]

        return {
            "Medication": flatten(ex._medication_patterns),
            "Symptom": flatten(ex._symptom_patterns),
            "Condition": flatten(ex._condition_patterns),
        }

    def mentions(self, sentence: str) -> List[Mention]:
        found: List[Mention] = []
        for etype, patterns in self._compiled.items():
            for rx, name in patterns:
                for m in rx.finditer(sentence):
                    found.append(Mention(name, etype, m.start(), m.end()))
        # Overlaps keep the longest match: "bone pain" beats "pain".
        found.sort(key=lambda m: (m.start, -(m.end - m.start)))
        kept: List[Mention] = []
        for m in found:
            if kept and m.start < kept[-1].end:
                if (m.end - m.start) > (kept[-1].end - kept[-1].start):
                    kept[-1] = m
                continue
            kept.append(m)
        return kept

    @staticmethod
    def sentences(text: str) -> List[str]:
        return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text or "")) if s]

    def extract(self, text: str) -> List[Claim]:
        claims: List[Claim] = []
        seen = set()
        for idx, sentence in enumerate(self.sentences(text)):
            lower = sentence.lower()
            mentions = self.mentions(sentence)
            meds = [m for m in mentions if m.type == "Medication"]
            targets = [m for m in mentions if m.type in ("Symptom", "Condition")]
            for med in meds:
                for tgt in targets:
                    claim = self._claim_for(lower, med, tgt, idx, sentence)
                    if claim is None:
                        continue
                    key = (claim.kind, claim.medication, claim.target, claim.negated)
                    if key in seen:
                        continue
                    seen.add(key)
                    claims.append(claim)
                    if len(claims) >= MAX_CLAIMS:
                        return claims
        return claims

    def _claim_for(
        self, lower: str, med: Mention, tgt: Mention, idx: int, sentence: str
    ) -> Optional[Claim]:
        first, second = (med, tgt) if med.start < tgt.start else (tgt, med)
        span = lower[first.end : second.start]
        if len(span) > MAX_SPAN_CHARS or ";" in span:
            return None
        if med.start < tgt.start:
            # "prevents opioid-induced constipation": the -induced modifier
            # belongs to the target, not to the relation.
            span = _TRAILING_INDUCED.sub("", span)

        kind: Optional[str] = None
        if tgt.type == "Condition" and _CONTRA_CUE.search(span):
            kind = "contraindicated"
            negated = bool(re.search(r"\bnot contraindicat", span))
        elif _SIDE_EFFECT_CUE.search(span):
            kind = "side_effect"
            negated = bool(_NEGATION.search(span))
        elif _TREATS_CUE.search(span):
            kind = "treats"
            negated = bool(_NEGATION.search(span))
        if kind is None:
            return None
        return Claim(
            kind=kind,
            medication=med.name,
            target=tgt.name,
            target_type=tgt.type,
            negated=negated,
            sentence_index=idx,
            sentence=sentence,
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def judge(claim: Claim, a_known: bool, b_known: bool, rels: Sequence[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Verdict for one claim given what the graph holds between its two entities.

    `rels` holds (relationship type, start node name). Direction matters only for
    SIDE_EFFECT_OF, which is stored SideEffect -> Medication, and for TREATS-like
    edges, stored Medication -> Symptom.
    """
    med = claim.medication.lower()
    types = {t for t, _ in rels}
    treats = any(t in TREATS_RELS and (s or "").lower() == med for t, s in rels)
    causes = any(
        (t == "SIDE_EFFECT_OF" and (s or "").lower() != med)
        or (t == "CAUSES" and (s or "").lower() == med)
        for t, s in rels
    )
    contra = "CONTRAINDICATES" in types

    if claim.kind == "treats":
        if not claim.negated:
            if treats:
                return "supported", "graph records treatment"
            if contra:
                return "contradicted", "graph records a contraindication"
            if causes:
                return "contradicted", "graph records it as a side effect of the medication"
        else:
            if treats:
                return "contradicted", "graph records treatment"
    elif claim.kind == "side_effect":
        if not claim.negated:
            if causes:
                return "supported", "graph records the side effect"
            if treats:
                return "contradicted", "graph records the medication as treating it"
        else:
            if causes:
                return "contradicted", "graph records the side effect"
    elif claim.kind == "contraindicated":
        if not claim.negated and contra:
            return "supported", "graph records the contraindication"
        if claim.negated and contra:
            return "contradicted", "graph records a contraindication"

    if not a_known:
        return "unsupported", "medication not in graph"
    if not b_known:
        return "unsupported", f"{claim.target_type.lower()} not in graph"
    return "unsupported", "no relationship in graph"


class KGCrossReferenceChecker:
    """Runs extraction, one graph lookup, and the configured policy."""

    def __init__(
        self,
        graph: Any,
        mode: str = "off",
        timeout_s: float = 1.5,
        block_unsupported: bool = False,
        extractor: Optional[ClaimExtractor] = None,
    ):
        self.graph = graph
        self.mode = mode
        self.timeout_s = timeout_s
        self.block_unsupported = block_unsupported
        self._extractor = extractor

    @property
    def extractor(self) -> ClaimExtractor:
        if self._extractor is None:
            self._extractor = ClaimExtractor()
        return self._extractor

    async def check(self, answer: str) -> Tuple[str, KGCheckResult]:
        """Return the (possibly edited) answer and the check record."""
        started = time.perf_counter()
        result = KGCheckResult(mode=self.mode)
        try:
            if self.mode == "off":
                result.status = "skipped"
                result.detail = "kg_check disabled"
                return answer, result

            claims = self.extractor.extract(answer)
            result.claims = claims
            if not claims:
                result.status = "ok"
                result.detail = "no checkable claims"
                result.action = "logged" if self.mode == "shadow" else "none"
                return answer, result

            if self.graph is None or not self.graph.is_available():
                result.status = "unavailable"
                result.detail = "knowledge graph unavailable; answer sent unchecked"
                logger.warning("KG check skipped: knowledge graph unavailable")
                return answer, result

            try:
                rels, version = await asyncio.wait_for(
                    self._lookup(claims), timeout=self.timeout_s
                )
            except asyncio.TimeoutError:
                result.status = "timeout"
                result.detail = f"graph lookup exceeded {self.timeout_s}s; answer sent unchecked"
                logger.warning("KG check timed out after %.1fs", self.timeout_s)
                return answer, result
            except GraphUnavailable as e:
                result.status = "unavailable"
                result.detail = f"{e}; answer sent unchecked"
                logger.warning("KG check skipped: %s", e)
                return answer, result

            result.kg_version = version
            for claim, pr in zip(claims, rels):
                claim.verdict, claim.reason = judge(claim, pr.a_known, pr.b_known, pr.rels)

            return self._apply_policy(answer, result), result
        except Exception as e:  # never fail a user's turn on the checker
            logger.exception("KG check failed")
            result.status = "error"
            result.detail = f"{type(e).__name__}: {e}"
            return answer, result
        finally:
            result.latency_ms = (time.perf_counter() - started) * 1000
            if result.claims:
                logger.info(
                    "KG check mode=%s status=%s action=%s claims=%d supported=%d "
                    "contradicted=%d unsupported=%d (%.0f ms)",
                    result.mode, result.status, result.action, len(result.claims),
                    result.supported, result.contradicted, result.unsupported,
                    result.latency_ms,
                )

    async def _lookup(self, claims: List[Claim]):
        pairs = [(c.medication, c.target) for c in claims]
        rels = await self.graph.relations_between(pairs)
        version = await self.graph.version()
        return rels, version

    def _apply_policy(self, answer: str, result: KGCheckResult) -> str:
        if self.mode == "shadow":
            result.action = "logged"
            return answer

        contradicted = result.contradicted
        unsupported = result.unsupported
        if not contradicted and not unsupported:
            result.action = "none"
            return answer

        if self.mode == "flag":
            result.action = "flagged"
            note = CAUTION_FLAGGED_CONTRADICTED if contradicted else CAUTION_UNSUPPORTED
            return f"{answer.rstrip()}\n\n{note}"

        # block
        drop_verdicts = {"contradicted"} | ({"unsupported"} if self.block_unsupported else set())
        drop = {c.sentence_index for c in result.claims if c.verdict in drop_verdicts}
        if not drop:
            result.action = "flagged"
            return f"{answer.rstrip()}\n\n{CAUTION_UNSUPPORTED}"

        # Remove the offending sentences in place so paragraphs and lists keep
        # their shape.
        sentences = self.extractor.sentences(answer)
        remaining = answer
        for i in sorted(drop):
            remaining = remaining.replace(sentences[i], "", 1)
        remaining = re.sub(r"[ \t]{2,}", " ", remaining)
        remaining = re.sub(r"\n[ \t]+", "\n", remaining)
        remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
        result.removed_sentences = len(drop)
        result.action = "blocked"
        if len(remaining) < MIN_REMAINING_CHARS:
            return SAFE_FALLBACK
        note = CAUTION_CONTRADICTED if contradicted else CAUTION_UNSUPPORTED
        return f"{remaining}\n\n{note}"
