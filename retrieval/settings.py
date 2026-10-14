"""
Retrieval settings for the answer path: which retrievers run, how they are fused,
and whether the knowledge-graph cross-reference check is on.

One settings object is active per process. `SimpleRAGPipeline.query()` reads it,
and every channel (web, WhatsApp, the Gemini Live `search_medical_knowledge` tool,
the Sarvam voice fallback, Retell) calls that same `query()`, so web, WhatsApp and
phone users all get the same retrieval.

Precedence, highest first: CLI flag, environment variable, `retrieval:` block in
config.yaml, built-in default. Every default reproduces the behaviour before this
module existed: vector retrieval only, no graph, no claim check.

    retrieval:
      rag_method: vector          # vector | pageindex | hybrid | kg | graphrag | fused
      rrf_k: 60
      per_retriever_top_k: 5
      timeouts_s: {kg: 2.0, graphrag: 2.0, pageindex: 8.0}
      kg_check:
        mode: off                 # off | shadow | flag | block
        timeout_s: 1.5
        block_unsupported: false

Environment: RAG_METHOD, RAG_RRF_K, KG_CHECK_MODE, KG_CHECK_BLOCK_UNSUPPORTED,
KG_VERSION (an explicit label for the loaded graph snapshot).
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Methods that existed before this module. Their code path in query() is untouched.
LEGACY_METHODS = ("vector", "pageindex", "hybrid")
# Methods served by RetrievalRouter: vector is always one of the fused lists, so a
# graph outage degrades to plain vector retrieval rather than to no answer.
ROUTED_METHODS = ("kg", "graphrag", "fused")
RAG_METHODS = LEGACY_METHODS + ROUTED_METHODS

KG_CHECK_MODES = ("off", "shadow", "flag", "block")


@dataclass
class RetrievalSettings:
    """Everything that decides what the answer path retrieves and checks."""

    rag_method: str = "vector"
    rrf_k: int = 60
    per_retriever_top_k: int = 5
    kg_timeout_s: float = 2.0
    graphrag_timeout_s: float = 2.0
    pageindex_timeout_s: float = 8.0

    kg_check_mode: str = "off"
    kg_check_timeout_s: float = 1.5
    kg_check_block_unsupported: bool = False

    # Explicit graph snapshot label. When unset the checker derives one from the
    # live graph's node and relationship counts.
    kg_version_label: Optional[str] = None

    source: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rag_method not in RAG_METHODS:
            raise ValueError(
                f"Unknown rag_method {self.rag_method!r}; expected one of {RAG_METHODS}"
            )
        if self.kg_check_mode not in KG_CHECK_MODES:
            raise ValueError(
                f"Unknown kg_check mode {self.kg_check_mode!r}; expected one of {KG_CHECK_MODES}"
            )
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

    @property
    def is_routed(self) -> bool:
        return self.rag_method in ROUTED_METHODS

    @property
    def kg_check_enabled(self) -> bool:
        return self.kg_check_mode != "off"

    def fusion_description(self) -> Dict[str, Any]:
        """What gets written to the release log for fusion."""
        if not self.is_routed:
            return {"method": "none" if self.rag_method != "hybrid" else "concatenate"}
        return {
            "method": "rrf",
            "k": self.rrf_k,
            "per_retriever_top_k": self.per_retriever_top_k,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_yaml_block(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        block = data.get("retrieval") or {}
        return block if isinstance(block, dict) else {}
    except Exception:
        logger.exception("Could not read retrieval settings from %s", config_path)
        return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_retrieval_settings(
    config_path: Path = Path("config.yaml"),
    cli_rag_method: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> RetrievalSettings:
    """Build settings from config.yaml, then environment, then the CLI flag."""
    env = dict(os.environ) if env is None else env
    block = _load_yaml_block(config_path)
    kg_block = block.get("kg_check") or {}
    timeouts = block.get("timeouts_s") or {}
    source: Dict[str, str] = {}

    def pick(name: str, cli: Any, env_key: Optional[str], yaml_value: Any, default: Any) -> Any:
        if cli is not None:
            source[name] = "cli"
            return cli
        if env_key and env.get(env_key):
            source[name] = "env"
            return env[env_key]
        if yaml_value is not None:
            source[name] = "config"
            return yaml_value
        source[name] = "default"
        return default

    rag_method = str(pick("rag_method", cli_rag_method, "RAG_METHOD", block.get("rag_method"), "vector")).lower()
    rrf_k = int(pick("rrf_k", None, "RAG_RRF_K", block.get("rrf_k"), 60))
    mode = str(pick("kg_check_mode", None, "KG_CHECK_MODE", kg_block.get("mode"), "off")).lower()
    # YAML 1.1 reads a bare `off` as False.
    if mode in ("false", "none", ""):
        mode = "off"
    block_unsupported = _as_bool(
        pick(
            "kg_check_block_unsupported",
            None,
            "KG_CHECK_BLOCK_UNSUPPORTED",
            kg_block.get("block_unsupported"),
            False,
        )
    )

    return RetrievalSettings(
        rag_method=rag_method,
        rrf_k=rrf_k,
        per_retriever_top_k=int(block.get("per_retriever_top_k", 5)),
        kg_timeout_s=float(timeouts.get("kg", 2.0)),
        graphrag_timeout_s=float(timeouts.get("graphrag", 2.0)),
        pageindex_timeout_s=float(timeouts.get("pageindex", 8.0)),
        kg_check_mode=mode,
        kg_check_timeout_s=float(kg_block.get("timeout_s", 1.5)),
        kg_check_block_unsupported=block_unsupported,
        kg_version_label=env.get("KG_VERSION") or block.get("kg_version"),
        source=source,
    )


_active: Optional[RetrievalSettings] = None


def get_active_settings() -> RetrievalSettings:
    """The settings the running process answers with. Defaults until configured."""
    global _active
    if _active is None:
        _active = RetrievalSettings()
    return _active


def set_active_settings(settings: RetrievalSettings) -> RetrievalSettings:
    global _active
    _active = settings
    return settings
