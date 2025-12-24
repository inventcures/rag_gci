"""
GraphRAG Configuration Module for Palli Sahayak

Provides configuration management, validation, and environment variable
substitution for GraphRAG integration.

Usage:
    from graphrag_integration.config import GraphRAGConfig

    # Load from YAML
    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")

    # Access models
    chat_model = config.get_chat_model()
    embedding_model = config.get_embedding_model()

    # Access paths
    input_dir = config.input_dir
    output_dir = config.output_dir

    # Validate
    errors = config.validate()
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for a language model.

    Attributes:
        type: Model type ("chat" or "embedding")
        model_provider: Provider name ("openai", "azure_openai", "litellm")
        model: Model identifier
        api_key: API key for authentication
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        request_timeout: Request timeout in seconds
        retry_strategy: Retry strategy name
        max_retries: Maximum retry attempts
        batch_size: Batch size for embeddings
    """
    type: str  # "chat" or "embedding"
    model_provider: str  # "openai", "azure_openai", "litellm"
    model: str
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 180
    retry_strategy: str = "exponential_backoff"
    max_retries: int = 3
    batch_size: int = 16  # For embeddings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "model_provider": self.model_provider,
            "model": self.model,
            "api_key": "***" if self.api_key else "",  # Mask API key
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "retry_strategy": self.retry_strategy,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
        }


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking.

    Attributes:
        size: Chunk size in tokens
        overlap: Overlap between chunks in tokens
        encoding_model: Tokenizer model name
    """
    size: int = 1200
    overlap: int = 100
    encoding_model: str = "cl100k_base"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "size": self.size,
            "overlap": self.overlap,
            "encoding_model": self.encoding_model,
        }


@dataclass
class EntityExtractionConfig:
    """
    Configuration for entity extraction.

    Attributes:
        prompt: Path to prompt template
        entity_types: List of entity types to extract
        max_gleanings: Number of extraction passes
    """
    prompt: str = "./prompts/entity_extraction.txt"
    entity_types: List[str] = field(default_factory=lambda: [
        "Symptom", "Medication", "Condition", "Treatment",
        "SideEffect", "Dosage", "Route", "CareGoal"
    ])
    max_gleanings: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "entity_types": self.entity_types,
            "max_gleanings": self.max_gleanings,
        }


@dataclass
class SearchConfig:
    """
    Configuration for search operations.

    Attributes:
        max_tokens: Maximum tokens for response
        data_max_tokens: Maximum tokens for context data
        concurrency: Number of concurrent requests
        text_unit_prop: Proportion of text units in context
        community_prop: Proportion of community reports in context
        top_k_entities: Number of top entities to retrieve
        top_k_relationships: Number of top relationships to retrieve
    """
    max_tokens: int = 12000
    data_max_tokens: int = 12000
    concurrency: int = 32
    text_unit_prop: float = 0.5
    community_prop: float = 0.1
    top_k_entities: int = 10
    top_k_relationships: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "data_max_tokens": self.data_max_tokens,
            "concurrency": self.concurrency,
            "text_unit_prop": self.text_unit_prop,
            "community_prop": self.community_prop,
            "top_k_entities": self.top_k_entities,
            "top_k_relationships": self.top_k_relationships,
        }


class GraphRAGConfig:
    """
    Configuration manager for GraphRAG integration.

    Handles loading, validation, and environment variable substitution
    for GraphRAG settings.

    Attributes:
        root_dir: Root directory for GraphRAG data
        models: Dictionary of model configurations
        chunking: Chunking configuration
        extraction: Entity extraction configuration
        search: Search configuration

    Example:
        # Load from YAML file
        config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")

        # Access model configurations
        chat_model = config.get_chat_model()
        embedding_model = config.get_embedding_model()

        # Access paths
        print(config.input_dir)   # ./data/graphrag/input
        print(config.output_dir)  # ./data/graphrag/output

        # Validate configuration
        errors = config.validate()
        if errors:
            print("Configuration errors:", errors)

        # Convert to native GraphRAG config
        native_config = config.to_graphrag_config()
    """

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(
        self,
        root_dir: str = "./data/graphrag",
        settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GraphRAG configuration.

        Args:
            root_dir: Root directory for GraphRAG data
            settings: Optional settings dictionary (if not loading from YAML)
        """
        self.root_dir = Path(root_dir)
        self._settings = settings or {}
        self._models: Dict[str, ModelConfig] = {}
        self._initialized = False

        # Initialize default configurations
        self.chunking = ChunkingConfig()
        self.extraction = EntityExtractionConfig()
        self.search = SearchConfig()

        if settings:
            self._parse_settings(settings)
            self._initialized = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GraphRAGConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to settings.yaml file

        Returns:
            GraphRAGConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Substitute environment variables
        content = cls._substitute_env_vars(raw_content)

        try:
            settings = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Determine root_dir from yaml_path
        root_dir = yaml_path.parent

        return cls(root_dir=str(root_dir), settings=settings)

    @classmethod
    def _substitute_env_vars(cls, content: str) -> str:
        """
        Substitute ${VAR_NAME} patterns with environment variable values.

        Args:
            content: YAML content string

        Returns:
            Content with substituted values
        """
        def replace_match(match):
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                logger.warning(f"Environment variable not set: {var_name}")
            return value

        return cls.ENV_VAR_PATTERN.sub(replace_match, content)

    def _parse_settings(self, settings: Dict[str, Any]) -> None:
        """
        Parse settings dictionary into configuration objects.

        Args:
            settings: Raw settings dictionary from YAML
        """
        # Parse models
        models_config = settings.get("models", {})
        for model_id, model_settings in models_config.items():
            self._models[model_id] = ModelConfig(
                type=model_settings.get("type", "chat"),
                model_provider=model_settings.get("model_provider", "openai"),
                model=model_settings.get("model", "gpt-4o-mini"),
                api_key=model_settings.get("api_key", ""),
                max_tokens=model_settings.get("max_tokens", 4096),
                temperature=model_settings.get("temperature", 0.0),
                request_timeout=model_settings.get("request_timeout", 180),
                retry_strategy=model_settings.get("retry_strategy", "exponential_backoff"),
                max_retries=model_settings.get("max_retries", 3),
                batch_size=model_settings.get("batch_size", 16),
            )

        # Parse chunking
        chunks_config = settings.get("chunks", {})
        self.chunking = ChunkingConfig(
            size=chunks_config.get("size", 1200),
            overlap=chunks_config.get("overlap", 100),
            encoding_model=chunks_config.get("encoding_model", "cl100k_base"),
        )

        # Parse entity extraction
        extract_config = settings.get("extract_graph", {})
        self.extraction = EntityExtractionConfig(
            prompt=extract_config.get("prompt", "./prompts/entity_extraction.txt"),
            entity_types=extract_config.get("entity_types", [
                "Symptom", "Medication", "Condition", "Treatment",
                "SideEffect", "Dosage", "Route", "CareGoal"
            ]),
            max_gleanings=extract_config.get("max_gleanings", 1),
        )

        # Parse search configuration
        global_search = settings.get("global_search", {})
        local_search = settings.get("local_search", {})
        self.search = SearchConfig(
            max_tokens=global_search.get("max_tokens", 12000),
            data_max_tokens=global_search.get("data_max_tokens", 12000),
            concurrency=global_search.get("concurrency", 32),
            text_unit_prop=local_search.get("text_unit_prop", 0.5),
            community_prop=local_search.get("community_prop", 0.1),
            top_k_entities=local_search.get("top_k_entities", 10),
            top_k_relationships=local_search.get("top_k_relationships", 10),
        )

    def get_chat_model(self) -> ModelConfig:
        """
        Get the default chat model configuration.

        Returns:
            ModelConfig for the default chat model
        """
        if "default_chat_model" in self._models:
            return self._models["default_chat_model"]

        # Return default configuration
        return ModelConfig(
            type="chat",
            model_provider="openai",
            model="gpt-4o-mini",
            api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
        )

    def get_embedding_model(self) -> ModelConfig:
        """
        Get the default embedding model configuration.

        Returns:
            ModelConfig for the default embedding model
        """
        if "default_embedding_model" in self._models:
            return self._models["default_embedding_model"]

        # Return default configuration
        return ModelConfig(
            type="embedding",
            model_provider="openai",
            model="text-embedding-3-small",
            api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
        )

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig or None if not found
        """
        return self._models.get(model_id)

    @property
    def input_dir(self) -> Path:
        """Get input directory path."""
        return self.root_dir / "input"

    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return self.root_dir / "output"

    @property
    def artifacts_dir(self) -> Path:
        """Get artifacts directory path."""
        return self.output_dir / "artifacts"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.root_dir / "cache"

    @property
    def prompts_dir(self) -> Path:
        """Get prompts directory path."""
        return self.root_dir / "prompts"

    def get_prompt_path(self, prompt_name: str) -> Path:
        """
        Get path to a prompt file.

        Args:
            prompt_name: Prompt name (without .txt extension)

        Returns:
            Path to the prompt file
        """
        return self.prompts_dir / f"{prompt_name}.txt"

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required directories
        if not self.root_dir.exists():
            errors.append(f"Root directory does not exist: {self.root_dir}")

        # Check API key for chat model
        chat_model = self.get_chat_model()
        if not chat_model.api_key:
            errors.append("Chat model API key not configured (set GRAPHRAG_API_KEY)")

        # Check API key for embedding model
        embedding_model = self.get_embedding_model()
        if not embedding_model.api_key:
            errors.append("Embedding model API key not configured (set GRAPHRAG_API_KEY)")

        # Check entity types
        if not self.extraction.entity_types:
            errors.append("No entity types configured for extraction")

        # Check chunk size
        if self.chunking.size <= 0:
            errors.append(f"Invalid chunk size: {self.chunking.size}")

        if self.chunking.overlap >= self.chunking.size:
            errors.append(f"Chunk overlap ({self.chunking.overlap}) must be less than size ({self.chunking.size})")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "root_dir": str(self.root_dir),
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "chunking": self.chunking.to_dict(),
            "extraction": self.extraction.to_dict(),
            "search": self.search.to_dict(),
            "paths": {
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "cache_dir": str(self.cache_dir),
                "prompts_dir": str(self.prompts_dir),
            },
            "initialized": self._initialized,
        }

    def to_graphrag_config(self) -> Any:
        """
        Convert to native GraphRAG configuration object.

        Returns:
            graphrag.config.GraphRagConfig instance or dict if GraphRAG not installed
        """
        try:
            from graphrag.config import GraphRagConfig
            from graphrag.config.load_config import load_config

            return load_config(str(self.root_dir))
        except ImportError:
            logger.warning("GraphRAG not installed, returning dict config")
            return self._settings
        except Exception as e:
            logger.warning(f"Failed to load native GraphRAG config: {e}")
            return self._settings

    def __repr__(self) -> str:
        return (
            f"GraphRAGConfig("
            f"root_dir={self.root_dir}, "
            f"models={list(self._models.keys())}, "
            f"entity_types={len(self.extraction.entity_types)}, "
            f"initialized={self._initialized})"
        )

    def __str__(self) -> str:
        return self.__repr__()
