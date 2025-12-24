"""
Phase 2 Tests: Configuration Module

Run with: python3 tests/test_graphrag_config.py
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_creation():
    """Test basic configuration creation."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig(root_dir="./data/graphrag")

    assert config.root_dir == Path("./data/graphrag")
    assert config._initialized is False
    print("PASS: Configuration creation")
    return True


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    from graphrag_integration.config import GraphRAGConfig

    yaml_path = Path("./data/graphrag/settings.yaml")
    if not yaml_path.exists():
        print(f"SKIP: YAML file not found: {yaml_path}")
        return True

    config = GraphRAGConfig.from_yaml(str(yaml_path))

    assert config.root_dir == Path("./data/graphrag")
    assert config._initialized is True
    print("PASS: Configuration from YAML")
    return True


def test_config_env_substitution():
    """Test environment variable substitution."""
    from graphrag_integration.config import GraphRAGConfig

    # Set test environment variable
    os.environ["TEST_GRAPHRAG_VAR"] = "test_value_123"
    content = "api_key: ${TEST_GRAPHRAG_VAR}"
    result = GraphRAGConfig._substitute_env_vars(content)

    assert result == "api_key: test_value_123", f"Expected 'api_key: test_value_123', got '{result}'"

    # Clean up
    del os.environ["TEST_GRAPHRAG_VAR"]

    print("PASS: Environment variable substitution")
    return True


def test_config_model_access():
    """Test accessing model configurations."""
    from graphrag_integration.config import GraphRAGConfig

    yaml_path = Path("./data/graphrag/settings.yaml")
    if not yaml_path.exists():
        print(f"SKIP: YAML file not found")
        return True

    config = GraphRAGConfig.from_yaml(str(yaml_path))

    chat_model = config.get_chat_model()
    embedding_model = config.get_embedding_model()

    assert chat_model.type == "chat", f"Expected 'chat', got '{chat_model.type}'"
    assert embedding_model.type == "embedding", f"Expected 'embedding', got '{embedding_model.type}'"
    assert chat_model.model == "gpt-4o-mini"
    assert embedding_model.model == "text-embedding-3-small"

    print("PASS: Model access")
    return True


def test_config_paths():
    """Test path properties."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig(root_dir="./data/graphrag")

    assert config.input_dir == Path("./data/graphrag/input")
    assert config.output_dir == Path("./data/graphrag/output")
    assert config.cache_dir == Path("./data/graphrag/cache")
    assert config.artifacts_dir == Path("./data/graphrag/output/artifacts")
    assert config.prompts_dir == Path("./data/graphrag/prompts")

    print("PASS: Path properties")
    return True


def test_config_validation():
    """Test configuration validation."""
    from graphrag_integration.config import GraphRAGConfig

    # Test with non-existent directory
    config = GraphRAGConfig(root_dir="./nonexistent_directory")
    errors = config.validate()

    assert len(errors) > 0, "Expected validation errors for non-existent directory"
    assert any("does not exist" in e for e in errors)

    print("PASS: Configuration validation")
    return True


def test_config_validation_with_api_key():
    """Test validation with API key set."""
    from graphrag_integration.config import GraphRAGConfig

    # Set API key
    os.environ["GRAPHRAG_API_KEY"] = "test_key"

    yaml_path = Path("./data/graphrag/settings.yaml")
    if yaml_path.exists():
        config = GraphRAGConfig.from_yaml(str(yaml_path))
        errors = config.validate()

        # Should only have directory errors, not API key errors
        api_key_errors = [e for e in errors if "API key" in e]
        assert len(api_key_errors) == 0, f"Unexpected API key errors: {api_key_errors}"

    # Clean up
    del os.environ["GRAPHRAG_API_KEY"]

    print("PASS: Validation with API key")
    return True


def test_config_dataclasses():
    """Test configuration dataclasses."""
    from graphrag_integration.config import (
        ModelConfig,
        ChunkingConfig,
        EntityExtractionConfig,
        SearchConfig,
    )

    # Test ModelConfig
    model = ModelConfig(
        type="chat",
        model_provider="openai",
        model="gpt-4o-mini",
        api_key="test",
    )
    assert model.type == "chat"
    assert model.max_tokens == 4096  # Default

    # Test ChunkingConfig
    chunking = ChunkingConfig()
    assert chunking.size == 1200
    assert chunking.overlap == 100

    # Test EntityExtractionConfig
    extraction = EntityExtractionConfig()
    assert "Symptom" in extraction.entity_types
    assert "Medication" in extraction.entity_types

    # Test SearchConfig
    search = SearchConfig()
    assert search.max_tokens == 12000
    assert search.top_k_entities == 10

    print("PASS: Dataclasses")
    return True


def test_config_to_dict():
    """Test configuration serialization."""
    from graphrag_integration.config import GraphRAGConfig

    yaml_path = Path("./data/graphrag/settings.yaml")
    if not yaml_path.exists():
        print(f"SKIP: YAML file not found")
        return True

    config = GraphRAGConfig.from_yaml(str(yaml_path))
    config_dict = config.to_dict()

    assert "root_dir" in config_dict
    assert "models" in config_dict
    assert "chunking" in config_dict
    assert "extraction" in config_dict
    assert "search" in config_dict
    assert "paths" in config_dict
    assert "initialized" in config_dict

    print("PASS: Configuration to dict")
    return True


def test_config_entity_types():
    """Test entity types configuration."""
    from graphrag_integration.config import GraphRAGConfig

    yaml_path = Path("./data/graphrag/settings.yaml")
    if not yaml_path.exists():
        print(f"SKIP: YAML file not found")
        return True

    config = GraphRAGConfig.from_yaml(str(yaml_path))

    expected_types = [
        "Symptom", "Medication", "Condition", "Treatment",
        "SideEffect", "Dosage", "Route", "CareGoal"
    ]

    for entity_type in expected_types:
        assert entity_type in config.extraction.entity_types, \
            f"Missing entity type: {entity_type}"

    print("PASS: Entity types configuration")
    return True


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2 Tests: Configuration Module")
    print("=" * 60)
    print()

    tests = [
        test_config_creation,
        test_config_from_yaml,
        test_config_env_substitution,
        test_config_model_access,
        test_config_paths,
        test_config_validation,
        test_config_validation_with_api_key,
        test_config_dataclasses,
        test_config_to_dict,
        test_config_entity_types,
    ]

    results = []
    for test in tests:
        print(f"Running: {test.__name__}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"FAIL: {e}")
            results.append(False)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("Phase 2 COMPLETE - Ready for Phase 3")
        return 0
    else:
        print("Phase 2 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
