"""
Phase 1 Tests: Foundation Setup

Run with: python tests/test_graphrag_setup.py
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_directory_structure_exists():
    """Verify all required directories exist."""
    # Only the directories the repository ships. output/, output/artifacts/ and
    # cache/ are made by the indexer at run time and are not in a fresh checkout;
    # requiring them made this test pass only on a machine that had indexed.
    required_dirs = [
        "graphrag_integration",
        "graphrag_integration/prompts",
        "data/graphrag",
        "tests",
    ]

    base = Path(__file__).parent.parent
    errors = []

    for dir_path in required_dirs:
        full_path = base / dir_path
        if not full_path.exists():
            errors.append(f"Missing directory: {dir_path}")

    assert errors == [], errors
    print("PASS: All directories exist")
    return True


def test_init_files_exist():
    """Verify __init__.py files exist."""
    init_files = [
        "graphrag_integration/__init__.py",
        "graphrag_integration/prompts/__init__.py",
    ]

    base = Path(__file__).parent.parent
    errors = []

    for init_file in init_files:
        full_path = base / init_file
        if not full_path.exists():
            errors.append(f"Missing init file: {init_file}")

    assert errors == [], errors
    print("PASS: All init files exist")
    return True


def test_input_symlink():
    """Verify input directory symlinks to uploads."""
    base = Path(__file__).parent.parent
    input_path = base / "data" / "graphrag" / "input"
    uploads_path = base / "uploads"

    # The link is what the repository ships; its target (uploads/) is created by
    # the server on first upload, so the link is checked, not what it points at.
    if input_path.is_symlink():
        target = Path(os.readlink(input_path))
        assert target.name == uploads_path.name, f"input links to {target}, not uploads/"
    else:
        assert input_path.is_dir(), f"Input path is not a directory or symlink: {input_path}"
    print(f"PASS: Input is {'symlink' if input_path.is_symlink() else 'directory'}")
    return True


def test_module_structure():
    """Verify module can be found."""
    base = Path(__file__).parent.parent

    # Check main module file
    main_init = base / "graphrag_integration" / "__init__.py"
    assert main_init.exists(), "Main __init__.py not found"

    # Check content has expected exports
    content = main_init.read_text()
    expected = ["GraphRAGConfig", "GraphRAGIndexer", "GraphRAGQueryEngine", "GraphRAGDataLoader"]

    missing = [export for export in expected if export not in content]
    assert missing == [], f"Missing exports in __init__.py: {missing}"

    print("PASS: Module structure correct")
    return True


def test_prompts_module():
    """Verify prompts module is properly set up."""
    base = Path(__file__).parent.parent
    prompts_init = base / "graphrag_integration" / "prompts" / "__init__.py"

    assert prompts_init.exists(), "Prompts __init__.py not found"

    content = prompts_init.read_text()
    assert "get_prompt" in content, "get_prompt function not in prompts module"

    print("PASS: Prompts module correct")
    return True


def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1 Tests: Foundation Setup")
    print("=" * 60)
    print()

    tests = [
        test_directory_structure_exists,
        test_init_files_exist,
        test_input_symlink,
        test_module_structure,
        test_prompts_module,
    ]

    results = []
    for test in tests:
        print(f"Running: {test.__name__}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(False)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("Phase 1 COMPLETE - Ready for Phase 2")
        return 0
    else:
        print("Phase 1 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
