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
    required_dirs = [
        "graphrag_integration",
        "graphrag_integration/prompts",
        "data/graphrag",
        "data/graphrag/output",
        "data/graphrag/output/artifacts",
        "data/graphrag/cache",
        "tests",
    ]

    base = Path(__file__).parent.parent
    errors = []

    for dir_path in required_dirs:
        full_path = base / dir_path
        if not full_path.exists():
            errors.append(f"Missing directory: {dir_path}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return False

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

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return False

    print("PASS: All init files exist")
    return True


def test_input_symlink():
    """Verify input directory symlinks to uploads."""
    base = Path(__file__).parent.parent
    input_path = base / "data" / "graphrag" / "input"
    uploads_path = base / "uploads"

    if not input_path.exists():
        print(f"FAIL: Input path does not exist: {input_path}")
        return False

    if input_path.is_symlink():
        target = input_path.resolve()
        print(f"PASS: Input is symlink -> {target}")
        return True
    elif input_path.is_dir():
        print(f"PASS: Input is directory (not symlink): {input_path}")
        return True
    else:
        print(f"FAIL: Input path is not directory or symlink")
        return False


def test_module_structure():
    """Verify module can be found."""
    base = Path(__file__).parent.parent

    # Check main module file
    main_init = base / "graphrag_integration" / "__init__.py"
    if not main_init.exists():
        print(f"FAIL: Main __init__.py not found")
        return False

    # Check content has expected exports
    content = main_init.read_text()
    expected = ["GraphRAGConfig", "GraphRAGIndexer", "GraphRAGQueryEngine", "GraphRAGDataLoader"]

    for export in expected:
        if export not in content:
            print(f"FAIL: Missing export in __init__.py: {export}")
            return False

    print("PASS: Module structure correct")
    return True


def test_prompts_module():
    """Verify prompts module is properly set up."""
    base = Path(__file__).parent.parent
    prompts_init = base / "graphrag_integration" / "prompts" / "__init__.py"

    if not prompts_init.exists():
        print(f"FAIL: Prompts __init__.py not found")
        return False

    content = prompts_init.read_text()
    if "get_prompt" not in content:
        print(f"FAIL: get_prompt function not in prompts module")
        return False

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
