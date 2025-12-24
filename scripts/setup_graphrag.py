#!/usr/bin/env python3
"""
GraphRAG Setup Script for Palli Sahayak

Creates necessary directory structure and initializes configuration.

Usage:
    python scripts/setup_graphrag.py

This script is idempotent - safe to run multiple times.
"""

import os
import shutil
from pathlib import Path


def setup_graphrag_directories(base_path: str = ".") -> dict:
    """
    Create GraphRAG directory structure.

    Args:
        base_path: Base path for the project

    Returns:
        Dictionary with created paths
    """
    base = Path(base_path)

    # Define directory structure
    directories = {
        "module": base / "graphrag_integration",
        "prompts": base / "graphrag_integration" / "prompts",
        "data_root": base / "data" / "graphrag",
        "input": base / "data" / "graphrag" / "input",
        "output": base / "data" / "graphrag" / "output",
        "artifacts": base / "data" / "graphrag" / "output" / "artifacts",
        "cache": base / "data" / "graphrag" / "cache",
        "tests": base / "tests",
    }

    # Create directories
    for name, path in directories.items():
        if name != "input":  # Skip input as it will be a symlink
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path}")

    return directories


def create_symlink(base_path: str = ".") -> None:
    """Create symlink from input to uploads if not exists."""
    base = Path(base_path)
    uploads_path = base / "uploads"
    input_path = base / "data" / "graphrag" / "input"

    if uploads_path.exists():
        if input_path.is_symlink():
            print(f"Symlink already exists: {input_path}")
        elif input_path.exists():
            # Remove directory and create symlink
            shutil.rmtree(input_path)
            input_path.symlink_to(Path("../../uploads"))
            print(f"Created symlink: {input_path} -> uploads/")
        else:
            input_path.symlink_to(Path("../../uploads"))
            print(f"Created symlink: {input_path} -> uploads/")
    else:
        # Create uploads directory first
        uploads_path.mkdir(parents=True, exist_ok=True)
        print(f"Created uploads directory: {uploads_path}")
        input_path.symlink_to(Path("../../uploads"))
        print(f"Created symlink: {input_path} -> uploads/")


def create_init_files(base_path: str = ".") -> None:
    """Create __init__.py files for Python packages."""
    base = Path(base_path)

    init_files = [
        base / "graphrag_integration" / "__init__.py",
        base / "graphrag_integration" / "prompts" / "__init__.py",
    ]

    for init_file in init_files:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text('"""GraphRAG Integration Module."""\n')
            print(f"Created: {init_file}")
        else:
            print(f"Exists: {init_file}")


def verify_setup(base_path: str = ".") -> bool:
    """
    Verify the setup is complete.

    Returns:
        True if all checks pass, False otherwise
    """
    base = Path(base_path)
    errors = []

    # Check directories
    required_dirs = [
        "graphrag_integration",
        "graphrag_integration/prompts",
        "data/graphrag",
        "data/graphrag/output",
        "data/graphrag/output/artifacts",
        "data/graphrag/cache",
        "tests",
    ]

    for dir_path in required_dirs:
        full_path = base / dir_path
        if not full_path.exists():
            errors.append(f"Missing directory: {dir_path}")

    # Check symlink
    input_path = base / "data" / "graphrag" / "input"
    if not input_path.exists():
        errors.append("Missing: data/graphrag/input symlink")

    # Check init files
    init_files = [
        "graphrag_integration/__init__.py",
        "graphrag_integration/prompts/__init__.py",
    ]

    for init_file in init_files:
        full_path = base / init_file
        if not full_path.exists():
            errors.append(f"Missing init file: {init_file}")

    if errors:
        print("\nSetup verification FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\nSetup verification PASSED!")
        return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("GraphRAG Setup for Palli Sahayak")
    print("=" * 60)

    # Get base path (script is in scripts/ directory)
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent

    print(f"\nBase path: {base_path}")
    print()

    # Setup
    print("Creating directories...")
    setup_graphrag_directories(str(base_path))
    print()

    print("Creating symlink...")
    create_symlink(str(base_path))
    print()

    print("Checking init files...")
    create_init_files(str(base_path))
    print()

    # Verify
    print("Verifying setup...")
    success = verify_setup(str(base_path))

    print()
    print("=" * 60)
    if success:
        print("GraphRAG setup complete!")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install graphrag>=2.7.0")
        print("2. Create data/graphrag/settings.yaml")
        print("3. Set GRAPHRAG_API_KEY in .env")
        print("4. Implement Phase 2: Configuration Module")
    else:
        print("Setup incomplete. Please fix errors above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
