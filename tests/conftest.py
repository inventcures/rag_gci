"""
Pytest configuration for the study test suite.

Two problems this fixes, both of which made the suite lie about its own state.

Async tests were silently not running. The suite contains async tests for the
safety manager and the temporal API, and without an async plugin pytest collected
them, reported them as failures, and the reason was "async def functions are not
natively supported" rather than anything about the code. Two suites were failing
for want of a plugin, not a fix.

A module importing an optional dependency broke collection for the entire suite.
One test module imports google.genai, which is not installed. A collection error
aborts the whole run, so a single optional integration was preventing any suite
from reporting. It is now skipped at collection, which is what an absent optional
dependency should do.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent

# Modules whose imports require an optional dependency that is not installed.
# They are skipped rather than allowed to abort collection for everything else.
_OPTIONAL_DEPENDENCY_MODULES = {
    "test_new_live_models.py": "google.genai",
}


def _missing_optional_modules() -> list:
    """
    Modules to skip because their dependency is absent.

    collect_ignore is used rather than a pytest_ignore_collect hook because the
    hook signature has moved across pytest versions and silently did nothing on
    the version in use here, which left the collection error in place.
    """
    missing = []
    for name, module in _OPTIONAL_DEPENDENCY_MODULES.items():
        # The full dotted path has to be probed, not just the root package.
        # "google" resolves as a namespace package even when "google.genai" does
        # not exist, so checking the root made every dependency look present and
        # silently skipped the exclusion.
        if not _module_available(module):
            print(
                f"\nskipping {name}: optional dependency {module} is not installed",
                file=sys.stderr,
            )
            missing.append(name)
    return missing


def _module_available(dotted: str) -> bool:
    """True when the full dotted module path can actually be imported."""
    try:
        importlib.import_module(dotted)
    except Exception:
        return False
    return True


collect_ignore = _missing_optional_modules()


def pytest_collection_modifyitems(config, items):
    """Report what was collected, so an empty or truncated run is visible."""
    collected = len(items)
    if collected == 0:
        print(
            "\nNo tests were collected. A green run here means nothing was tested, "
            "so treat this as a failure rather than a pass.",
            file=sys.stderr,
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Fail the run if pytest-asyncio is missing.

    Without it, async tests fail with a message about the plugin rather than
    about the code, which is easy to misread as a real failure and easy to
    ignore entirely.
    """
    if importlib.util.find_spec("pytest_asyncio") is None:
        terminalreporter.write_line(
            "ERROR: pytest-asyncio is not installed; async tests cannot run. "
            "Install it, or this suite is not reporting what it appears to.",
            red=True,
        )
        terminalreporter._session.exitstatus = 1


@pytest.fixture
def study_release():
    """
    The study release record, so tests agree on which release they are exercising.

    Tests that assert provenance should not each construct their own; a test that
    invents a release id proves nothing about the real one.
    """
    from study_release import get_study_release

    return get_study_release()
