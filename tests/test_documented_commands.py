"""
The documented commands start the study server, and the tests they name can fail.

Round-3 review, finding A.4 and spec finding A.1: README, run.sh and AGENTS.md
started `python main.py`, which ran the Kotaemon `rag_server.py`. That server has
no Dose Boundary, no study release and no shared LLM client, so following the
README deployed an unguarded server. The 24 root `test_*.py` scripts were never
collected (pytest.ini sets `testpaths = tests`) and asserted nothing, and seven
root rebuild scripts each re-created the index, one with an English-only model.

What these pin down, by reading the repository a developer reads:

  * every start command in the README, the run scripts and AGENTS.md launches
    `simple_rag_server.py`, the study server
  * no module imports the Kotaemon pipeline or puts its tree on the path
  * no test file lives outside tests/, and every collected test asserts something
  * only the reindex service and the pipeline itself create, delete or embed a
    Chroma collection, so no script can rebuild the index with another model
"""

import ast
import re
import unittest
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Trees that are not first-party Python, or are generated.
SKIPPED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "cache", "out",
    "android-app", "docs", "site", ".claude",
}

STUDY_SERVER = "simple_rag_server.py"

# The documents and scripts a developer follows to start the system.
START_DOCUMENTS = ["README.md", "AGENTS.md", "how_to_run.md", "SETUP_INSTRUCTIONS.md"]
RUN_SCRIPTS = ["run.sh", "run_simple.sh", "start.sh"]

# A command line that starts a Python server from the repository root.
SERVER_LAUNCH = re.compile(
    r"\bpython3?\s+(?P<script>[\w./-]*?(?:main|rag_server|simple_rag_server|app)\.py)\b"
    r"|\buvicorn\s+(?P<module>[\w.]+):app\b"
)

KOTAEMON_MODULES = ("ktem", "kotaemon", "rag_server")


def first_party_python() -> Iterator[Path]:
    """Every first-party .py file under the repository root."""
    for path in ROOT.rglob("*.py"):
        parts = set(path.relative_to(ROOT).parts[:-1])
        if parts & SKIPPED_DIRS:
            continue
        yield path


def server_launches(text: str) -> List[str]:
    """The server scripts or modules each launch line in `text` starts."""
    found = []
    for match in SERVER_LAUNCH.finditer(text):
        found.append(match.group("script") or f"{match.group('module')}:app")
    return found


class StartCommandsLaunchTheStudyServer(unittest.TestCase):
    def test_every_documented_start_command_launches_the_study_server(self):
        for name in START_DOCUMENTS + RUN_SCRIPTS:
            path = ROOT / name
            if not path.exists():
                continue
            for launched in server_launches(path.read_text(encoding="utf-8")):
                with self.subTest(document=name, launches=launched):
                    self.assertEqual(
                        Path(launched).name, STUDY_SERVER,
                        f"{name} starts {launched}, not the study server",
                    )

    def test_the_run_scripts_end_by_starting_the_study_server(self):
        for name in ("run.sh", "run_simple.sh"):
            launches = server_launches((ROOT / name).read_text(encoding="utf-8"))
            with self.subTest(script=name):
                self.assertTrue(launches, f"{name} starts no server at all")
                self.assertEqual(Path(launches[-1]).name, STUDY_SERVER)

    def test_start_sh_hands_over_to_a_run_script_and_offers_no_other_server(self):
        text = (ROOT / "start.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"exec \./run(_simple)?\.sh")
        self.assertNotIn("--full)", text, "start.sh still offers the Kotaemon server")

    def test_agents_md_names_the_study_server_and_pytest(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn(f"python {STUDY_SERVER}", text)
        self.assertIn("pytest", text)

    def test_the_documents_name_no_root_test_script(self):
        root_script = re.compile(r"\bpython3?\s+(test_\w+\.py)")
        for name in START_DOCUMENTS:
            path = ROOT / name
            if not path.exists():
                continue
            with self.subTest(document=name):
                self.assertEqual(root_script.findall(path.read_text(encoding="utf-8")), [])


class NothingImportsTheKotaemonPipeline(unittest.TestCase):
    def imports_of(self, path: Path) -> List[Tuple[int, str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [(node.lineno, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append((node.lineno, node.module))
        return found

    def test_no_module_imports_kotaemon_or_its_server(self):
        offenders = []
        for path in first_party_python():
            for lineno, module in self.imports_of(path):
                if module.split(".")[0] in KOTAEMON_MODULES:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
        self.assertEqual(offenders, [])

    def test_no_module_puts_the_kotaemon_tree_on_the_path(self):
        # `Path(...) / "kotaemon-main" / "libs"` on sys.path is how main.py and
        # rag_server.py reached ktem without installing it.
        on_path = re.compile(r"""["']kotaemon-main["']\s*/|sys\.path[^\n]*kotaemon""")
        offenders = [
            str(path.relative_to(ROOT)) for path in first_party_python()
            if path != Path(__file__).resolve()
            and on_path.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_requirements_do_not_name_kotaemon(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        names = {re.split(r"[<>=\[ ]", line.strip())[0].lower() for line in lines
                 if line.strip() and not line.startswith("#")}
        self.assertFalse(names & {"ktem", "kotaemon"})


def is_check(node: ast.AST) -> bool:
    """An assert statement, or a call that fails a test when it does not hold."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name.startswith("assert") or name in {"raises", "fail", "warns"}
    return False


def collected_tests(tree: ast.Module) -> Iterator[ast.AST]:
    """Module-level test functions and test methods of module-level classes."""
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, functions) and node.name.startswith("test"):
            yield node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, functions) and item.name.startswith("test"):
                    yield item


class EveryTestIsCollectedAndAsserts(unittest.TestCase):
    def test_pytest_collects_only_the_tests_directory(self):
        ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
        self.assertRegex(ini, r"(?m)^testpaths\s*=\s*tests\s*$")

    def test_no_test_file_lives_outside_tests(self):
        outside = [
            str(path.relative_to(ROOT)) for path in first_party_python()
            if (path.name.startswith("test_") or path.name.endswith("_test.py"))
            and TESTS not in path.parents
        ]
        self.assertEqual(outside, [], "a test file outside tests/ is never collected")

    def test_every_collected_test_asserts_something(self):
        silent = []
        for path in sorted(TESTS.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for test in collected_tests(tree):
                if not any(is_check(node) for node in ast.walk(test)):
                    silent.append(f"{path.name}:{test.lineno} {test.name}")
        self.assertEqual(silent, [], "a test that asserts nothing cannot fail")


# The modules allowed to create, delete or embed a Chroma collection: the
# pipeline that serves the index, the shared embedder, and the reindex service,
# which refuses any model but the study release's.
INDEX_OWNERS = {"simple_rag_server.py", "reindex_service.py", "retrieval/embedder.py"}
INDEX_WRITES = re.compile(
    r"\b(create_collection|delete_collection|SentenceTransformerEmbeddingFunction)\s*\("
    r"|\b_rebuild_from_documents\b|\b_perform_auto_rebuild\b"
)


class OnlyTheReindexServiceRebuildsTheIndex(unittest.TestCase):
    def test_no_script_creates_or_embeds_a_collection_itself(self):
        offenders = []
        for path in first_party_python():
            rel = path.relative_to(ROOT).as_posix()
            if rel in INDEX_OWNERS or rel.startswith("tests/") or rel.startswith("kotaemon-main/"):
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if INDEX_WRITES.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_the_reindex_script_goes_through_the_reindex_service(self):
        text = (ROOT / "scripts" / "reindex_with_configured_model.py").read_text(encoding="utf-8")
        self.assertIn("from reindex_service import", text)


if __name__ == "__main__":
    unittest.main()
