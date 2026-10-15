"""The Android screens read in the participant's Supported Language (ticket 24).

Drives `scripts/check_no_hardcoded_strings.py`, the lint CI runs on the Android
sources: no user-facing literal in a screen, and every translation complete, in its
own script, under the table's code. The real tree must pass; the fixtures show each
way it can fail.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "check_no_hardcoded_strings", ROOT / "scripts" / "check_no_hardcoded_strings.py"
)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def _literals(kotlin: str) -> list:
    return [literal for _, literal in lint.find_literals(textwrap.dedent(kotlin))]


class TestNoLiteralInAScreen:
    def test_the_app_sources_have_no_user_facing_literal(self):
        findings = [
            f"{path.relative_to(ROOT)}:{line} {literal!r}"
            for path in lint.kotlin_sources()
            for line, literal in lint.find_literals(path.read_text(encoding="utf-8"))
        ]
        assert findings == []

    @pytest.mark.parametrize(
        "kotlin, literal",
        [
            # The shapes the earlier lint missed, each of which reached a Hindi screen.
            ('val (c, label) = when { deferral -> red to "Not answered." }', "Not answered."),
            ('enum class Destination(val label: String) { ASK("Ask") }', "Ask"),
            ('semantics { stateDescription = if (a) "selected" else "not selected" }', "selected"),
            ('CareRole.PATIENT -> "Patient"', "Patient"),
            ('it.copy(errorMessage = "Something went wrong.")', "Something went wrong."),
            ('Text("Earlier")', "Earlier"),
            ('val s = """Raw text"""', "Raw text"),
        ],
    )
    def test_a_literal_anywhere_in_a_screen_is_flagged(self, kotlin, literal):
        assert literal in _literals(kotlin)

    def test_icon_names_comments_and_marked_lines_are_not_text(self):
        kotlin = """
            val ask = glyph("Ask") { }
            val badge = builder("Patient") { }
            ImageVector.Builder(
                name = "Offline",
            )
            // Text("A comment is not shown")
            /* contentDescription = "Nor is a block comment" */
            val key = "ask:${id}" // not user-facing: a ViewModel store key
            val url = "${base}${path}"
            val n = "0%"
        """
        assert _literals(kotlin) == []

    def test_a_double_slash_inside_a_literal_does_not_hide_the_rest_of_the_line(self):
        assert set(_literals('val a = "http://x"; Text("Shown")')) == {"http://x", "Shown"}


def _res(tmp_path: Path, files: dict) -> Path:
    res = tmp_path / "res"
    for directory, entries in files.items():
        (res / directory).mkdir(parents=True)
        body = "".join(f'    <string name="{k}">{v}</string>\n' for k, v in entries.items())
        (res / directory / "strings.xml").write_text(
            f"<resources>\n{body}</resources>\n", encoding="utf-8"
        )
    return res


BASE = {"ask_prompt": "Ask a question", "evidence_count": "From %1$d passages"}
HINDI = {"ask_prompt": "सवाल पूछें", "evidence_count": "पुस्तिका के %1$d अंशों से"}


class TestEveryTranslationIsComplete:
    def test_the_app_resources_are_complete_in_hindi(self):
        assert lint.check_translations() == []
        assert (lint.RES / "values-hi" / "strings.xml").exists()

    def test_every_hindi_string_is_devanagari(self):
        hindi = lint.read_strings(lint.RES / "values-hi" / "strings.xml")
        english = lint.read_strings(lint.RES / "values" / "strings.xml")
        assert set(hindi) == set(english)
        for name, text in hindi.items():
            assert lint.scripts_in(lint.PLACEHOLDER.sub("", text)) == {"Deva"}, name

    def test_a_complete_translation_passes(self, tmp_path):
        assert lint.check_translations(_res(tmp_path, {"values": BASE, "values-hi": HINDI})) == []

    def test_hindi_is_required(self, tmp_path):
        problems = lint.check_translations(_res(tmp_path, {"values": BASE}))
        assert problems == ["values-hi/strings.xml is missing"]

    def test_a_missing_key_fails(self, tmp_path):
        res = _res(tmp_path, {"values": BASE, "values-hi": {"ask_prompt": "सवाल पूछें"}})
        assert lint.check_translations(res) == ["values-hi: 'evidence_count' is not translated"]

    def test_english_left_in_a_translation_fails(self, tmp_path):
        res = _res(tmp_path, {"values": BASE, "values-hi": {**HINDI, "ask_prompt": "Ask a question"}})
        [problem] = lint.check_translations(res)
        assert "'ask_prompt' is not all in Deva" in problem

    def test_a_lost_placeholder_fails(self, tmp_path):
        res = _res(tmp_path, {"values": BASE, "values-hi": {**HINDI, "evidence_count": "अंशों से"}})
        assert lint.check_translations(res) == ["values-hi: 'evidence_count' has different placeholders"]

    def test_odia_lives_under_the_table_code_not_od(self, tmp_path):
        odia = {"ask_prompt": "ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ", "evidence_count": "%1$d ଅଂଶରୁ"}
        assert lint.check_translations(
            _res(tmp_path, {"values": BASE, "values-hi": HINDI, "values-or": odia})
        ) == []
        [problem] = lint.check_translations(
            _res(tmp_path / "od", {"values": BASE, "values-hi": HINDI, "values-od": odia})
        )
        assert problem.startswith("values-od: not a Supported Language code")

    def test_another_language_in_the_wrong_script_fails(self, tmp_path):
        res = _res(tmp_path, {"values": BASE, "values-hi": HINDI, "values-bn": HINDI})
        problems = lint.check_translations(res)
        assert problems and all("is not all in Beng" in p for p in problems)
