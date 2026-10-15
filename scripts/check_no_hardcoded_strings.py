#!/usr/bin/env python3
"""
Fail if a user-facing literal reaches a Compose screen, or a translation is incomplete.

Criterion 4 says no screen requires English to operate. Auditing for that by reading
found three hardcoded literals in one sitting: "Ask a question", "Stop speaking",
"Play the answer again", all sitting inside a Hindi app. A screen is fine and a
neighbouring screen is broken, which is exactly what an audit misses.

This makes it a build failure instead. It is a lint rather than a test because the
thing being checked is a source pattern, not a behaviour.

Two checks:

1. **No literal in a screen.** Every string literal in the app's Kotlin sources is
   treated as user-facing, wherever it sits. An earlier version looked only at
   `Text(...)`, `text =` and `contentDescription =`, and so passed while an answer
   card's "Not answered" label (built in a `when`), the nav labels (enum
   arguments), the role names (in the data module) and every `stateDescription`
   reached a Hindi screen in English (review 3, ticket 24). The only exemptions are
   named: an icon's internal name (`glyph("...")`, `builder("...")`, `name = "..."`
   on an ImageVector builder, never shown or announced), and a line marked
   `// not user-facing: <why>`, so each one says why in the code.

2. **Every translation is complete.** Hindi (`values-hi`) must exist, every
   `values-<code>` directory must be a Supported Language's code from the table
   (Odia is `or`, never `od`), and each translated file must hold every key of
   `values/strings.xml`, the same format placeholders, and only letters of that
   language's script. A translation with an English string left in is how a Hindi
   screen shows English while the check reports clean.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supported_languages import SUPPORTED_LANGUAGES, get_language, scripts_in  # noqa: E402

ANDROID = ROOT / "android-app"
COMPOSE = ANDROID / "app" / "src" / "main" / "java"
RES = ANDROID / "app" / "src" / "main" / "res"

# Screen state built outside the app module whose text reaches the screen. The
# ViewModel's error messages were English literals in core/data, out of reach of a
# scan of the app module.
SCREEN_STATE = [
    ANDROID / "core" / "data" / "src" / "main" / "kotlin" / "org" / "inventcures"
    / "pallisahayak" / "data" / "AskViewModel.kt",
]

# Translations that must exist. Hindi first (spec, Android "Localisation"); the
# other Supported Languages follow the same pattern and are checked the same way
# once their directory exists.
REQUIRED_TRANSLATIONS = ("hi",)

# A literal with no letter in it ("", "0%", "1") cannot be English.
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# A double-quoted Kotlin string on one line. Raw strings are matched separately.
LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
RAW_LITERAL = re.compile(r'"""(.*?)"""', re.S)

# Literals that name an icon for debugging. They are never drawn or announced.
ICON_NAME = re.compile(r'(?:\bglyph\(|\bbuilder\(|\bname\s*=)\s*$')

MARKER = "// not user-facing:"

PLACEHOLDER = re.compile(r"%(?:\d+\$)?[-#+ 0,(]*\d*(?:\.\d+)?[a-zA-Z%]")


def strip_comments(source: str) -> str:
    """
    Remove comments without removing code, keeping line numbers.

    Block comments are replaced by their newlines so a finding's line number is
    still the line in the file. A `//` inside a string literal ("https://...") is
    not a comment, so literals are skipped over rather than cut.
    """
    source = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.S)
    out: List[str] = []
    for line in source.split("\n"):
        out.append(_strip_line_comment(line))
    return "\n".join(out)


def _strip_line_comment(line: str) -> str:
    in_string = False
    i = 0
    while i < len(line):
        char = line[i]
        if in_string:
            if char == "\\":
                i += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif line.startswith("//", i):
            return line[:i]
        i += 1
    return line


def find_literals(source: str) -> List[Tuple[int, str]]:
    """The user-facing literals in one Kotlin source, as (line, literal)."""
    marked = {
        number
        for number, line in enumerate(source.split("\n"), start=1)
        if MARKER in line
    }
    code = strip_comments(source)
    findings: List[Tuple[int, str]] = []
    raw_spans = []
    for match in RAW_LITERAL.finditer(code):
        raw_spans.append(match.span())
        _consider(code, match.start(), match.group(1), marked, findings)
    for match in LITERAL.finditer(code):
        if any(start <= match.start() < end for start, end in raw_spans):
            continue
        _consider(code, match.start(), match.group(1), marked, findings)
    return sorted(findings)


def _consider(code: str, start: int, literal: str, marked: set, findings: list) -> None:
    # Template expressions are code, not text: "${a}${b}" holds no words.
    words = re.sub(r"\$\{[^}]*\}|\$[A-Za-z_]\w*", "", literal)
    if not HAS_LETTER.search(words):
        return
    line = code[:start].count("\n") + 1
    if line in marked:
        return
    if ICON_NAME.search(code[max(0, start - 40):start]):
        return
    findings.append((line, literal))


def kotlin_sources() -> Iterable[Path]:
    yield from sorted(COMPOSE.rglob("*.kt"))
    yield from (path for path in SCREEN_STATE if path.exists())


def read_strings(path: Path) -> Dict[str, str]:
    """The `<string>` entries of one strings.xml, name to text."""
    root = ET.parse(path).getroot()
    return {
        node.get("name"): "".join(node.itertext())
        for node in root.iter("string")
        if node.get("translatable") != "false"
    }


def check_translations(res: Path = RES) -> List[str]:
    """Every problem with the translated string resources under ``res``."""
    problems: List[str] = []
    base = read_strings(res / "values" / "strings.xml")
    present = {
        path.name[len("values-"):]: path
        for path in sorted(res.glob("values-*"))
        if (path / "strings.xml").exists()
    }
    for code in REQUIRED_TRANSLATIONS:
        if code not in present:
            problems.append(f"values-{code}/strings.xml is missing")

    codes = {lang.code for lang in SUPPORTED_LANGUAGES}
    for qualifier, directory in present.items():
        language = get_language(qualifier)
        if qualifier not in codes or language is None:
            problems.append(
                f"values-{qualifier}: not a Supported Language code "
                f"(use one of {sorted(codes)}; Odia is 'or')"
            )
            continue
        translated = read_strings(directory / "strings.xml")
        for name in sorted(set(base) - set(translated)):
            problems.append(f"values-{qualifier}: '{name}' is not translated")
        for name in sorted(set(translated) - set(base)):
            problems.append(f"values-{qualifier}: '{name}' is not in values/strings.xml")
        for name in sorted(set(base) & set(translated)):
            text = translated[name]
            if Counter(PLACEHOLDER.findall(text)) != Counter(PLACEHOLDER.findall(base[name])):
                problems.append(f"values-{qualifier}: '{name}' has different placeholders")
            scripts = scripts_in(PLACEHOLDER.sub("", text))
            if scripts != {language.script}:
                problems.append(
                    f"values-{qualifier}: '{name}' is not all in {language.script} "
                    f"(found {sorted(scripts)}): {text!r}"
                )
    return problems


def main() -> int:
    findings = []
    for path in kotlin_sources():
        for line, literal in find_literals(path.read_text(encoding="utf-8")):
            findings.append(f'{path.relative_to(ROOT)}:{line}  "{literal}"')
    problems = check_translations()

    if findings:
        print("User-facing literals must be string resources:")
        for finding in findings:
            print(f"  {finding}")
        print(
            "\nEvery one of these renders in English whatever the phone's language. "
            "Move it to strings.xml, or mark a line that is truly never shown with "
            f"'{MARKER} <why>'."
        )
    if problems:
        print("String resources are incomplete:")
        for problem in problems:
            print(f"  {problem}")
    if findings or problems:
        return 1
    print("No user-facing literals in Compose sources; every translation is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
