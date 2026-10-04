#!/usr/bin/env python3
"""
Fail if a user-facing English literal reaches a Compose screen.

Criterion 4 says no screen requires English to operate. Auditing for that by reading
found three hardcoded literals in one sitting: "Ask a question", "Stop speaking",
"Play the answer again", all sitting inside a Hindi app. A screen is fine and a
neighbouring screen is broken, which is exactly what an audit misses.

This makes it a build failure instead. It is a lint rather than a test because the
thing being checked is a source pattern, not a behaviour, and putting it in the test
suite would mean a failing Python test nobody can act on without reading Compose
source.

Scope is deliberately narrow. It flags an English word inside Text(...) or
contentDescription, and ignores comments and string resources, because those are the
places English legitimately lives.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "android-app" / "app" / "src" / "main" / "java"

# Any literal handed to a user-facing composable is English. There is no point
# trying to recognise it by shape: an earlier version required two capitalised words
# and so missed every single-word label, which is most of them. "Earlier", "Ask" and
# "Working" all sailed through while sitting in the middle of a Hindi screen.
#
# Only genuinely non-textual values are exempt, and they are named rather than
# pattern-matched, because a pattern here is exactly what let those through.
NOT_USER_FACING = {"", "0%", "1"}

# Matches both call and assignment forms. The first version required a parenthesis
# and so caught Text("...") while missing contentDescription = "...", which is the
# form most of this app's own semantics blocks use. Half a check reads as a pass.
# Case-insensitive on the name because Compose parameters are lowercase: `text = "..."`
# is at least as common as Text("..."), and a case-sensitive pattern misses every
# one of them. That is not hypothetical: it hid two real strings on the ask screen
# immediately after the check was added.
SUSPECT = re.compile(
    r'(?:Text|contentDescription|text)\s*(?:\(\s*)?(?:=\s*)?"([^"]+)"',
    re.IGNORECASE,
)

# Nothing is exempted. Every literal found here reaches a user in the user's own
# language, including the ones that look like proper nouns.
ALLOWED: set[str] = set()


def strip_comments(source: str) -> str:
    """
    Remove comments without removing code.

    The first version applied re.S to both patterns. That makes "." match newlines,
    so the line-comment pattern deleted everything from the first // to the end of
    the file. Half of every source file was invisible to the check, which is how a
    contentDescription assignment went unflagged while the check reported clean.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def main() -> int:
    findings = []
    for path in sorted(COMPOSE.rglob("*.kt")):
        source = strip_comments(path.read_text(encoding="utf-8"))
        for match in SUSPECT.finditer(source):
            literal = match.group(1)
            if literal in NOT_USER_FACING:
                continue
            # No identifier exemption. Python identifiers may start with a capital,
            # so isidentifier() accepted "Earlier", "Ask" and "Working", which are
            # exactly the labels that were slipping through. Anything handed to a
            # user-facing composable is text, so nothing here needs to be excused.
            line = source[: match.start()].count("\n") + 1
            findings.append(f'{path.relative_to(ROOT)}:{line}  "{literal}"')

    if not findings:
        print("No user-facing English literals in Compose sources.")
        return 0

    print("User-facing English literals must be string resources:")
    for finding in findings:
        print(f"  {finding}")
    print(
        "\nEvery one of these renders in the user's language and is untranslatable. "
        "Move it to strings.xml."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
