#!/usr/bin/env python3
"""
Fail if a container holding text is given a fixed height.

Criterion 5 is legibility at the largest system font, and the way that fails is not
unreadable text, it is clipped text. A container with `.height(56.dp)` keeps 56dp
when the font grows to twice its size, and the label is cut off along the bottom
edge. Nothing reports an error, the text is still in the semantics tree, and a user
who cannot read it has no way to learn why.

A rendering test cannot catch this here. Robolectric performs no real layout, so
measured heights come back zero and a comparison between two of them proves nothing.
The precondition is checkable statically, so that is what is checked.

`heightIn(min = ...)` is the fix: the touch target keeps its floor and the container
grows with the label.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "android-app" / "app" / "src" / "main" / "java"

FIXED_HEIGHT = re.compile(r"\.height\(\s*[0-9.]+\s*\.dp\s*\)")


def main() -> int:
    findings = []
    for path in sorted(COMPOSE.rglob("*.kt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not FIXED_HEIGHT.search(line):
                continue
            # A Spacer is fixed by nature and holds no text.
            if "Spacer" in line:
                continue
            # Look forward a little: a text-bearing container is one whose next few
            # lines mention Text or a glyph.
            window = "\\n".join(lines[number - 1: number + 12])
            if "Text(" in window or "Icon(" in window:
                findings.append(
                    f"{path.relative_to(ROOT)}:{number}  {line.strip()}"
                )

    if not findings:
        print("No text-bearing container uses a fixed height.")
        return 0

    print("Containers holding text must use heightIn(min = ...), not height(...):")
    for finding in findings:
        print(f"  {finding}")
    print(
        "\nAt twice the system font a fixed height clips the label off the bottom. "
        "The text stays in the tree and nothing reports an error."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
