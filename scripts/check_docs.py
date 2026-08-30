#!/usr/bin/env python3
"""Verify that every image and relative link in the documentation resolves.

Broken figure links are the most common way a portfolio README rots: a figure is
renamed, the markdown still points at the old path, and the page renders with a
row of broken-image icons for everyone except the author, whose local checkout
still has the file.  This runs in CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    bad = []
    for doc in DOCS:
        if not doc.exists():
            continue
        for target in LINK.findall(doc.read_text()):
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).resolve().exists():
                bad.append(f"{doc.relative_to(ROOT)} -> {target}")
    for b in bad:
        print(f"broken link: {b}", file=sys.stderr)
    print(f"checked {len(DOCS)} documents, {len(bad)} broken link(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
