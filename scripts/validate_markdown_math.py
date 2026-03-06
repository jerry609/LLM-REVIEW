from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FENCE = re.compile(r"^```")
INLINE_MATH = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def strip_code_fences(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if CODE_FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def check_display_math(path: Path, text: str) -> list[str]:
    cleaned = strip_code_fences(text)
    if cleaned.count("$$") % 2:
        return [f"{path}: unmatched $$ block"]
    return []


def check_table_pipes(path: Path, text: str) -> list[str]:
    issues = []
    cleaned = strip_code_fences(text)
    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        if not line.lstrip().startswith("|"):
            continue
        line_wo_escapes = line.replace(r"\|", "")
        for segment in INLINE_MATH.findall(line_wo_escapes):
            if "|" in segment:
                issues.append(f"{path}:{lineno}: raw pipe inside inline math in table -> {segment}")
                break
    return issues


def check_links(path: Path, text: str) -> list[str]:
    issues = []
    for target in LINK_PATTERN.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        resolved = (ROOT / path.parent / target_path).resolve()
        if not resolved.exists():
            issues.append(f"{path}: missing local link target -> {target}")
    return issues


def main() -> int:
    issues: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        issues.extend(check_display_math(rel, text))
        issues.extend(check_table_pipes(rel, text))
        issues.extend(check_links(rel, text))

    if issues:
        print("Documentation validation failed:\n")
        for item in issues:
            print(f"- {item}")
        return 1

    print("Documentation validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
