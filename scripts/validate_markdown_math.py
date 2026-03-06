from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FENCE = re.compile(r'^```')
INLINE_MATH = re.compile(r'(?<!\$)\$[^$\n]+\$(?!\$)')
LINK_PATTERN = re.compile(r'\[[^\]]+\]\(([^)]+)\)')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def tracked_markdown_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ['git', 'ls-files', '-z', '--', '*.md'],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(root.rglob('*.md'))

    paths: list[Path] = []
    for raw in result.stdout.split(b'\0'):
        if not raw:
            continue
        path = root / raw.decode('utf-8')
        if path.is_file() and path.suffix.lower() == '.md':
            paths.append(path)
    return sorted(set(paths))


def iter_markdown_targets(root: Path, targets: list[str]) -> list[Path]:
    if not targets:
        return tracked_markdown_files(root)

    resolved: list[Path] = []
    for target in targets:
        candidate = Path(target)
        path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not path.exists():
            continue
        if path.is_dir():
            resolved.extend(p for p in path.rglob('*.md') if p.is_file())
        elif path.is_file() and path.suffix.lower() == '.md':
            resolved.append(path)
    return sorted(set(resolved))


def strip_code_fences(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        if CODE_FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return '\n'.join(lines)


def check_display_math(path: Path, text: str) -> list[str]:
    cleaned = strip_code_fences(text)
    if cleaned.count('$$') % 2:
        return [f'{path}: unmatched $$ block']
    return []


def check_table_pipes(path: Path, text: str) -> list[str]:
    issues = []
    cleaned = strip_code_fences(text)
    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        if not line.lstrip().startswith('|'):
            continue
        line_wo_escapes = line.replace(r'\|', '')
        for segment in INLINE_MATH.findall(line_wo_escapes):
            if '|' in segment:
                issues.append(f'{path}:{lineno}: raw pipe inside inline math in table -> {segment}')
                break
    return issues


def check_links(path: Path, text: str) -> list[str]:
    issues = []
    for target in LINK_PATTERN.findall(text):
        if target.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        target_path = target.split('#', 1)[0]
        resolved = (ROOT / path.parent / target_path).resolve()
        if not resolved.exists():
            issues.append(f'{path}: missing local link target -> {target}')
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate markdown math blocks and local links.')
    parser.add_argument('paths', nargs='*', help='Optional markdown files or directories to scan.')
    args = parser.parse_args()

    issues: list[str] = []
    for path in iter_markdown_targets(ROOT, args.paths):
        text = path.read_text(encoding='utf-8')
        rel = path.relative_to(ROOT)
        issues.extend(check_display_math(rel, text))
        issues.extend(check_table_pipes(rel, text))
        issues.extend(check_links(rel, text))

    if issues:
        print('Documentation validation failed:\n')
        for item in issues:
            print(f'- {item}')
        return 1

    print('Documentation validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
