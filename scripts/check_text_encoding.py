from __future__ import annotations

import argparse
from pathlib import Path

TEXT_EXTS = {
    '.md', '.py', '.txt', '.json', '.toml', '.yml', '.yaml', '.html', '.css', '.js'
}
TEXT_NAMES = {'.gitignore', '.gitattributes', '.editorconfig'}


def is_text_file(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_EXTS


def iter_targets(root: Path, targets: list[str]) -> list[Path]:
    if not targets:
        return [p for p in root.rglob('*') if p.is_file() and is_text_file(p)]

    resolved: list[Path] = []
    for target in targets:
        path = (root / target).resolve()
        if path.is_dir():
            resolved.extend(p for p in path.rglob('*') if p.is_file() and is_text_file(p))
        elif path.is_file() and is_text_file(path):
            resolved.append(path)
    return sorted(set(resolved))


def normalize_text(raw: bytes) -> str:
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    text = raw.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if not text.endswith('\n'):
        text += '\n'
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description='Check or normalize text encodings in the repo.')
    parser.add_argument('paths', nargs='*', help='Optional files or directories to scan, relative to repo root.')
    parser.add_argument('--write', action='store_true', help='Rewrite files as UTF-8 without BOM and LF line endings.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    issues: list[str] = []
    rewrites: list[Path] = []

    for path in iter_targets(root, args.paths):
        raw = path.read_bytes()
        rel = path.relative_to(root)

        try:
            normalized = normalize_text(raw)
        except UnicodeDecodeError as exc:
            issues.append(f'{rel}: non-UTF-8 decode error -> {exc}')
            continue

        current_text = raw.decode('utf-8-sig', errors='strict')
        has_bom = raw.startswith(b'\xef\xbb\xbf')
        has_crlf = b'\r\n' in raw or b'\r' in raw.replace(b'\r\n', b'')
        has_final_newline = current_text.endswith('\n')

        if has_bom or has_crlf or not has_final_newline:
            issues.append(
                f'{rel}: needs normalization '
                f'(bom={has_bom}, crlf={has_crlf}, final_newline={has_final_newline})'
            )
            if args.write:
                path.write_text(normalized, encoding='utf-8', newline='\n')
                rewrites.append(rel)

    if args.write:
        if rewrites:
            print('Normalized files:')
            for item in rewrites:
                print(f'- {item}')
        else:
            print('No files needed normalization.')
        return 0

    if issues:
        print('Encoding issues found:')
        for item in issues:
            print(f'- {item}')
        return 1

    print('All checked text files are UTF-8, BOM-free, and LF-normalized.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
