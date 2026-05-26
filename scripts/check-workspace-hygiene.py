#!/usr/bin/env python3
"""Detect duplicate/conflict artifacts before they reach tests or packages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("apps", "packages", "docs", "scripts")
GENERATED_DIRS = {
    ".git",
    ".next",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    "runtime",
    "test-results",
    "playwright-report",
}
CONFLICT_NAME_RE = re.compile(r"(?:\s+\d+|\s+copy)(?=\.[^.]+$)", re.IGNORECASE)
STALE_SUFFIX_RE = re.compile(r"\.(?:bak|orig)$", re.IGNORECASE)
GIT_REF_CONFLICT_RE = re.compile(r"\s+(?:2|6|10|\d+)(?:$|[./-])")


def is_conflict_artifact(path: Path) -> bool:
    name = path.name
    return bool(CONFLICT_NAME_RE.search(name) or STALE_SUFFIX_RE.search(name))


def should_skip(path: Path, *, include_generated: bool) -> bool:
    if include_generated:
        return False
    return any(part in GENERATED_DIRS for part in path.relative_to(ROOT).parts)


def find_workspace_conflicts(*, include_generated: bool) -> list[Path]:
    conflicts: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if should_skip(path, include_generated=include_generated):
                continue
            if path.is_file() and is_conflict_artifact(path):
                conflicts.append(path)
    return sorted(conflicts)


def find_git_ref_conflicts() -> list[str]:
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        return []
    findings: list[str] = []
    for ref_root in (git_dir / "refs", git_dir / "logs" / "refs"):
        if not ref_root.exists():
            continue
        for path in ref_root.rglob("*"):
            if path.is_file() and is_conflict_artifact(path):
                findings.append(str(path.relative_to(ROOT)))
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        try:
            for line_number, line in enumerate(packed_refs.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if GIT_REF_CONFLICT_RE.search(line):
                    findings.append(f".git/packed-refs:{line_number}: {line}")
        except OSError:
            findings.append(".git/packed-refs: unreadable")
    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-generated", action="store_true", help="also scan generated/cache directories")
    parser.add_argument("--ignore-git-refs", action="store_true", help="only report workspace conflicts")
    args = parser.parse_args(argv)

    workspace_conflicts = find_workspace_conflicts(include_generated=args.include_generated)
    git_ref_conflicts = [] if args.ignore_git_refs else find_git_ref_conflicts()

    if workspace_conflicts:
        print("Duplicate/conflict workspace artifacts:")
        for path in workspace_conflicts:
            print(f"  {path.relative_to(ROOT)}")
    if git_ref_conflicts:
        print("Duplicate/conflict git refs:")
        for item in git_ref_conflicts:
            print(f"  {item}")
    if not workspace_conflicts and not git_ref_conflicts:
        print("Workspace hygiene OK: no duplicate/conflict artifacts found.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
