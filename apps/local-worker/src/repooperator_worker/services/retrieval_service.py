"""Query-aware file retrieval for the legacy read-only context path.

The active agent loop uses agent_core.context_service and primitive tools. This
module is kept for older read-only context callers and only routes by explicit
file hints or a structured repository-wide flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Tuneable limits ────────────────────────────────────────────────────────────
# Keep total prompt additions within ~30k chars so small local models stay happy.

MAX_FILE_CHARS = 6_000        # per-file content sent to the model
MAX_FILES_RETRIEVED = 8       # hard cap on files per request
MAX_TREE_ENTRIES = 80         # entries in a directory tree listing

# ── Well-known file names ──────────────────────────────────────────────────────

ENTRYPOINT_NAMES: list[str] = [
    "main.py", "app.py", "server.py", "run.py", "__main__.py", "wsgi.py", "asgi.py",
    "main.js", "index.js", "server.js", "app.js",
    "main.ts", "index.ts", "server.ts", "app.ts",
    "main.go", "main.rs", "main.rb", "main.c", "main.cpp",
    "Program.cs", "Main.java",
]

CONFIG_NAMES: list[str] = [
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json",
    "Cargo.toml", "go.mod",
    "requirements.txt", "requirements.in",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example",
    "Makefile",
]

SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".rb", ".php",
    ".cs", ".cpp", ".c", ".h", ".hpp",
    ".swift", ".kt", ".scala", ".elm",
    ".ex", ".exs", ".clj", ".cljs",
    ".sh", ".bash", ".zsh", ".ps1", ".fish",
    ".sql",
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg",
    ".tf", ".hcl", ".proto",
    ".md", ".rst", ".txt",
})

# Directories skipped when walking the tree
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env",
    ".next", "dist", "build", "out",
    ".cache", ".pytest_cache", "htmlcov",
    ".mypy_cache", ".ruff_cache", ".tox",
    "coverage", "target",  # Rust/Java build dirs
})

# ── Structured retrieval intent ────────────────────────────────────────────────

_FILE_RE = re.compile(r'\b([\w./\\-]+\.[a-zA-Z]{1,6})\b')

class QueryType:
    FILE_SPECIFIC = "file_specific"
    DIR_SPECIFIC = "dir_specific"
    PROJECT_REVIEW = "project_review"
    ARCHITECTURE = "architecture"
    DEPENDENCY = "dependency"
    GENERAL = "general"


@dataclass
class StructuredRetrievalIntent:
    """Retrieval hints for callers that already have structured evidence."""
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    file_types_requested: list[str] = field(default_factory=list)
    repository_wide: bool = False


def classify_query(task: str) -> tuple[str, list[str]]:
    """
    Compatibility classifier for callers that have not been migrated to
    StructuredRetrievalIntent yet.

    This intentionally avoids natural-language phrase routing. It only extracts
    explicit file references; otherwise it returns GENERAL so broad workflow
    decisions come from the LLM classifier and validated structured fields.
    """
    file_refs: list[str] = []
    for match in _FILE_RE.finditer(task):
        candidate = match.group(1)
        suffix = Path(candidate).suffix.lower()
        if suffix in SOURCE_EXTENSIONS and not candidate.lower().startswith("http"):
            file_refs.append(candidate)

    if file_refs:
        return QueryType.FILE_SPECIFIC, file_refs
    return QueryType.GENERAL, []


def classify_structured_intent(intent: StructuredRetrievalIntent | dict | None) -> tuple[str, list[str]]:
    if intent is None:
        return QueryType.GENERAL, []
    if isinstance(intent, dict):
        intent = StructuredRetrievalIntent(
            target_files=[str(item) for item in intent.get("target_files") or []],
            target_symbols=[str(item) for item in intent.get("target_symbols") or []],
            file_types_requested=[str(item) for item in intent.get("file_types_requested") or []],
            repository_wide=bool(intent.get("repository_wide")),
        )
    if intent.target_files:
        return QueryType.FILE_SPECIFIC, intent.target_files
    if intent.repository_wide:
        return QueryType.PROJECT_REVIEW, []
    return QueryType.GENERAL, []


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RetrievedFile:
    relative_path: str
    content: str
    truncated: bool

    def format_block(self) -> str:
        trunc = "  [content truncated]" if self.truncated else ""
        return f"=== {self.relative_path}{trunc} ===\n{self.content}"


@dataclass
class RetrievalResult:
    query_type: str
    targets: list[str]
    files: list[RetrievedFile] = field(default_factory=list)
    directory_tree: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def files_read(self) -> list[str]:
        return [f.relative_path for f in self.files]

    def to_context_block(self) -> str:
        """Build the text block injected into the model prompt."""
        parts: list[str] = []
        if self.directory_tree:
            parts.append(f"Repository file tree:\n{self.directory_tree}")
        for rf in self.files:
            parts.append(rf.format_block())
        if self.notes:
            parts.append("Retrieval notes:\n" + "\n".join(f"- {n}" for n in self.notes))
        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not self.files and not self.directory_tree


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_repo_file(repo_path: Path, file_path: Path) -> RetrievedFile:
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as exc:
        return RetrievedFile(
            relative_path=_rel(repo_path, file_path),
            content=f"(could not read: {exc})",
            truncated=False,
        )
    truncated = len(raw) > MAX_FILE_CHARS
    return RetrievedFile(
        relative_path=_rel(repo_path, file_path),
        content=raw[:MAX_FILE_CHARS],
        truncated=truncated,
    )


def _rel(repo_path: Path, file_path: Path) -> str:
    try:
        return str(file_path.relative_to(repo_path))
    except ValueError:
        return file_path.name


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _walk_sources(repo_path: Path, limit: int = MAX_TREE_ENTRIES) -> list[Path]:
    """Walk repo for source files, respecting skip list."""
    found: list[Path] = []
    for path in sorted(repo_path.rglob("*"), key=lambda p: str(p)):
        if len(found) >= limit:
            break
        if _is_skipped(path):
            continue
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
            found.append(path)
    return found


def _build_tree(repo_path: Path) -> str:
    """Build a compact source-file tree for the repo."""
    lines: list[str] = [f"{repo_path.name}/"]
    count = 0
    for path in sorted(repo_path.rglob("*"), key=lambda p: str(p)):
        if count >= MAX_TREE_ENTRIES:
            lines.append("  ... (truncated)")
            break
        if _is_skipped(path):
            continue
        if path.name.startswith(".") and path.is_dir():
            continue
        try:
            rel = path.relative_to(repo_path)
        except ValueError:
            continue
        depth = len(rel.parts) - 1
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{path.name}{suffix}")
        count += 1
    return "\n".join(lines)


def _find_file(repo_path: Path, filename: str) -> list[Path]:
    """
    Find files matching `filename` anywhere in the repo.
    Supports exact name match and substring match (e.g. 'webui.py' finds 'demo_webui.py').
    Returns results sorted shallowest-first.
    """
    target_name = Path(filename).name.lower()
    matches: list[Path] = []

    for path in repo_path.rglob("*"):
        if _is_skipped(path):
            continue
        if not path.is_file():
            continue
        path_name = path.name.lower()
        if path_name == target_name or target_name in path_name:
            matches.append(path)

    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches


# ── Main entry point ───────────────────────────────────────────────────────────

def retrieve_context(repo_path: Path, task: str, intent: StructuredRetrievalIntent | dict | None = None) -> RetrievalResult:
    """Retrieve relevant files from structured hints or explicit task file refs."""
    query_type, targets = classify_structured_intent(intent) if intent is not None else classify_query(task)

    if query_type == QueryType.FILE_SPECIFIC:
        return _retrieve_files(repo_path, targets)
    if query_type == QueryType.DIR_SPECIFIC:
        return _retrieve_directory(repo_path, targets)
    if query_type == QueryType.PROJECT_REVIEW:
        return _retrieve_project_review(repo_path)
    if query_type == QueryType.ARCHITECTURE:
        return _retrieve_architecture(repo_path)
    if query_type == QueryType.DEPENDENCY:
        return _retrieve_dependencies(repo_path)
    return _retrieve_general(repo_path)


# ── Strategies ────────────────────────────────────────────────────────────────

def _retrieve_files(repo_path: Path, filenames: list[str]) -> RetrievalResult:
    result = RetrievalResult(query_type=QueryType.FILE_SPECIFIC, targets=filenames)
    seen: set[Path] = set()

    for filename in filenames[:6]:
        matches = _find_file(repo_path, filename)
        if not matches:
            result.notes.append(f"'{filename}' not found in repository.")
            continue

        target = matches[0]
        if len(matches) > 1:
            result.notes.append(
                f"'{filename}' matched {len(matches)} file(s); reading {_rel(repo_path, target)}."
            )

        if target not in seen and len(result.files) < MAX_FILES_RETRIEVED:
            seen.add(target)
            result.files.append(_read_repo_file(repo_path, target))

        # Also read __init__.py in the same package if present
        init = target.parent / "__init__.py"
        if init.exists() and init not in seen and len(result.files) < MAX_FILES_RETRIEVED:
            seen.add(init)
            result.files.append(_read_repo_file(repo_path, init))

    if not result.files:
        result.notes.append(
            "No requested files could be located. "
            "Showing general project context instead."
        )
        # Graceful fallback
        fallback = _retrieve_general(repo_path)
        result.files = fallback.files
        result.directory_tree = fallback.directory_tree

    return result


def _retrieve_directory(repo_path: Path, dir_names: list[str]) -> RetrievalResult:
    result = RetrievalResult(query_type=QueryType.DIR_SPECIFIC, targets=dir_names)

    for dir_name in dir_names[:2]:
        # Try direct child, then recursive search
        target_dir: Path | None = None
        candidate = repo_path / dir_name
        if candidate.is_dir():
            target_dir = candidate
        else:
            for path in repo_path.rglob("*"):
                if path.is_dir() and path.name.lower() == dir_name.lower():
                    target_dir = path
                    break

        if target_dir is None:
            result.notes.append(f"Directory '{dir_name}' not found.")
            continue

        # Build directory tree
        dir_files = [
            p for p in sorted(target_dir.rglob("*"), key=lambda p: str(p))
            if p.is_file() and not _is_skipped(p)
        ][:MAX_TREE_ENTRIES]

        rel_dir = _rel(repo_path, target_dir)
        tree_lines: list[str] = [f"{rel_dir}/"]
        for f in dir_files:
            try:
                tree_lines.append(f"  {f.relative_to(target_dir)}")
            except ValueError:
                tree_lines.append(f"  {f.name}")
        result.directory_tree = "\n".join(tree_lines)

        # Prioritise non-test source files, entrypoints first
        priority = [f for f in dir_files if f.suffix.lower() in SOURCE_EXTENSIONS]
        priority.sort(key=lambda p: (
            0 if p.name.lower() in {"main.py", "index.py", "app.py", "__init__.py"} else 1,
            len(p.parts),
            str(p),
        ))
        for path in priority[:6]:
            if len(result.files) >= MAX_FILES_RETRIEVED:
                break
            result.files.append(_read_repo_file(repo_path, path))

    return result


def _retrieve_project_review(repo_path: Path) -> RetrievalResult:
    result = RetrievalResult(query_type=QueryType.PROJECT_REVIEW, targets=[])
    result.directory_tree = _build_tree(repo_path)

    to_read: list[Path] = []

    # Config / manifest files
    for name in CONFIG_NAMES:
        path = repo_path / name
        if path.exists() and path not in to_read:
            to_read.append(path)

    # Entrypoints
    for name in ENTRYPOINT_NAMES:
        path = repo_path / name
        if path.exists() and path not in to_read:
            to_read.append(path)

    # Representative source files (skip tests and hidden)
    if len(to_read) < MAX_FILES_RETRIEVED:
        for path in _walk_sources(repo_path):
            if len(to_read) >= MAX_FILES_RETRIEVED:
                break
            if path in to_read:
                continue
            rel_str = str(path.relative_to(repo_path))
            if "test" in rel_str.lower() or path.name.startswith("."):
                continue
            to_read.append(path)

    for path in to_read[:MAX_FILES_RETRIEVED]:
        result.files.append(_read_repo_file(repo_path, path))

    return result


def _retrieve_architecture(repo_path: Path) -> RetrievalResult:
    result = RetrievalResult(query_type=QueryType.ARCHITECTURE, targets=[])
    result.directory_tree = _build_tree(repo_path)

    to_read: list[Path] = []

    # Entrypoints first
    for name in ENTRYPOINT_NAMES:
        path = repo_path / name
        if path.exists() and path not in to_read:
            to_read.append(path)

    # Architecture-flavoured filenames
    arch_keywords = {
        "route", "router", "service", "handler", "controller",
        "middleware", "config", "settings", "app", "api",
    }
    for path in _walk_sources(repo_path):
        if len(to_read) >= MAX_FILES_RETRIEVED:
            break
        if path in to_read:
            continue
        stem = path.stem.lower()
        if any(kw in stem for kw in arch_keywords):
            to_read.append(path)

    # A few top config files
    for name in CONFIG_NAMES[:5]:
        if len(to_read) >= MAX_FILES_RETRIEVED:
            break
        path = repo_path / name
        if path.exists() and path not in to_read:
            to_read.append(path)

    for path in to_read[:MAX_FILES_RETRIEVED]:
        result.files.append(_read_repo_file(repo_path, path))

    return result


def _retrieve_dependencies(repo_path: Path) -> RetrievalResult:
    result = RetrievalResult(query_type=QueryType.DEPENDENCY, targets=[])

    dep_names = {
        "requirements.txt", "requirements.in", "requirements-dev.txt",
        "web_requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
        "package.json", "Cargo.toml", "go.mod", "Gemfile", "composer.json",
    }
    lock_suffixes = {".lock", "-lock.json"}

    # First pass: root level
    for name in sorted(dep_names):
        if len(result.files) >= MAX_FILES_RETRIEVED:
            break
        path = repo_path / name
        if not path.exists():
            continue
        limit = 2_000 if any(name.endswith(s) for s in lock_suffixes) else MAX_FILE_CHARS
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            result.files.append(RetrievedFile(
                relative_path=_rel(repo_path, path),
                content=raw[:limit],
                truncated=len(raw) > limit,
            ))
        except (OSError, PermissionError):
            pass

    # Second pass: recursive search if nothing found at root
    if not result.files:
        for path in sorted(repo_path.rglob("*"), key=lambda p: (len(p.parts), str(p))):
            if len(result.files) >= MAX_FILES_RETRIEVED:
                break
            if _is_skipped(path) or not path.is_file():
                continue
            if path.name in dep_names:
                limit = 2_000 if any(path.name.endswith(s) for s in lock_suffixes) else MAX_FILE_CHARS
                try:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    result.files.append(RetrievedFile(
                        relative_path=_rel(repo_path, path),
                        content=raw[:limit],
                        truncated=len(raw) > limit,
                    ))
                except (OSError, PermissionError):
                    pass

    if not result.files:
        result.notes.append("No dependency manifest files found in this repository.")

    return result


def _retrieve_general(repo_path: Path) -> RetrievalResult:
    """
    Fallback for unclassified queries: read entrypoints + one config file so
    the model has more than just the README to work with.
    """
    result = RetrievalResult(query_type=QueryType.GENERAL, targets=[])

    for name in ENTRYPOINT_NAMES[:5]:
        if len(result.files) >= 4:
            break
        path = repo_path / name
        if path.exists():
            result.files.append(_read_repo_file(repo_path, path))

    for name in CONFIG_NAMES[:3]:
        if len(result.files) >= 5:
            break
        path = repo_path / name
        if path.exists():
            result.files.append(_read_repo_file(repo_path, path))
            break

    return result
