"""Evidence-first edit target selection.

This module is intentionally feature-agnostic. It turns repository evidence
that the agent already has into ranked edit target candidates and keeps the
diagnostics public-safe for debug/context surfaces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.request_parsing import extract_file_tokens
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".java",
    ".kt",
    ".swift",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}
PYTHON_ENTRYPOINTS = {"main.py", "bot.py", "app.py", "server.py", "__main__.py"}
JS_ENTRYPOINTS = {
    "index.js",
    "index.ts",
    "index.jsx",
    "index.tsx",
    "main.js",
    "main.ts",
    "main.jsx",
    "main.tsx",
    "app.js",
    "app.ts",
    "app.jsx",
    "app.tsx",
    "server.js",
    "server.ts",
}
CONFIG_BASENAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "next.config.ts",
    "next.config.js",
    "Makefile",
    "Dockerfile",
}
DOC_SUFFIXES = {".md", ".rst", ".txt"}
TEST_PARTS = {"test", "tests", "__tests__", "spec"}
SKIP_DIRS = {
    ".git",
    ".claude",
    "node_modules",
    "runtime",
    ".next",
    "dist",
    "build",
    "out",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c_cpp",
    ".cpp": "c_cpp",
    ".h": "c_cpp",
    ".hpp": "c_cpp",
}

CONFIG_LANGUAGE_HINTS = {
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "package.json": "typescript",
    "tsconfig.json": "typescript",
    "vite.config.ts": "typescript",
    "next.config.ts": "typescript",
    "vite.config.js": "javascript",
    "next.config.js": "javascript",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
}

LANGUAGE_FALLBACKS = {
    "python": {
        "queries": ["*.py", "main.py", "bot.py", "app.py", "server.py", "__main__.py", "requirements.txt", "pyproject.toml", "setup.py"],
        "file_globs": ["cogs/*.py", "**/cogs/*.py", "**/*.py"],
    },
    "typescript": {
        "queries": ["*.ts", "*.tsx", "*.js", "*.jsx", "package.json", "tsconfig.json"],
        "file_globs": ["src/index.*", "src/main.*", "src/app.*", "src/server.*", "src/**/*.ts", "src/**/*.tsx", "app/**/*.tsx", "pages/**/*.tsx"],
    },
    "javascript": {
        "queries": ["*.js", "*.jsx", "*.ts", "*.tsx", "package.json"],
        "file_globs": ["index.*", "server.*", "src/index.*", "src/app.*", "src/server.*", "src/**/*.js", "src/**/*.jsx", "app/**/*.jsx", "pages/**/*.jsx"],
    },
    "go": {"queries": ["*.go", "go.mod", "main.go"], "file_globs": ["**/*.go"]},
    "rust": {"queries": ["*.rs", "Cargo.toml", "main.rs", "lib.rs"], "file_globs": ["src/**/*.rs"]},
    "java": {"queries": ["*.java", "pom.xml", "build.gradle"], "file_globs": ["src/**/*.java"]},
    "kotlin": {"queries": ["*.kt", "build.gradle", "settings.gradle"], "file_globs": ["src/**/*.kt"]},
    "csharp": {"queries": ["*.cs", "*.csproj"], "file_globs": ["**/*.cs"]},
    "ruby": {"queries": ["*.rb", "Gemfile"], "file_globs": ["**/*.rb"]},
    "php": {"queries": ["*.php", "composer.json"], "file_globs": ["**/*.php"]},
    "c_cpp": {"queries": ["*.c", "*.cpp", "*.h", "*.hpp", "Makefile"], "file_globs": ["src/**/*.c", "src/**/*.cpp", "include/**/*.h"]},
}

TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "into",
    "from",
    "please",
    "add",
    "fix",
    "update",
    "change",
    "implement",
    "refactor",
    "support",
    "feature",
    "request",
    "code",
    "file",
    "repo",
    "project",
    "make",
    "create",
    "use",
    "using",
    "without",
    "before",
    "after",
    "continue",
}

STRONG_TARGET_SCORE = 70.0


@dataclass
class ProjectLanguageProfile:
    dominant_language: str
    source_counts: dict[str, int] = field(default_factory=dict)
    config_files: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)
    fallback_queries: list[str] = field(default_factory=list)
    fallback_file_globs: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass
class EditTargetCandidate:
    path: str
    score: float = 0.0
    role: str = "other"
    language: str | None = None
    already_read: bool = False
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)
    prior_reused: bool = False

    def add(self, amount: float, reason: str, source: str | None = None) -> None:
        self.score += amount
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        if source and source not in self.sources:
            self.sources.append(source)
        if source and source.startswith("prior"):
            self.prior_reused = True

    @property
    def confidence(self) -> float:
        return round(max(0.0, min(1.0, self.score / 120.0)), 3)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "path": self.path,
                "score": round(self.score, 2),
                "confidence": self.confidence,
                "role": self.role,
                "language": self.language,
                "already_read": self.already_read,
                "sources": self.sources,
                "reasons": self.reasons[:12],
                "symbols": self.symbols[:20],
                "imports": self.imports[:20],
                "matched_terms": self.matched_terms[:20],
                "framework_hints": self.framework_hints[:12],
                "prior_reused": self.prior_reused,
            }
        )


@dataclass
class TargetSelectionResult:
    candidates: list[EditTargetCandidate]
    selected_target_files: list[str]
    project_profile: ProjectLanguageProfile
    prior_evidence_reused: bool = False
    fallback_attempts: int = 0
    failed_search_patterns: list[str] = field(default_factory=list)
    discovery_queries: list[str] = field(default_factory=list)
    discovery_text_queries: list[str] = field(default_factory=list)
    discovery_file_globs: list[str] = field(default_factory=list)
    blocked_reason: str | None = None

    @property
    def strong_read_targets(self) -> list[EditTargetCandidate]:
        return [item for item in self.candidates if item.already_read and item.score >= STRONG_TARGET_SCORE and _is_implementation_role(item.role)]

    @property
    def unread_promotable_targets(self) -> list[EditTargetCandidate]:
        return [item for item in self.candidates if not item.already_read and item.score >= STRONG_TARGET_SCORE and _is_implementation_role(item.role)]

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "selected_target_files": self.selected_target_files,
                "strong_target_threshold": STRONG_TARGET_SCORE,
                "prior_evidence_reused": self.prior_evidence_reused,
                "fallback_attempts": self.fallback_attempts,
                "failed_search_patterns": self.failed_search_patterns,
                "discovery_queries": self.discovery_queries,
                "discovery_text_queries": self.discovery_text_queries,
                "discovery_file_globs": self.discovery_file_globs,
                "blocked_reason": self.blocked_reason,
                "project_profile": self.project_profile.model_dump(),
                "candidates": [item.model_dump() for item in self.candidates],
            }
        )


def select_edit_target_candidates(
    state: Any,
    frame: Any,
    request: AgentRunRequest,
    *,
    model_targets: list[str] | None = None,
    record: bool = True,
) -> TargetSelectionResult:
    repo = _repo_path(request)
    profile = detect_project_language_profile(request, state=state)
    contents = _read_contents_by_path(state)
    files_read = _state_files_read(state)
    all_files = _repository_files(repo)
    semantic_terms = _semantic_terms(frame, request)
    explicit = _resolve_paths(repo, getattr(frame, "mentioned_files", []) or [], preferred=[*files_read, *all_files])
    model_paths = _resolve_paths(repo, model_targets or [], preferred=[*files_read, *all_files])
    ide_paths = _ide_context_targets(state, request)
    prior_paths = _prior_target_paths(state, request)
    prior_candidates = _prior_candidate_records(state, request)
    current_anchor_paths = _dedupe([*explicit, *model_paths, *ide_paths])
    current_symbols = _current_mentioned_symbols(state, frame)
    prior_context_text = _prior_context_text_by_path(state, request)
    candidates: dict[str, EditTargetCandidate] = {}

    def candidate_for(path: str) -> EditTargetCandidate | None:
        cleaned = str(path).strip().lstrip("/")
        if not cleaned or cleaned not in all_files:
            return None
        item = candidates.get(cleaned)
        if item is None:
            item = EditTargetCandidate(
                path=cleaned,
                role=file_role(cleaned),
                language=language_for_path(cleaned),
                already_read=cleaned in files_read or cleaned in contents,
            )
            candidates[cleaned] = item
        return item

    for path in explicit:
        if item := candidate_for(path):
            item.add(120.0, "explicitly mentioned by the current user request", "explicit_request")
    for path in ide_paths:
        if item := candidate_for(path):
            item.add(90.0, "active editor context points at this file", "ide_context")
    for path in model_paths:
        if item := candidate_for(path):
            item.add(70.0, "model planner proposed this file", "model_target")
    for path in files_read:
        if item := candidate_for(path):
            if path in contents:
                item.add(30.0, "file was already read in this run", "read_file")
            else:
                item.add(8.0, "file path was recorded as read but content evidence is not available", "read_file_path_only")
    for record_item in _search_candidate_records(state):
        path = str(record_item.get("path") or "")
        if item := candidate_for(path):
            search_score = max(0.0, min(75.0, _score_value(record_item.get("score"), default=0.0)))
            item.add(search_score, "repository search ranked this file", "search_result")
    for record_item in prior_candidates:
        path = str(record_item.get("path") or "")
        if not _prior_candidate_compatible(
            path,
            record_item,
            prior_context_text.get(path, ""),
            semantic_terms=semantic_terms,
            current_symbols=current_symbols,
            current_anchor_paths=current_anchor_paths,
        ):
            continue
        if item := candidate_for(path):
            prior_score = max(40.0, min(80.0, _score_value(record_item.get("score"), default=65.0)))
            item.add(prior_score, "structured prior turn target candidate", "prior_target_candidate")
    for path in prior_paths:
        if not _prior_candidate_compatible(
            path,
            {},
            prior_context_text.get(path, ""),
            semantic_terms=semantic_terms,
            current_symbols=current_symbols,
            current_anchor_paths=current_anchor_paths,
        ):
            continue
        if item := candidate_for(path):
            source = "prior_proposal_target" if path in _prior_proposal_paths(state, request) else "prior_context_file"
            amount = 72.0 if source == "prior_proposal_target" else 28.0
            item.add(amount, "prior turn context identified this file", source)

    for path, item in list(candidates.items()):
        content = contents.get(path, "")
        _score_file_evidence(item, content, semantic_terms, profile, contents)
        if not _is_viable_candidate(item):
            item.add(-80.0, "file role is not a safe default implementation target")
    _downweight_prior_conflicts(
        candidates.values(),
        semantic_terms=semantic_terms,
        current_symbols=current_symbols,
        current_anchor_paths=current_anchor_paths,
    )

    # If the agent has already read source files, make sure they are considered
    # even when later fallback searches were empty.
    for path in files_read:
        if path in candidates:
            continue
        if item := candidate_for(path):
            content = contents.get(path, "")
            if content:
                item.add(30.0, "file was already read in this run", "read_file")
            else:
                item.add(8.0, "file path was recorded as read but content evidence is not available", "read_file_path_only")
            _score_file_evidence(item, content, semantic_terms, profile, contents)

    ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.path))
    selected = _selected_targets(ordered, explicit=explicit, model_paths=model_paths, ide_paths=ide_paths)
    discovery = language_aware_edit_discovery(request, frame, state=state, profile=profile)
    result = TargetSelectionResult(
        candidates=ordered[:12],
        selected_target_files=selected,
        project_profile=profile,
        prior_evidence_reused=any(item.prior_reused for item in ordered),
        fallback_attempts=_fallback_attempt_count(state),
        failed_search_patterns=_state_zero_result_queries(state),
        discovery_queries=discovery["queries"],
        discovery_text_queries=discovery["text_queries"],
        discovery_file_globs=discovery["file_globs"],
        blocked_reason=None if selected else _blocked_reason(ordered),
    )
    if record:
        record_target_selection(state, result)
    return result


def record_target_selection(state: Any, result: TargetSelectionResult) -> None:
    payload = result.model_dump()
    if hasattr(state, "edit_target_candidates"):
        state.edit_target_candidates = payload["candidates"]
    if hasattr(state, "target_selection_diagnostics"):
        state.target_selection_diagnostics = payload
    if result.prior_evidence_reused and hasattr(state, "memories_used"):
        marker = "prior_edit_target_evidence"
        if marker not in state.memories_used:
            state.memories_used.append(marker)
    if hasattr(state, "recommendation_context"):
        existing = state.recommendation_context if isinstance(state.recommendation_context, dict) else {}
        state.recommendation_context = json_safe({**existing, "target_selection": payload, "edit_target_candidates": payload["candidates"]})


def detect_project_language_profile(request: AgentRunRequest, *, state: Any | None = None) -> ProjectLanguageProfile:
    repo = _repo_path(request)
    files = _repository_files(repo)
    counts: dict[str, int] = {}
    config_files: list[str] = []
    for rel in files:
        language = language_for_path(rel)
        if language:
            counts[language] = counts.get(language, 0) + 1
        name = Path(rel).name
        if name in CONFIG_BASENAMES or name in CONFIG_LANGUAGE_HINTS:
            config_files.append(rel)
            hinted = CONFIG_LANGUAGE_HINTS.get(name)
            if hinted:
                counts[hinted] = counts.get(hinted, 0) + 3
    dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0] if counts else "unknown"
    if dominant == "javascript" and counts.get("typescript", 0) >= counts.get("javascript", 0):
        dominant = "typescript"
    framework_hints = _framework_hints(request, state=state, config_files=config_files)
    fallback = LANGUAGE_FALLBACKS.get(dominant) or _generic_fallbacks(files)
    return ProjectLanguageProfile(
        dominant_language=dominant,
        source_counts=dict(sorted(counts.items())),
        config_files=config_files[:20],
        framework_hints=framework_hints,
        fallback_queries=list(fallback.get("queries") or []),
        fallback_file_globs=list(fallback.get("file_globs") or []),
    )


def language_aware_edit_discovery(
    request: AgentRunRequest,
    frame: Any,
    *,
    state: Any | None = None,
    profile: ProjectLanguageProfile | None = None,
) -> dict[str, Any]:
    profile = profile or detect_project_language_profile(request, state=state)
    explicit = [str(item) for item in getattr(frame, "mentioned_files", []) or [] if str(item).strip()]
    symbols = [str(item) for item in getattr(frame, "mentioned_symbols", []) or [] if str(item).strip()]
    terms = _semantic_terms(frame, request)[:8]
    queries = _dedupe([*explicit, *symbols, *terms, *profile.fallback_queries])
    text_queries = _dedupe([*symbols, *terms])[:8]
    file_globs = _dedupe(profile.fallback_file_globs)
    return json_safe(
        {
            "dominant_language": profile.dominant_language,
            "queries": queries[:24],
            "text_queries": text_queries,
            "file_globs": file_globs[:12],
        }
    )


def file_role(path: str) -> str:
    rel = Path(path)
    parts = {part.lower() for part in rel.parts}
    name = rel.name
    lowered_name = name.lower()
    suffix = rel.suffix.lower()
    if parts & TEST_PARTS or lowered_name.startswith("test_") or lowered_name.endswith((".test.ts", ".spec.ts", ".test.js", ".spec.js")):
        return "tests"
    if suffix in DOC_SUFFIXES or lowered_name.startswith("readme"):
        return "docs"
    if name in CONFIG_BASENAMES or lowered_name in {item.lower() for item in CONFIG_BASENAMES}:
        return "configs"
    if suffix in SOURCE_SUFFIXES and (
        name in PYTHON_ENTRYPOINTS
        or name in JS_ENTRYPOINTS
        or lowered_name in {"main.go", "main.rs", "lib.rs", "program.cs"}
        or str(rel).lower().startswith(("src/main.", "src/app.", "app/main."))
    ):
        return "entrypoints"
    if suffix in {".tsx", ".jsx", ".css", ".html"} or parts & {"components", "pages", "ui", "views"}:
        return "UI/components"
    if suffix in SOURCE_SUFFIXES:
        return "app/service modules"
    return "other"


def language_for_path(path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())


def _selected_targets(candidates: list[EditTargetCandidate], *, explicit: list[str], model_paths: list[str], ide_paths: list[str]) -> list[str]:
    strong = [item for item in candidates if item.already_read and item.score >= STRONG_TARGET_SCORE and _is_implementation_role(item.role)]
    if not strong:
        return []
    anchored = [path for path in [*explicit, *ide_paths, *model_paths] if any(item.path == path for item in strong)]
    if anchored:
        return _dedupe(anchored)[:3]
    top = strong[0]
    if top.role == "entrypoints":
        module_owner = next((item for item in strong[1:] if item.role in {"app/service modules", "UI/components"} and top.score - item.score <= 25.0), None)
        if module_owner:
            return [module_owner.path]
    return [top.path]


def _score_file_evidence(
    item: EditTargetCandidate,
    content: str,
    semantic_terms: list[str],
    profile: ProjectLanguageProfile,
    all_contents: dict[str, str],
) -> None:
    if _is_implementation_role(item.role):
        item.add(22.0, "source-like implementation file", "path_role")
    elif item.role == "configs":
        item.add(-25.0, "configuration files are supporting evidence, not default edit targets", "path_role")
    elif item.role == "docs":
        item.add(-35.0, "documentation is supporting evidence, not default implementation", "path_role")
    elif item.role == "tests":
        item.add(-20.0, "test files are lower-priority unless explicitly requested", "path_role")
    if item.role == "entrypoints":
        item.add(10.0, "entrypoint relationship signal", "entrypoint")
    if item.language and profile.dominant_language != "unknown":
        if item.language == profile.dominant_language or {item.language, profile.dominant_language} <= {"javascript", "typescript"}:
            item.add(10.0, f"matches dominant project language: {profile.dominant_language}", "project_language")
        else:
            item.add(-12.0, f"does not match dominant project language: {profile.dominant_language}", "project_language")
    if item.path in _entrypoint_names_for_profile(profile):
        item.add(6.0, "language-aware fallback entrypoint", "language_fallback")

    if content:
        symbols = extract_symbols(content, language=item.language)
        imports = extract_imports(content, language=item.language)
        item.symbols = _dedupe([*item.symbols, *symbols])[:30]
        item.imports = _dedupe([*item.imports, *imports])[:30]
        if symbols:
            item.add(min(22.0, 4.0 * len(symbols)), "file defines implementation symbols", "symbols")
        if imports:
            item.add(4.0, "file has import/dependency relationships", "imports")
        matched = _matched_terms(content, item.path, symbols, semantic_terms)
        item.matched_terms = _dedupe([*item.matched_terms, *matched])
        if matched:
            item.add(min(42.0, 7.0 * len(matched)), "requested behavior terms match file content or symbols", "semantic_match")
        framework_matches = [hint for hint in profile.framework_hints if hint.lower() in content.lower()]
        if framework_matches:
            item.framework_hints = _dedupe([*item.framework_hints, *framework_matches])
            item.add(5.0, "file references detected framework/dependency hints", "framework")
    relationship_hits = _entrypoint_relationship_hits(item.path, all_contents)
    if relationship_hits:
        item.add(min(18.0, 6.0 * len(relationship_hits)), "entrypoint/config relationship references this module", "entrypoint_relationship")


def extract_symbols(content: str, *, language: str | None = None) -> list[str]:
    del language
    patterns = [
        r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
        r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\b",
        r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|async\s+)*class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|async\s+)+[A-Za-z0-9_<>,\[\]?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    ]
    symbols: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, content or "", flags=re.MULTILINE):
            symbol = match.group(1)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols[:40]


def extract_imports(content: str, *, language: str | None = None) -> list[str]:
    del language
    imports: list[str] = []
    patterns = [
        r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))",
        r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"^\s*(?:const|let|var)\s+.*?=\s+require\(['\"]([^'\"]+)['\"]\)",
        r"^\s*using\s+([A-Za-z0-9_\.]+)\s*;",
        r"^\s*import\s+([A-Za-z0-9_\.]+)\s*;",
        r"^\s*use\s+([A-Za-z0-9_:]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, content or "", flags=re.MULTILINE):
            value = next((group for group in match.groups() if group), None)
            if value and value not in imports:
                imports.append(value)
    return imports[:40]


def _semantic_terms(frame: Any, request: AgentRunRequest) -> list[str]:
    values = [
        str(getattr(frame, "user_goal", "") or ""),
        str(request.task or ""),
        " ".join(str(item) for item in getattr(frame, "requested_outputs", []) or []),
        " ".join(str(item) for item in getattr(frame, "mentioned_symbols", []) or []),
    ]
    terms: list[str] = []
    for value in values:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", value):
            lowered = token.lower()
            if lowered in TOKEN_STOPWORDS:
                continue
            terms.append(lowered)
    return _dedupe(terms)[:20]


def _matched_terms(content: str, path: str, symbols: list[str], semantic_terms: list[str]) -> list[str]:
    if not semantic_terms:
        return []
    haystack = " ".join([content[:120_000], path, " ".join(symbols)]).lower()
    return [term for term in semantic_terms if term and term in haystack]


def _entrypoint_relationship_hits(path: str, contents: dict[str, str]) -> list[str]:
    stem = Path(path).stem
    if not stem:
        return []
    hits: list[str] = []
    for other_path, content in contents.items():
        if other_path == path or file_role(other_path) not in {"entrypoints", "configs"}:
            continue
        if stem in content or path in content:
            hits.append(other_path)
    return hits[:6]


def _entrypoint_names_for_profile(profile: ProjectLanguageProfile) -> set[str]:
    if profile.dominant_language == "python":
        return PYTHON_ENTRYPOINTS
    if profile.dominant_language in {"javascript", "typescript"}:
        return JS_ENTRYPOINTS
    return {"main.go", "main.rs", "lib.rs", "program.cs"}


def _framework_hints(request: AgentRunRequest, *, state: Any | None, config_files: list[str]) -> list[str]:
    repo = _repo_path(request)
    hints: list[str] = []
    packet = _context_packet(state)
    for content in _high_signal_contents(packet).values():
        hints.extend(_framework_hints_from_text(content))
    for rel in config_files[:12]:
        target = repo / rel
        if not target.is_file() or not is_supported_text_file(target):
            continue
        try:
            hints.extend(_framework_hints_from_text(target.read_text(encoding="utf-8", errors="replace")[:80_000]))
        except OSError:
            continue
    return _dedupe(hints)[:20]


def _framework_hints_from_text(text: str) -> list[str]:
    lowered = (text or "").lower()
    known = [
        "django",
        "flask",
        "fastapi",
        "discord",
        "click",
        "pytest",
        "react",
        "next",
        "vite",
        "express",
        "vue",
        "svelte",
        "go-chi",
        "gin",
        "spring",
        "junit",
        "tokio",
        "axum",
    ]
    return [item for item in known if item in lowered]


def _generic_fallbacks(files: list[str]) -> dict[str, list[str]]:
    suffixes = _dedupe([f"*{Path(path).suffix.lower()}" for path in files if Path(path).suffix.lower() in SOURCE_SUFFIXES])
    return {"queries": suffixes[:8] or ["*.py", "*.ts", "*.js"], "file_globs": []}


def _repo_path(request: AgentRunRequest) -> Path:
    try:
        return resolve_project_path(request.project_path).resolve()
    except ValueError:
        return Path(request.project_path).resolve()


def _repository_files(repo: Path) -> list[str]:
    files: list[str] = []
    if not repo.exists():
        return files
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part.lower() in SKIP_DIRS for part in rel.parts):
            continue
        if not is_supported_text_file(path):
            continue
        files.append(str(rel))
    return sorted(files, key=lambda item: (0 if _is_implementation_role(file_role(item)) else 1, len(Path(item).parts), item.lower()))


def _resolve_paths(repo: Path, values: list[Any], *, preferred: list[str]) -> list[str]:
    files = _repository_files(repo)
    out: list[str] = []
    for value in values:
        text = str(value).strip().strip("`'\"").lstrip("/")
        if not text:
            continue
        path = repo / text
        try:
            resolved = path.resolve().relative_to(repo)
            if (repo / resolved).is_file():
                out.append(str(resolved))
                continue
        except (OSError, ValueError):
            pass
        lowered = text.lower()
        basename = Path(text).name.lower()
        matches = [item for item in preferred if item.lower() == lowered or Path(item).name.lower() == basename]
        if not matches:
            matches = [item for item in files if item.lower() == lowered or Path(item).name.lower() == basename]
        if not matches and _looks_like_glob(text):
            matches = [item for item in files if Path(item).match(text)]
        out.extend(matches[:4])
    return _dedupe(out)


def _read_contents_by_path(state: Any) -> dict[str, str]:
    contents: dict[str, str] = {}
    packet = _context_packet(state)
    file_evidence = packet.get("file_evidence") if isinstance(packet.get("file_evidence"), dict) else {}
    for key in ("included_files", "summaries"):
        section = file_evidence.get(key) if isinstance(file_evidence.get(key), dict) else {}
        for path, content in section.items():
            contents[str(path)] = str(content or "")
    for path, content in _high_signal_contents(packet).items():
        contents.setdefault(path, content)
    for result in _state_action_results(state):
        payload = _result_payload(result)
        raw = payload.get("contents") if isinstance(payload.get("contents"), dict) else {}
        for path, content in raw.items():
            contents[str(path)] = str(content or "")
    return contents


def _high_signal_contents(packet: dict[str, Any]) -> dict[str, str]:
    contents: dict[str, str] = {}
    for key in ("high_signal_files", "project_instructions"):
        section = packet.get(key) if isinstance(packet.get(key), dict) else {}
        for path, content in section.items():
            contents[str(path)] = str(content or "")
    base = packet.get("base_context") if isinstance(packet.get("base_context"), dict) else {}
    for key in ("high_signal_files", "project_instructions"):
        section = base.get(key) if isinstance(base.get(key), dict) else {}
        for path, content in section.items():
            contents[str(path)] = str(content or "")
    return contents


def _search_candidate_records(state: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in reversed(_state_action_results(state)):
        payload = _result_payload(result)
        details = payload.get("candidate_details") if isinstance(payload.get("candidate_details"), list) else []
        for detail in details:
            if isinstance(detail, dict) and detail.get("path"):
                records.append(detail)
        if records:
            break
        for path in payload.get("candidates") or []:
            if isinstance(path, str):
                records.append({"path": path, "score": 35.0, "reasons": ["candidate list"]})
    return records[:20]


def _prior_candidate_records(state: Any, request: AgentRunRequest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    packet = _context_packet(state)
    thread_context = packet.get("thread_context") if isinstance(packet.get("thread_context"), dict) else {}
    for source in (
        packet.get("prior_target_candidates"),
        thread_context.get("target_candidates"),
        thread_context.get("last_target_candidates"),
        packet.get("edit_target_candidates"),
    ):
        if isinstance(source, list):
            for item in source:
                if isinstance(item, dict) and item.get("path"):
                    records.append(item)
    for message in request.conversation_history[-8:]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        for key in ("edit_target_candidates", "target_candidates", "prior_target_candidates"):
            for item in metadata.get(key) or []:
                if isinstance(item, dict) and item.get("path"):
                    records.append(item)
        proposal = metadata.get("change_set_proposal") if isinstance(metadata.get("change_set_proposal"), dict) else None
        if proposal:
            for path in _proposal_paths(proposal):
                records.append({"path": path, "score": 80.0, "source": "conversation_change_set"})
    return _dedupe_records(records)


def _current_mentioned_symbols(state: Any, frame: Any) -> list[str]:
    symbols: list[str] = []
    request_understanding = getattr(state, "request_understanding", None)
    if isinstance(state, dict):
        request_understanding = state.get("request_understanding") or state.get("request_understanding_snapshot")
    if isinstance(request_understanding, dict):
        symbols.extend(str(item) for item in request_understanding.get("mentioned_symbols") or [] if str(item).strip())
    elif request_understanding is not None:
        symbols.extend(str(item) for item in getattr(request_understanding, "mentioned_symbols", []) or [] if str(item).strip())

    classifier = getattr(state, "classifier_result", None)
    if isinstance(state, dict):
        classifier = state.get("classifier_result") or state.get("classifier_snapshot")
    if isinstance(classifier, dict):
        symbols.extend(str(item) for item in classifier.get("target_symbols") or [] if str(item).strip())
    elif classifier is not None:
        symbols.extend(str(item) for item in getattr(classifier, "target_symbols", []) or [] if str(item).strip())

    for item in getattr(frame, "mentioned_symbols", []) or []:
        text = str(item).strip()
        if _looks_like_code_symbol(text):
            symbols.append(text)
    return _dedupe([symbol.lower() for symbol in symbols if symbol])


def _looks_like_code_symbol(value: str) -> bool:
    if not value:
        return False
    if "." in value or "_" in value:
        return True
    if re.search(r"[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*", value):
        return True
    return value.isupper() and len(value) >= 3


def _prior_candidate_compatible(
    path: str,
    record: dict[str, Any],
    context_text: str,
    *,
    semantic_terms: list[str],
    current_symbols: list[str],
    current_anchor_paths: list[str],
) -> bool:
    cleaned = str(path).strip().lstrip("/")
    if not cleaned:
        return False
    if current_anchor_paths and cleaned not in current_anchor_paths:
        return False

    haystack = " ".join([cleaned, _record_text(record), context_text]).lower()
    if current_symbols:
        path_stem = Path(cleaned).stem.lower()
        return any(symbol and (symbol in haystack or symbol == path_stem) for symbol in current_symbols)

    terms = [term.lower() for term in semantic_terms if term]
    if terms:
        return any(term in haystack for term in terms)
    return bool(record or context_text)


def _downweight_prior_conflicts(
    candidates: Any,
    *,
    semantic_terms: list[str],
    current_symbols: list[str],
    current_anchor_paths: list[str],
) -> None:
    items = list(candidates)
    current_winners = [
        item
        for item in items
        if not item.prior_reused
        and _is_implementation_role(item.role)
        and _has_current_target_signal(item)
        and (item.score >= STRONG_TARGET_SCORE or item.matched_terms)
    ]
    for item in items:
        if not item.prior_reused:
            continue
        if current_anchor_paths and item.path not in current_anchor_paths:
            item.add(-120.0, "current request names a different target", "current_target_conflict")
            continue
        if current_symbols and not _candidate_matches_current_signals(item, semantic_terms=semantic_terms, current_symbols=current_symbols, current_anchor_paths=current_anchor_paths):
            item.add(-70.0, "current request names different symbols", "current_symbol_conflict")
            continue
        if current_winners and not _candidate_matches_current_signals(item, semantic_terms=semantic_terms, current_symbols=current_symbols, current_anchor_paths=current_anchor_paths):
            item.add(-65.0, "current evidence points at a different target", "current_evidence_conflict")


def _has_current_target_signal(item: EditTargetCandidate) -> bool:
    return bool({"explicit_request", "ide_context", "model_target", "search_result", "read_file"} & set(item.sources))


def _candidate_matches_current_signals(
    item: EditTargetCandidate,
    *,
    semantic_terms: list[str],
    current_symbols: list[str],
    current_anchor_paths: list[str],
) -> bool:
    if item.path in current_anchor_paths:
        return True
    symbol_set = {symbol.lower() for symbol in item.symbols}
    if current_symbols and symbol_set & set(current_symbols):
        return True
    if semantic_terms and item.matched_terms:
        return True
    return False


def _prior_context_text_by_path(state: Any, request: AgentRunRequest) -> dict[str, str]:
    texts: dict[str, list[str]] = {}

    def add(path: Any, value: Any) -> None:
        cleaned = str(path or "").strip().lstrip("/")
        if not cleaned:
            return
        text = _record_text(value)
        if text:
            texts.setdefault(cleaned, []).append(text)

    packet = _context_packet(state)
    thread_context = packet.get("thread_context") if isinstance(packet.get("thread_context"), dict) else {}
    for item in thread_context.get("last_target_candidates") or thread_context.get("target_candidates") or []:
        if isinstance(item, dict) and item.get("path"):
            add(item.get("path"), item)
    plan = thread_context.get("last_implementation_plan") if isinstance(thread_context.get("last_implementation_plan"), dict) else {}
    for path in [*(plan.get("target_files") or []), *(plan.get("evidence_files") or [])]:
        add(path, plan)
    proposed = thread_context.get("last_proposed_target_file")
    if proposed:
        add(proposed, thread_context)
    for item in thread_context.get("last_evidence_basis") or []:
        if isinstance(item, dict):
            for path in item.get("paths") or item.get("selected_target_files") or []:
                add(path, item)

    for message in request.conversation_history[-8:]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        message_text = str(message.content or "")
        for item in metadata.get("edit_target_candidates") or metadata.get("target_candidates") or []:
            if isinstance(item, dict) and item.get("path"):
                add(item.get("path"), [item, message_text])
        proposal = metadata.get("change_set_proposal") if isinstance(metadata.get("change_set_proposal"), dict) else None
        if proposal:
            for path in _proposal_paths(proposal):
                add(path, [proposal.get("plan") or {}, message_text])
        plan = metadata.get("implementation_plan") if isinstance(metadata.get("implementation_plan"), dict) else {}
        for path in plan.get("target_files") or []:
            add(path, [plan, message_text])
        for path in extract_file_tokens(message_text):
            add(path, message_text)
    return {path: " ".join(values) for path, values in texts.items()}


def _record_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value[:4_000]
    try:
        return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)[:8_000]
    except (TypeError, ValueError):
        return str(value)[:4_000]


def _prior_target_paths(state: Any, request: AgentRunRequest) -> list[str]:
    paths: list[str] = []
    packet = _context_packet(state)
    thread_context = packet.get("thread_context") if isinstance(packet.get("thread_context"), dict) else {}
    for key in ("last_proposed_target_file", "last_analyzed_file"):
        if thread_context.get(key):
            paths.append(str(thread_context[key]))
    for key in ("last_candidate_files", "recent_files", "prior_files_read"):
        values = thread_context.get(key) or packet.get(key) or []
        if isinstance(values, list):
            paths.extend(str(item) for item in values if str(item))
    for message in request.conversation_history[-8:]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        for key in ("selected_target_file", "proposal_relative_path"):
            if metadata.get(key):
                paths.append(str(metadata[key]))
        for key in ("target_files", "resolved_files", "files_read"):
            values = metadata.get(key) or []
            if isinstance(values, str):
                values = [values]
            paths.extend(str(item) for item in values if str(item))
        if message.role == "assistant":
            paths.extend(extract_file_tokens(message.content or ""))
    return _dedupe(paths)


def _prior_proposal_paths(state: Any, request: AgentRunRequest) -> list[str]:
    paths: list[str] = []
    packet = _context_packet(state)
    thread_context = packet.get("thread_context") if isinstance(packet.get("thread_context"), dict) else {}
    if thread_context.get("last_proposed_target_file"):
        paths.append(str(thread_context["last_proposed_target_file"]))
    for message in request.conversation_history[-8:]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        if metadata.get("proposal_relative_path"):
            paths.append(str(metadata["proposal_relative_path"]))
        proposal = metadata.get("change_set_proposal") if isinstance(metadata.get("change_set_proposal"), dict) else None
        if proposal:
            paths.extend(_proposal_paths(proposal))
    return _dedupe(paths)


def _proposal_paths(proposal: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    plan = proposal.get("plan") if isinstance(proposal.get("plan"), dict) else {}
    paths.extend(str(item) for item in plan.get("target_files") or [] if str(item))
    for change in proposal.get("changes") or []:
        if isinstance(change, dict) and change.get("path"):
            paths.append(str(change["path"]))
    return _dedupe(paths)


def _ide_context_targets(state: Any, request: AgentRunRequest) -> list[str]:
    del request
    packet = _context_packet(state)
    ide_context = packet.get("ide_context") if isinstance(packet.get("ide_context"), dict) else None
    if not ide_context and isinstance(getattr(state, "ide_context", None), dict):
        ide_context = state.ide_context
    if not ide_context:
        return []
    active = str(ide_context.get("active_file") or "")
    return [active] if active else []


def _context_packet(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state.get("context_packet") or {})
    return dict(getattr(state, "context_packet", None) or {})


def _state_action_results(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return list(state.get("action_results") or [])
    return list(getattr(state, "action_results", []) or [])


def _state_files_read(state: Any) -> list[str]:
    if isinstance(state, dict):
        return [str(item) for item in state.get("files_read") or [] if str(item)]
    return [str(item) for item in getattr(state, "files_read", []) or [] if str(item)]


def _state_zero_result_queries(state: Any) -> list[str]:
    if isinstance(state, dict):
        return [str(item) for item in state.get("zero_result_queries") or [] if str(item)]
    return [str(item) for item in getattr(state, "zero_result_queries", []) or [] if str(item)]


def _fallback_attempt_count(state: Any) -> int:
    count = 0
    for action in _state_actions_taken(state):
        payload = _action_payload(action)
        if payload.get("source") in {"edit_discovery", "recovery_filename_search", "recovery_broaden_search"}:
            count += 1
    return count


def _state_actions_taken(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return list(state.get("actions_taken") or [])
    return list(getattr(state, "actions_taken", []) or [])


def _result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result.get("payload") or {})
    return dict(getattr(result, "payload", {}) or {})


def _action_payload(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return dict(action.get("payload") or {})
    return dict(getattr(action, "payload", {}) or {})


def _is_viable_candidate(item: EditTargetCandidate) -> bool:
    if _is_implementation_role(item.role):
        return True
    return any(source in item.sources for source in ("explicit_request", "ide_context", "model_target"))


def _is_implementation_role(role: str) -> bool:
    return role in {"entrypoints", "app/service modules", "UI/components"}


def _blocked_reason(candidates: list[EditTargetCandidate]) -> str | None:
    if not candidates:
        return "no candidate files were found from read evidence, prior context, or bounded search"
    best = candidates[0]
    if best.already_read:
        return f"best read candidate `{best.path}` scored {best.score:.1f}, below the strong target threshold"
    return f"best candidate `{best.path}` has not been read yet"


def _looks_like_glob(value: str) -> bool:
    return "*" in value or "?" in value or "[" in value


def _dedupe(items: list[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        path = str(record.get("path") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(record)
    return out


def _score_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
