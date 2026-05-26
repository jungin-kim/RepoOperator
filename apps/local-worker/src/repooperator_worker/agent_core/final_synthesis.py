from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.context_budget import ContextBudget, compact_file_contents
from repooperator_worker.agent_core.events import append_work_trace
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient as _OpenAICompatibleModelClient, split_visible_reasoning
from repooperator_worker.services.response_quality_service import clean_user_visible_response

NONPUBLIC_MODEL_DELTA_TYPE = "reasoning" + "_delta"


class FinalSynthesisService:
    """Compatibility seam for final synthesis and deterministic answer repair."""

    def build_answer(self, state: AgentCoreState, request: AgentRunRequest, *, skills_context: str = "", on_delta: Any | None = None) -> str:
        from repooperator_worker.agent_core.graph.final_answer_support import build_final_answer_text

        return build_final_answer_text(state, request, skills_context=skills_context, on_delta=on_delta)

    def validate_or_repair(self, answer: str, state: AgentCoreState, request: AgentRunRequest) -> str:
        return validate_or_repair_final_answer(answer, state, request)


def _answer_with_model(
    request: AgentRunRequest,
    file_contents: dict[str, str],
    *,
    state: AgentCoreState | None = None,
    repo_observation: str = "",
    skills_context: str = "",
    on_delta: Any | None = None,
) -> str:
    try:
        compacted = compact_file_contents(
            file_contents,
            ContextBudget(),
            explicit_files=list(getattr(state, "files_read", []) or []),
        )
        model_files = compacted.included_files
        try:
            resolved_repo = str(resolve_project_path(request.project_path))
        except ValueError:
            resolved_repo = request.project_path
        prompt = ModelGenerationRequest(
            system_prompt=(
                "You are RepoOperator, a local-first coding agent proxy. Answer with visible, evidence-backed "
                "work summaries only. Do not include non-public deliberation. Keep the response grounded in the supplied "
                "repository context. Answer the user's actual request first, synthesize across evidence, and avoid "
                "file-by-file dumps unless the user explicitly asks for one. Never say files were unavailable when "
                "file contents are supplied.\n"
                + (f"\nEnabled skills with provenance:\n{skills_context}\n" if skills_context else "")
            ),
            user_prompt=json.dumps(
                {
                    "task": request.task,
                    "repo": request.project_path,
                    "active_repository": f"source: {request.git_provider}\npath: {resolved_repo}",
                    "branch": request.branch,
                    "repo_observation": repo_observation,
                    "files": model_files,
                    "context_compaction": compacted.model_dump(),
                },
                ensure_ascii=False,
            ),
        )
        pieces: list[str] = []
        client = _compat_model_client()()
        for delta in client.stream_text(prompt):
            if delta.get("type") == NONPUBLIC_MODEL_DELTA_TYPE:
                continue
            elif delta.get("type") == "assistant_delta":
                text = str(delta.get("delta") or "")
                pieces.append(text)
        raw = "".join(pieces) or client.generate_text(prompt)
        _reasoning, visible = split_visible_reasoning(raw)
        cleaned, _ = clean_user_visible_response(visible, user_task=request.task)
        guarded = _quality_guard_answer(cleaned.strip(), file_contents=file_contents)
        if guarded is None and cleaned.strip() and state is not None:
            append_work_trace(
                run_id=state.run_id,
                request=request,
                activity_id="final-synthesis-repair",
                phase="Finished",
                label="Rebuilt final answer",
                status="completed",
                safe_reasoning_summary="The draft answer did not match gathered evidence, so I rebuilt it from collected files.",
                observation="Final answer repaired without storing the rejected draft text.",
                safety_note="Rejected draft text is not exposed in events.",
            )
        accepted = guarded or synthesize_answer_from_evidence(request, state, file_contents, repo_observation)
        if state is not None:
            before_validation = accepted
            accepted = validate_or_repair_final_answer(accepted, state, request)
            if accepted != before_validation:
                append_work_trace(
                    run_id=state.run_id,
                    request=request,
                    activity_id="final-synthesis-repair",
                    phase="Finished",
                    label="Rebuilt final answer",
                    status="completed",
                    safe_reasoning_summary="The draft answer did not match gathered evidence, so I rebuilt it from collected files.",
                    observation="Final answer repaired without storing the rejected draft text.",
                    safety_note="Rejected draft text is not exposed in events.",
                )
        if on_delta and accepted:
            for chunk in _chunk_text(accepted):
                on_delta(chunk)
        return accepted
    except Exception:
        accepted = synthesize_answer_from_evidence(request, state, file_contents, repo_observation)
        if on_delta and accepted:
            for chunk in _chunk_text(accepted):
                on_delta(chunk)
        return accepted


def _quality_guard_answer(answer: str, *, file_contents: dict[str, str]) -> str | None:
    if not answer:
        return None
    lowered = answer.lower()
    if file_contents and any(bad in lowered for bad in ("cannot read", "can't read", "files object is empty", "no file contents")):
        return None
    if re.search(r"[�]{2,}|[A-Za-z]{2,}[가-힣]{2,}[A-Za-z]{2,}", answer):
        return None
    return answer


def validate_or_repair_final_answer(answer: str, state: AgentCoreState, request: AgentRunRequest) -> str:
    text = answer or ""
    lowered = text.lower()
    file_contents = collect_file_contents(state)
    edit_proposal = _latest_edit_proposal(state)
    if _needs_general_final_answer_repair(text, state):
        return repair_final_answer(text, state, request, file_contents)
    if edit_proposal and not edit_proposal.get("applied"):
        proposal_only = "proposed patch only" in lowered or "no files were modified" in lowered
        false_write = any(
            phrase in lowered
            for phrase in ("applied", "modified the file", "i modified", "changed the file", "wrote", "saved", "수정했습니다", "적용했습니다")
        )
        if false_write and not proposal_only:
            return _format_edit_proposal(edit_proposal)
    if state.files_read and any(
        phrase in lowered
        for phrase in (
            "cannot read",
            "can't read",
            "files object is empty",
            "repository structure only",
            "only see repository structure",
            "파일을 읽을 수 없습니다",
        )
    ):
        return synthesize_answer_from_evidence(request, state, file_contents, "\n".join(state.observations[-6:]))
    if re.search(r"[�]{2,}|[A-Za-z]{2,}[가-힣]{2,}[A-Za-z]{2,}|[\u0400-\u04ff\u0600-\u06ff]{8,}", text):
        return synthesize_answer_from_evidence(request, state, file_contents, "\n".join(state.observations[-6:]))
    if _looks_like_project_summary_request(request) and _looks_like_file_dump(text):
        return synthesize_answer_from_evidence(request, state, file_contents, "\n".join(state.observations[-6:]), force_mode="project_summary")
    if _is_placeholder_answer(text):
        return synthesize_answer_from_evidence(request, state, file_contents, "\n".join(state.observations[-6:]))
    if not state.files_read and not state.commands_run and len(state.observations) <= 1 and _makes_repo_claim(text):
        return synthesize_answer_from_evidence(request, state, file_contents, "\n".join(state.observations[-6:]))
    return text


def _needs_general_final_answer_repair(answer: str, state: AgentCoreState) -> bool:
    lowered = (answer or "").lower()
    if not answer.strip():
        return True
    internal_markers = (
        "the user asks",
        "the user is asking",
        "i need to",
        "need clarification",
        "i should ask",
        "we need to",
        "private " + "reasoning",
    )
    if any(marker in lowered for marker in internal_markers):
        return True
    raw_stop_markers = ("max_loop_iterations", "max_file_reads", "max_commands", "timed_out")
    if any(marker in lowered for marker in raw_stop_markers):
        return True
    if _looks_like_malformed_mixed_language(answer):
        return True
    if _looks_like_raw_metadata_dump(answer):
        return True
    if "technical log" in lowered or "work log" in lowered:
        return True
    return False


def repair_final_answer(answer: str, state: AgentCoreState, request: AgentRunRequest, file_contents: dict[str, str]) -> str:
    repo_observation = "\n".join(state.observations[-6:])
    try:
        raw = _compat_model_client()().generate_text(
            ModelGenerationRequest(
                system_prompt=(
                    "Repair the assistant final answer. Return only a clean user-facing answer in the same language as the user. "
                    "Ground it in the supplied evidence. Do not include internal planning, raw loop limits, technical logs, work logs, "
                    "or claims that files were modified unless files_changed is non-empty."
                ),
                user_prompt=json.dumps(
                    {
                        "task": request.task,
                        "draft_answer": answer[:4000],
                        "files_read": state.files_read,
                        "files_changed": state.files_changed,
                        "observations": state.observations[-8:],
                        "evidence_files": {path: content[:20_000] for path, content in file_contents.items()},
                    },
                    ensure_ascii=False,
                ),
            )
        )
        _reasoning, visible = split_visible_reasoning(raw)
        cleaned, _ = clean_user_visible_response(visible, user_task=request.task)
        if cleaned.strip() and not _needs_general_final_answer_repair(cleaned, state):
            return cleaned.strip()
    except Exception:
        pass
    return synthesize_answer_from_evidence(request, state, file_contents, repo_observation)


def _looks_like_malformed_mixed_language(answer: str) -> bool:
    return bool(re.search(r"[�]{2,}|[A-Za-z]{2,}[가-힣]{2,}[A-Za-z]{2,}|[\u0400-\u04ff\u0600-\u06ff]{8,}", answer or ""))


def _looks_like_raw_metadata_dump(answer: str) -> bool:
    text = answer or ""
    lowered = text.lower()
    if any(marker in lowered for marker in ("context_pack_report", "file_evidence", "short_term_memory", "included_files", "omitted_files")):
        return True
    if lowered.count('"scripts"') + lowered.count('"dependencies"') + lowered.count('"devdependencies"') >= 2:
        return True
    if len(re.findall(r"https?://img\.shields\.io|badge\.svg|!\[[^\]]*\]\(", text, flags=re.IGNORECASE)) >= 2:
        return True
    if text.count('",') > 20 and any(marker in lowered for marker in ("package.json", '"name"', '"version"')):
        return True
    return False


def _is_placeholder_answer(answer: str) -> bool:
    lowered = answer.lower()
    placeholder_fragments = (
        ("i inspected the gathered", "project evidence"),
        ("i can give", "grounded summary"),
        ("should be", "synthesized"),
        ("model answer", "needed repair"),
        ("ask for a narrower", "change"),
        ("i can answer", "from those files"),
        ("keeping this", "grounded summary"),
    )
    return any(
        all(fragment in lowered for fragment in fragments)
        for fragments in placeholder_fragments
    )


def _looks_like_project_summary_request(request: AgentRunRequest) -> bool:
    task_text = getattr(request, "task", "")
    lowered = task_text.lower()
    if any(term in task_text for term in ("아키텍처", "구조", "실행 흐름")) or any(term in lowered for term in ("architecture", "execution flow", "entrypoint")):
        return False
    return "project" in lowered or "프로젝트" in task_text


def _looks_like_file_dump(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    file_like_lines = [line for line in lines if re.search(r"\b[\w./-]+\.(py|cs|js|ts|tsx|md|json|toml)\b", line)]
    reviewed_lines = [line for line in lines if line.lower().startswith(("reviewed ", "- reviewed", "* reviewed"))]
    return len(file_like_lines) >= 6 or len(reviewed_lines) >= 4


def _makes_repo_claim(answer: str) -> bool:
    lowered = answer.lower()
    return any(term in lowered for term in ("this repository", "this project", "codebase", "프로젝트", "저장소"))


def collect_file_contents(state: AgentCoreState) -> dict[str, str]:
    contents: dict[str, str] = {}
    for result in state.action_results:
        for path, content in (result.payload.get("contents") or {}).items():
            if isinstance(path, str) and isinstance(content, str):
                contents[path] = content[:100_000]
    return contents


def synthesize_answer_from_evidence(
    request: AgentRunRequest,
    state: AgentCoreState | None,
    file_contents: dict[str, str],
    repo_observation: str,
    *,
    force_mode: str | None = None,
) -> str:
    language = "ko" if re.search(r"[가-힣]", request.task or "") else "en"
    useful_contents = {path: text for path, text in file_contents.items() if text.strip()}
    if not useful_contents:
        if state and state.files_read:
            files = ", ".join(f"`{path}`" for path in state.files_read)
            return (
                f"읽은 파일 기록은 있습니다: {files}. 다만 현재 응답 복구에 사용할 파일 본문이 남아 있지 않아 세부 내용은 단정하지 않겠습니다."
                if language == "ko"
                else f"Recorded evidence files: {files}. The file bodies are not available for deterministic repair, so I will not invent details."
            )
        if state and state.commands_run:
            return "실행한 명령: " + ", ".join(state.commands_run) if language == "ko" else "Commands run: " + ", ".join(state.commands_run)
        return (
            "아직 답변에 필요한 파일 증거가 충분하지 않습니다. 확인할 파일이나 범위를 지정해 주세요."
            if language == "ko"
            else "I do not have enough file evidence yet. Please specify the file or scope to inspect."
        )
    mode = force_mode or infer_answer_mode(request, useful_contents)
    signals = summarize_evidence_signals(useful_contents)
    evidence = ", ".join(f"`{path}`" for path in useful_contents)
    if language == "ko":
        if mode == "execution_flow":
            return korean_execution_flow_answer(signals, evidence)
        if mode == "architecture":
            return korean_architecture_answer(signals, evidence)
        if mode == "project_summary":
            return korean_project_summary_answer(signals, evidence)
        return korean_generic_answer(signals, evidence)
    if mode == "execution_flow":
        return english_execution_flow_answer(signals, evidence)
    if mode == "architecture":
        return english_architecture_answer(signals, evidence)
    if mode == "project_summary":
        return english_project_summary_answer(signals, evidence)
    return english_generic_answer(signals, evidence)


def infer_answer_mode(request: AgentRunRequest, file_contents: dict[str, str]) -> str:
    task = request.task or ""
    lowered = task.lower()
    if any(term in task for term in ("실행 흐름", "실행", "흐름")) or any(term in lowered for term in ("execution flow", "flow", "entrypoint", "how it starts")):
        return "execution_flow"
    if any(term in task for term in ("아키텍처", "구조")) or any(term in lowered for term in ("architecture", "modules", "structure")):
        return "architecture"
    if _looks_like_project_summary_request(request):
        return "project_summary"
    if any(Path(path).name.lower() in {"main.py", "index.ts", "index.js", "app.tsx", "cli.py"} for path in file_contents):
        return "execution_flow"
    return "generic"


def summarize_evidence_signals(file_contents: dict[str, str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path, text in file_contents.items():
        files.append(
            {
                "path": path,
                "basename": Path(path).name,
                "summary": summarize_file_signal(path, text),
                "headings": extract_markdown_headings(text),
                "paragraph": extract_first_paragraph(text),
                "imports": extract_imports(text),
                "functions": extract_functions(text),
                "classes": extract_classes(text),
                "entrypoint": bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]|def\s+main\s*\(", text)),
            }
        )
    project_terms = " ".join(item["paragraph"] for item in files if item["paragraph"])[:800]
    return {"files": files, "project_terms": project_terms}


def summarize_file_signal(path: str, text: str) -> str:
    name = Path(path).name
    if name.lower().startswith("readme"):
        paragraph = extract_first_paragraph(text)
        return paragraph or "Project documentation."
    imports = extract_imports(text)
    functions = extract_functions(text)
    classes = extract_classes(text)
    parts: list[str] = []
    if imports:
        parts.append("imports " + ", ".join(imports[:5]))
    if classes:
        parts.append("defines classes " + ", ".join(classes[:5]))
    if functions:
        parts.append("defines functions " + ", ".join(functions[:6]))
    if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", text):
        parts.append("contains the Python script entrypoint")
    return "; ".join(parts) or "Source/config evidence."


def extract_markdown_headings(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)][:6]


def extract_first_paragraph(text: str) -> str:
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for block in re.split(r"\n\s*\n", cleaned):
        block = " ".join(line.strip("# ").strip() for line in block.splitlines()).strip()
        if len(block) > 20 and not block.startswith(("import ", "from ")):
            return block[:500]
    return ""


def extract_imports(text: str) -> list[str]:
    imports: list[str] = []
    for match in re.finditer(r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", text, flags=re.MULTILINE):
        value = match.group(1) or match.group(2)
        if value and value not in imports:
            imports.append(value)
    return imports[:12]


def extract_functions(text: str) -> list[str]:
    names = re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.MULTILINE)
    names.extend(re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    return _dedupe(names)[:12]


def extract_classes(text: str) -> list[str]:
    names = re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", text, flags=re.MULTILINE)
    return _dedupe(names)[:12]


def project_conclusion(signals: dict[str, Any]) -> str:
    if any(path["basename"] == "package.json" for path in signals["files"]):
        return "JavaScript/TypeScript project"
    if any(path["basename"] in {"pyproject.toml", "main.py"} or path["basename"].endswith(".py") for path in signals["files"]):
        return "Python project"
    if any(path["basename"].endswith(".cs") for path in signals["files"]):
        return "C# project"
    if any(path["basename"] == "go.mod" or path["basename"].endswith(".go") for path in signals["files"]):
        return "Go project"
    if any(path["basename"] == "Cargo.toml" or path["basename"].endswith(".rs") for path in signals["files"]):
        return "Rust project"
    return "software project"


def key_file_lines(signals: dict[str, Any], *, limit: int = 6) -> list[str]:
    return [f"- `{item['path']}`: {item['summary']}" for item in signals["files"][:limit]]


def korean_project_summary_answer(signals: dict[str, Any], evidence: str) -> str:
    conclusion = project_conclusion(signals)
    lines = [
        f"이 레포는 {conclusion}로 보입니다.",
        "",
        f"근거 파일: {evidence}",
        "목적: " + (signals.get("project_terms") or "읽은 파일 기준으로 프로젝트 목적과 주요 구성 요소를 파악했습니다."),
        "주요 파일:",
        *key_file_lines(signals),
        "제한: 읽은 파일에 근거한 요약이라, 실행 결과나 전체 파일을 모두 검증한 것은 아닙니다.",
    ]
    return "\n".join(lines)


def korean_execution_flow_answer(signals: dict[str, Any], evidence: str) -> str:
    entry = next((item for item in signals["files"] if item["entrypoint"] or item["basename"] in {"main.py", "index.js", "index.ts", "cli.py"}), signals["files"][0])
    steps = [
        f"1. `{entry['path']}`가 실행 시작점 역할을 합니다.",
        f"2. 이 파일은 {entry['summary']} 흐름을 구성합니다.",
    ]
    if entry["imports"]:
        steps.append(f"3. 주요 모듈로 {', '.join(entry['imports'][:6])}을 불러와 시뮬레이션/처리 로직을 위임합니다.")
    if entry["functions"] or entry["classes"]:
        steps.append(f"4. 핵심 함수/클래스: {', '.join([*entry['functions'][:4], *entry['classes'][:4]])}.")
    return "\n".join([f"근거 파일: {evidence}", "", "실행 흐름:", *steps, "제한: 정적 파일 내용 기준 설명입니다."])


def korean_architecture_answer(signals: dict[str, Any], evidence: str) -> str:
    return "\n".join([
        "아키텍처 요약:",
        f"- 근거 파일: {evidence}",
        f"- 상위 목적: {project_conclusion(signals)}",
        "- 모듈 책임:",
        *key_file_lines(signals),
        "- 흐름: entrypoint가 설정/입력을 준비하고, 도메인 모듈의 클래스와 함수가 계산/상태 처리를 담당하는 구조로 보입니다.",
    ])


def korean_generic_answer(signals: dict[str, Any], evidence: str) -> str:
    return "\n".join([f"근거 파일: {evidence}", "읽은 내용 요약:", *key_file_lines(signals)])


def english_project_summary_answer(signals: dict[str, Any], evidence: str) -> str:
    return "\n".join([
        f"This repository appears to be a {project_conclusion(signals)}.",
        "",
        f"Evidence used: {evidence}",
        "Purpose: " + (signals.get("project_terms") or "The read files identify the project purpose and main components."),
        "Key files:",
        *key_file_lines(signals),
        "Limitation: this is based on the files read, not a full runtime verification.",
    ])


def english_execution_flow_answer(signals: dict[str, Any], evidence: str) -> str:
    entry = next((item for item in signals["files"] if item["entrypoint"] or item["basename"] in {"main.py", "index.js", "index.ts", "cli.py"}), signals["files"][0])
    steps = [f"1. `{entry['path']}` is the start/entrypoint evidence.", f"2. It {entry['summary']}."]
    if entry["imports"]:
        steps.append(f"3. It delegates to modules such as {', '.join(entry['imports'][:6])}.")
    if entry["functions"] or entry["classes"]:
        steps.append(f"4. Key functions/classes visible there: {', '.join([*entry['functions'][:4], *entry['classes'][:4]])}.")
    return "\n".join([f"Evidence used: {evidence}", "", "Execution flow:", *steps, "Limitation: static file evidence only."])


def english_architecture_answer(signals: dict[str, Any], evidence: str) -> str:
    return "\n".join(["Architecture summary:", f"- Evidence used: {evidence}", f"- Purpose: {project_conclusion(signals)}", "- Module responsibilities:", *key_file_lines(signals)])


def english_generic_answer(signals: dict[str, Any], evidence: str) -> str:
    return "\n".join([f"Evidence used: {evidence}", "Summary:", *key_file_lines(signals)])


def _latest_edit_proposal(state: AgentCoreState) -> dict[str, Any] | None:
    for result in reversed(state.action_results):
        proposals = result.payload.get("edit_proposals") or []
        if proposals:
            return {"applied": bool(result.payload.get("applied")), "proposals": proposals}
    return None


def _format_edit_proposal(payload: dict[str, Any]) -> str:
    proposals = [item for item in payload.get("proposals") or [] if isinstance(item, dict)]
    if not proposals:
        return "I prepared no file changes because there was not enough safe evidence to build a minimal patch."
    sections = ["I prepared a proposed patch only. No files were modified in this run."]
    for item in proposals[:3]:
        file_path = str(item.get("file") or "unknown file")
        before = str(item.get("before_summary") or "before state recorded")
        after = str(item.get("after_summary") or "after state recorded")
        diff = str(item.get("diff_summary") or "").strip()
        notes = [str(note) for note in item.get("risk_notes") or [] if str(note)]
        notes_text = ("\nRisk notes: " + "; ".join(notes)) if notes else ""
        sections.append(
            f"\n`{file_path}`\nBefore: {before}\nAfter: {after}{notes_text}\n\n```diff\n{diff[:3000]}\n```"
        )
    return "\n".join(sections)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _chunk_text(text: str, chunk_size: int = 96):
    for start in range(0, len(text or ""), chunk_size):
        chunk = text[start : start + chunk_size]
        if chunk:
            yield chunk


def _compat_model_client():
    try:
        from repooperator_worker.agent_core.graph import support

        return getattr(support, "OpenAICompatibleModelClient", _OpenAICompatibleModelClient)
    except Exception:
        return _OpenAICompatibleModelClient
