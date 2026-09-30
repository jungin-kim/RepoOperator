#!/usr/bin/env python3
"""Live-model eval of the agent loop against a throwaway fixture repository.

Runs a fixed set of scenarios through the real LangGraph runtime with the model
configured in ~/.repooperator/config.json, and reports for each run whether the
outcome matched the request type, how often the policy checks fired, whether
the model fixed itself after feedback, and token / prompt-cache usage.

Nothing touches your real repositories: every scenario runs on a fresh copy of
a tiny fixture repo in a temp dir, and edits/commands stop at their approval
cards (they are never applied or executed).

    cd apps/local-worker && .venv/bin/python ../../scripts/eval_agent_live.py \
        --out ../../.eval-out [--repeat 2] [--only read_only_question]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("REPOOPERATOR_AGENTIC_TOOL_CALLING", "1")

FIXTURE = {
    "README.md": "# calc\n\nA tiny calculator used by the RepoOperator live eval.\n\nRun `python calc.py` to print a few results.\n",
    "calc.py": (
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def subtract(a, b):\n"
        "    return a + b  # bug: should subtract\n\n\n"
        "def divide(a, b):\n"
        "    return a / b\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    print(add(2, 3), subtract(5, 2), divide(8, 2))\n"
    ),
    "test_calc.py": (
        "from calc import add, subtract\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_subtract():\n    assert subtract(5, 2) == 3\n"
    ),
}

# expect: "answer" (no proposal, no command card), "proposal" (change set awaiting
# approval), "command" (command approval card).
SCENARIOS = [
    {"id": "project_summary", "task": "이 프로젝트가 뭐 하는 건지 설명해줘", "expect": "answer"},
    {"id": "read_only_question", "task": "add 함수는 뭘 반환해?", "expect": "answer"},
    {"id": "find_bugs", "task": "calc.py에서 버그 찾아줘", "expect": "answer"},
    {"id": "plan_only", "task": "곱셈 기능을 추가하려면 어떤 작업이 필요한지 계획 세워줘", "expect": "answer"},
    {"id": "add_docstring", "task": "add 함수에 docstring 추가해줘", "expect": "proposal"},
    {"id": "fix_bug", "task": "subtract 함수 버그 고쳐줘", "expect": "proposal"},
    {"id": "run_command", "task": "python calc.py 실행해줘", "expect": "command"},
]


def _make_repo(root: Path) -> Path:
    repo = root / "calc-fixture"
    repo.mkdir(parents=True)
    for name, text in FIXTURE.items():
        (repo / name).write_text(text, encoding="utf-8")
    os.system(f"cd '{repo}' && git init -q && git add -A && git -c user.name=eval -c user.email=eval@local commit -qm init")
    return repo


def _outcome(response) -> str:
    if getattr(response, "change_set_proposal", None) or getattr(response, "proposal_id", None):
        return "proposal"
    if getattr(response, "command_approval", None) or response.response_type == "command_approval":
        return "command"
    return "answer"


def run(out: Path, repeat: int, only: set[str]) -> int:
    from repooperator_worker.agent_core.agentic_loop import tool_calling_available
    from repooperator_worker.agent_core.graph import runtime
    from repooperator_worker.config import get_settings
    from repooperator_worker.schemas import AgentRunRequest

    settings = get_settings()
    header = {
        "model": settings.openai_model,
        "provider": settings.configured_model_provider,
        "base_url": settings.openai_base_url,
        "tool_calling": tool_calling_available(settings),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(header, ensure_ascii=False))
    if not header["tool_calling"]:
        print("!! tool calling is not available for this model/config; the model-driven loop will not run.")

    # The eval repo is never the "active" repository of the UI; skip that check.
    runtime.validate_active_repository = lambda request: None

    out.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="ro-eval-") as tmp:
        for scenario in SCENARIOS:
            if only and scenario["id"] not in only:
                continue
            for n in range(repeat):
                repo = _make_repo(Path(tmp) / f"{scenario['id']}-{n}")
                run_id = f"eval-{scenario['id']}-{n}-{uuid.uuid4().hex[:6]}"
                started = time.time()
                row = {"scenario": scenario["id"], "repeat": n, "expect": scenario["expect"], "run_id": run_id}
                try:
                    request = AgentRunRequest(project_path=str(repo), task=scenario["task"], thread_id=run_id)
                    response = runtime.run_langgraph_controller(request, run_id=run_id)
                    usage = response.model_usage or {}
                    row.update(
                        outcome=_outcome(response),
                        stop_reason=response.stop_reason,
                        files_read=response.files_read,
                        answer=(response.response or "")[:600],
                        usage=usage,
                        gate_codes={k[5:]: v for k, v in usage.items() if k.startswith("gate:")},
                    )
                except Exception as exc:  # keep going; one broken scenario should not end the eval
                    row.update(outcome="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
                row["seconds"] = round(time.time() - started, 1)
                row["pass"] = row["outcome"] == scenario["expect"]
                results.append(row)
                print(json.dumps({k: row.get(k) for k in ("scenario", "outcome", "expect", "pass", "seconds", "gate_codes")}, ensure_ascii=False), flush=True)
                (out / "results.json").write_text(json.dumps({"header": header, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
                shutil.rmtree(repo, ignore_errors=True)

    _write_summary(out, header, results)
    return 0


def _write_summary(out: Path, header: dict, results: list[dict]) -> None:
    def total(key: str) -> int:
        return sum(int((r.get("usage") or {}).get(key, 0)) for r in results)

    lines = [f"# Live agent eval — {header['model']} ({header['started']})", ""]
    lines.append("| scenario | expect | outcome | pass | s | calls | cache | gate hits | recovered | fallback |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        u = r.get("usage") or {}
        gates = ", ".join(f"{k}×{v}" for k, v in (r.get("gate_codes") or {}).items()) or "-"
        lines.append(
            f"| {r['scenario']} | {r['expect']} | {r['outcome']} | {'✅' if r['pass'] else '❌'} | {r['seconds']} | "
            f"{u.get('calls', 0)} | {round(100 * u.get('cache_hit_ratio', 0))}% | {gates} | "
            f"{u.get('gate_feedback_recovered', 0)} | {u.get('gate_fallbacks', 0)} |"
        )
    hits = sum(sum((r.get("gate_codes") or {}).values()) for r in results)
    lines += [
        "",
        f"- pass: {sum(r['pass'] for r in results)}/{len(results)}",
        f"- policy check hits: {hits}; recovered by feedback: {total('gate_feedback_recovered')}; fell back: {total('gate_fallbacks')}",
        f"- prompt tokens: {total('input_tokens')}, cached: {total('cached_input_tokens')}, output: {total('output_tokens')}",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".eval-out")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    sys.exit(run(Path(args.out), args.repeat, set(args.only)))
