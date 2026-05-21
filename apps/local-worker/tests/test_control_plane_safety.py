import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse, FileWriteRequest  # noqa: E402
from repooperator_worker.schemas.requests import CommandRunRequest, GitCommitRequest  # noqa: E402
from repooperator_worker.services.agent_run_coordinator import (  # noqa: E402
    cancel_queued_message,
    cancel_run,
    consume_steering,
    enqueue_message,
    list_queue,
    start_run,
    steer_run,
    stream_run,
)
from repooperator_worker.services.event_service import append_run_event, complete_active_run, get_active_runs, get_run, list_run_events, start_active_run  # noqa: E402
from repooperator_worker.services.file_service import write_text_file  # noqa: E402
from repooperator_worker.services.command_runner import run_command  # noqa: E402
from repooperator_worker.services.command_service import preview_command  # noqa: E402
from repooperator_worker.services.git_service import commit_changes  # noqa: E402
from repooperator_worker.services.model_client import split_visible_reasoning  # noqa: E402


def _write_config(home: Path, mode: str = "basic") -> Path:
    config = home / ".repooperator" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "permissions": {"mode": mode},
                "model": {
                    "connectionMode": "local-runtime",
                    "provider": "vllm",
                    "baseUrl": "http://127.0.0.1:8001/v1",
                    "model": "test-model",
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Demo\n", encoding="utf-8")


class ControlPlaneSafetyTests(unittest.TestCase):
    def test_cmd_run_uses_command_policy_for_mutating_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                with self.assertRaises(PermissionError) as raised:
                    run_command(CommandRunRequest(project_path=str(repo), command="git commit -m change"))
        self.assertIn("requires approval", str(raised.exception).lower())

    def test_cmd_run_allows_read_only_command_through_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                response = run_command(CommandRunRequest(project_path=str(repo), command="git status --short"))
        self.assertEqual(response.exit_code, 0)

    def test_git_branch_creation_is_not_classified_as_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                preview = preview_command(["git", "branch", "feature/demo"], project_path=str(repo))
                read_only_preview = preview_command(["git", "branch", "--show-current"], project_path=str(repo))
        self.assertTrue(preview["needs_approval"])
        self.assertFalse(preview["read_only"])
        self.assertFalse(read_only_preview["needs_approval"])
        self.assertTrue(read_only_preview["read_only"])

    def test_git_commit_route_requires_backend_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            (repo / "README.md").write_text("# Changed\n", encoding="utf-8")
            config = _write_config(home)
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                with self.assertRaises(PermissionError):
                    commit_changes(GitCommitRequest(project_path=str(repo), message="Update docs", stage_all=True))

    def test_basic_permission_allows_safe_repo_write_and_blocks_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home, mode="basic")
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                result = write_text_file(
                    FileWriteRequest(
                        project_path=str(repo),
                        relative_path="README.md",
                        content="# Updated\n",
                    )
                )
                self.assertEqual(result.bytes_written, len("# Updated\n".encode("utf-8")))
                with self.assertRaises(ValueError):
                    write_text_file(
                        FileWriteRequest(
                            project_path=str(repo),
                            relative_path=".git/config",
                            content="[core]\n",
                        )
                    )

    def test_coordinator_records_run_events_and_thread_scoped_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(
                project_path=str(repo),
                git_provider="local",
                branch="main",
                thread_id="thread-a",
                task="Summarize repository health.",
            )
            fake_response = AgentRunResponse(
                project_path=str(repo),
                git_provider="local",
                active_repository_source="local",
                active_repository_path=str(repo),
                active_branch="main",
                task=request.task,
                model="test-model",
                branch="main",
                repo_root_name="repo",
                context_summary="",
                top_level_entries=[],
                readme_included=False,
                diff_included=False,
                is_git_repository=True,
                files_read=["README.md"],
                response="Repository looks small and readable.",
                intent_classification="repo_analysis",
                graph_path="test",
            )
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True), patch(
                "repooperator_worker.services.agent_service.run_agent_task",
                return_value=fake_response,
            ):
                response = start_run(request)
                queue_item = enqueue_message("thread-a", str(repo), "main", "Continue with docs.")
                self.assertEqual(len(list_queue(thread_id="thread-a", repo=str(repo), branch="main")), 1)
                cancelled = cancel_queued_message(queue_item["id"])
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertIsNotNone(response.run_id)
                self.assertEqual(get_run(response.run_id)["status"], "completed")
                self.assertTrue(list_run_events(response.run_id))

    def test_steering_and_cancel_are_coordinator_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(project_path=str(repo), thread_id="thread-b", task="Analyze files.")
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                from repooperator_worker.services.event_service import new_run_id

                run_id = new_run_id()
                start_active_run(run_id=run_id, request=request, thread_id="thread-b")
                append_run_event(
                    run_id,
                    {
                        "type": "progress_delta",
                        "phase": "Reading",
                        "label": "Reading files",
                        "status": "running",
                        "started_at": "2026-05-06T00:00:00Z",
                    },
                )
                steering = steer_run(run_id, content="Prefer a narrower pass.")
                self.assertEqual(steering["status"], "recorded")
                self.assertIsNone(steering["steering"].get("accepted"))
                self.assertEqual(steering["steering"].get("parse_status"), "pending")
                self.assertTrue(consume_steering(run_id))
                cancelled = cancel_run(run_id)
                self.assertEqual(cancelled["status"], "cancelling")
                events = list_run_events(run_id)
                cancellation = [event for event in events if event.get("event_type") == "cancellation_requested"][0]
                self.assertEqual(cancellation["thread_id"], "thread-b")
                self.assertEqual(cancellation["repo"], str(repo))
                self.assertIn(run_id, {item["id"] for item in get_active_runs()})
                complete_active_run(run_id=run_id, status="cancelled", error="Cancelled by user.")
                self.assertNotIn(run_id, {item["id"] for item in get_active_runs()})

    def test_stream_run_returns_generator_object_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(project_path=str(repo), thread_id="thread-stream", task="Stream a summary.")

            def fake_stream(_request, *, run_id=None):
                yield json.dumps(
                    {
                        "type": "progress_delta",
                        "run_id": run_id,
                        "phase": "Thinking",
                        "label": "Working",
                        "status": "completed",
                    }
                )
                yield json.dumps(
                    {
                        "type": "final_message",
                        "run_id": run_id,
                        "result": AgentRunResponse(
                            project_path=str(repo),
                            task=request.task,
                            model="test-model",
                            repo_root_name="repo",
                            context_summary="",
                            top_level_entries=[],
                            readme_included=False,
                            diff_included=False,
                            is_git_repository=True,
                            response="Done.",
                        ).model_dump(),
                    }
                )

            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True), patch(
                "repooperator_worker.agent_core.controller_graph.stream_controller_graph",
                side_effect=fake_stream,
            ):
                run_id, stream = stream_run(request)
                self.assertTrue(hasattr(stream, "__iter__"))
                chunks = []
                for _ in range(10):
                    chunk = next(stream)
                    chunks.append(chunk)
                    if "[DONE]" in chunk:
                        break
                self.assertTrue(any("progress_delta" in chunk for chunk in chunks))
                self.assertTrue(chunks[-1].strip().endswith("[DONE]"))
                self.assertEqual(get_run(run_id)["status"], "completed")

    def test_stream_cancellation_stays_active_until_worker_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(project_path=str(repo), thread_id="thread-cancel-stream", task="Stream until cancelled.")

            def cancellable_stream(_request, *, run_id=None):
                yield {"type": "progress_delta", "run_id": run_id, "phase": "Thinking", "label": "Working", "status": "running"}
                deadline = time.time() + 2
                while time.time() < deadline and get_run(str(run_id)).get("status") != "cancelling":
                    time.sleep(0.01)
                response = AgentRunResponse(
                    project_path=str(repo),
                    task=request.task,
                    model="test-model",
                    repo_root_name="repo",
                    context_summary="",
                    top_level_entries=[],
                    readme_included=False,
                    diff_included=False,
                    is_git_repository=True,
                    response="Run cancelled.",
                    stop_reason="cancelled",
                )
                yield {"type": "final_message", "run_id": run_id, "result": response.model_dump(mode="json")}

            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True), patch(
                "repooperator_worker.agent_core.controller_graph.stream_controller_graph",
                side_effect=cancellable_stream,
            ):
                run_id, stream = stream_run(request)
                self.assertIn(run_id, {item["id"] for item in get_active_runs()})
                cancelling = cancel_run(run_id)
                self.assertEqual(cancelling["status"], "cancelling")
                self.assertIn(run_id, {item["id"] for item in get_active_runs()})
                for _ in range(20):
                    chunk = next(stream)
                    if "[DONE]" in chunk:
                        break
                self.assertEqual(get_run(run_id)["status"], "cancelled")
                self.assertNotIn(run_id, {item["id"] for item in get_active_runs()})

    def test_failed_stream_records_failed_run_and_error_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(project_path=str(repo), thread_id="thread-fail", task="Stream failure.")

            def failing_stream(_request, *, run_id=None):
                raise RuntimeError("simulated stream failure")
                yield "{}"

            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True), patch(
                "repooperator_worker.agent_core.controller_graph.stream_controller_graph",
                side_effect=failing_stream,
            ):
                run_id, stream = stream_run(request)
                chunks = []
                for _ in range(10):
                    chunk = next(stream)
                    chunks.append(chunk)
                    if "[DONE]" in chunk:
                        break
                self.assertEqual(get_run(run_id)["status"], "failed")
                self.assertTrue(any(event.get("type") == "error" for event in list_run_events(run_id)))

    def test_active_runs_exclude_terminal_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            home = Path(temp_home)
            repo = home / "repo"
            _init_repo(repo)
            config = _write_config(home)
            request = AgentRunRequest(project_path=str(repo), thread_id="thread-active", task="Check active runs.")
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
                from repooperator_worker.services.event_service import new_run_id

                run_id = new_run_id()
                start_active_run(run_id=run_id, request=request, thread_id="thread-active")
                self.assertIn(run_id, {item["id"] for item in get_active_runs()})
                complete_active_run(run_id=run_id, status="completed")
                self.assertNotIn(run_id, {item["id"] for item in get_active_runs()})

    def test_visible_reasoning_is_separated_from_final_answer(self) -> None:
        reasoning, answer = split_visible_reasoning("<think>check repository context</think>\nFinal answer")
        self.assertEqual(reasoning, "check repository context")
        self.assertEqual(answer, "Final answer")
        self.assertNotIn("<think>", answer)


if __name__ == "__main__":
    unittest.main()
