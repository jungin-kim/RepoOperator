import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.services.provider_service import (
    _build_github_api_base,
    _build_gitlab_api_base,
    list_recent_project_paths,
    list_recent_projects,
    record_recent_project,
    list_provider_projects,
)


class ProviderServiceTests(unittest.TestCase):
    def test_build_gitlab_api_base(self) -> None:
        self.assertEqual(
            _build_gitlab_api_base("https://gitlab.example.com/"),
            "https://gitlab.example.com/api/v4",
        )

    def test_build_github_api_base_for_public_github(self) -> None:
        self.assertEqual(
            _build_github_api_base("https://github.com"),
            "https://api.github.com",
        )

    def test_build_github_api_base_for_github_enterprise(self) -> None:
        self.assertEqual(
            _build_github_api_base("https://github.example.com"),
            "https://github.example.com/api/v3",
        )

    def test_list_recent_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            config_dir = Path(temp_home) / ".repooperator"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            config_path.write_text('{"gitProvider":{"provider":"gitlab"}}', encoding="utf-8")

            with patch.dict(
                os.environ,
                {"REPOOPERATOR_CONFIG_PATH": str(config_path)},
                clear=False,
            ):
                record_recent_project(
                    project_path="group-one/repo-one",
                    git_provider="gitlab",
                    display_name="repo-one",
                )
                record_recent_project(
                    project_path="group-two/repo-two",
                    git_provider="gitlab",
                    display_name="repo-two",
                )
                recent = list_recent_project_paths(limit=10)

            self.assertIn("group-one/repo-one", recent)
            self.assertIn("group-two/repo-two", recent)

    def test_recent_projects_are_cross_source_history_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            config_dir = Path(temp_home) / ".repooperator"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            config_path.write_text('{"gitProvider":{"provider":"github"}}', encoding="utf-8")
            local_project = Path(temp_home) / "work" / "local-demo"
            local_project.mkdir(parents=True, exist_ok=True)

            with patch.dict(
                os.environ,
                {"REPOOPERATOR_CONFIG_PATH": str(config_path)},
                clear=False,
            ):
                record_recent_project(
                    project_path="group/gitlab-demo",
                    git_provider="gitlab",
                    display_name="gitlab-demo",
                )
                record_recent_project(
                    project_path="owner/github-demo",
                    git_provider="github",
                    display_name="github-demo",
                )
                record_recent_project(
                    project_path=str(local_project),
                    git_provider="local",
                    display_name="local-demo",
                    is_git_repo=False,
                )
                recent = list_recent_projects(limit=10)

            projects_by_source = {
                (project.git_provider, project.project_path)
                for project in recent
            }
            self.assertIn(("gitlab", "group/gitlab-demo"), projects_by_source)
            self.assertIn(("github", "owner/github-demo"), projects_by_source)
            self.assertIn(("local", str(local_project)), projects_by_source)

    def test_local_recent_projects_are_returned_from_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            config_dir = Path(temp_home) / ".repooperator"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            config_path.write_text('{"gitProvider":{"provider":"local"}}', encoding="utf-8")

            local_project = Path(temp_home) / "work" / "demo-project"
            local_project.mkdir(parents=True, exist_ok=True)

            with patch.dict(
                os.environ,
                {"REPOOPERATOR_CONFIG_PATH": str(config_path)},
                clear=False,
            ):
                record_recent_project(
                    project_path=str(local_project),
                    git_provider="local",
                    display_name="demo-project",
                    is_git_repo=False,
                )
                payload = list_provider_projects("local", search=str(local_project))

            self.assertTrue(payload.projects)
            self.assertEqual(payload.projects[0].git_provider, "local")
            self.assertEqual(payload.projects[0].project_path, str(local_project.resolve()))
            self.assertFalse(payload.projects[0].is_git_repository)

    def test_explicit_provider_listing_ignores_different_default_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            config_dir = Path(temp_home) / ".repooperator"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            config_path.write_text(
                """
                {
                  "gitProvider": {
                    "provider": "github",
                    "baseUrl": "https://github.example.com",
                    "token": "github-default-token"
                  },
                  "repositorySources": [
                    {
                      "provider": "github",
                      "baseUrl": "https://github.example.com",
                      "token": "github-source-token"
                    },
                    {
                      "provider": "gitlab",
                      "baseUrl": "https://gitlab.example.com",
                      "token": "gitlab-source-token"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            requested_urls: list[str] = []

            def fake_provider_json(url: str, headers: dict[str, str], provider: str):
                requested_urls.append(url)
                if provider == "gitlab":
                    self.assertEqual(headers["PRIVATE-TOKEN"], "gitlab-source-token")
                    return [
                        {
                            "path_with_namespace": "group/gitlab-demo",
                            "name_with_namespace": "Group / GitLab Demo",
                            "default_branch": "main",
                        }
                    ]
                if provider == "github":
                    self.assertEqual(headers["Authorization"], "Bearer github-source-token")
                    return [
                        {
                            "full_name": "owner/github-demo",
                            "name": "github-demo",
                            "default_branch": "main",
                        }
                    ]
                raise AssertionError(f"Unexpected provider: {provider}")

            with patch.dict(
                os.environ,
                {
                    "HOME": temp_home,
                    "REPOOPERATOR_CONFIG_PATH": str(config_path),
                    "GITHUB_BASE_URL": "",
                    "GITHUB_TOKEN": "",
                    "GITLAB_BASE_URL": "",
                    "GITLAB_TOKEN": "",
                },
                clear=False,
            ), patch(
                "repooperator_worker.services.provider_service._request_provider_json",
                side_effect=fake_provider_json,
            ):
                gitlab_payload = list_provider_projects("gitlab")
                github_payload = list_provider_projects("github")

            self.assertEqual(gitlab_payload.git_provider, "gitlab")
            self.assertEqual(gitlab_payload.projects[0].project_path, "group/gitlab-demo")
            self.assertEqual(github_payload.git_provider, "github")
            self.assertEqual(github_payload.projects[0].project_path, "owner/github-demo")
            self.assertTrue(
                any(url.startswith("https://gitlab.example.com/api/v4/projects?") for url in requested_urls)
            )
            self.assertTrue(
                any(url.startswith("https://github.example.com/api/v3/user/repos?") for url in requested_urls)
            )


if __name__ == "__main__":
    unittest.main()
