"""Phase 0: native tool-calling primitives and provider clients."""

from __future__ import annotations

import unittest

from repooperator_worker.agent_core.model_profile import detect_model_profile
from repooperator_worker.services import model_client
from repooperator_worker.services.model_tools import (
    ToolCall,
    ToolCallResponse,
    parse_anthropic_response,
    parse_openai_response,
    to_anthropic_tools,
    to_openai_tools,
)


SPECS = [
    {
        "name": "read_file",
        "description": "Read a repo file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {"name": "final_answer", "description": "Finish.", "input_schema": {}},
    {"name": "", "description": "skip me"},  # nameless spec is dropped
]


class SchemaConversionTests(unittest.TestCase):
    def test_to_openai_tools_shape(self) -> None:
        tools = to_openai_tools(SPECS)
        self.assertEqual(len(tools), 2)
        first = tools[0]
        self.assertEqual(first["type"], "function")
        self.assertEqual(first["function"]["name"], "read_file")
        self.assertEqual(first["function"]["parameters"]["required"], ["path"])
        # Empty schema is normalized to a valid object schema.
        self.assertEqual(tools[1]["function"]["parameters"], {"type": "object", "properties": {}})

    def test_to_anthropic_tools_shape(self) -> None:
        tools = to_anthropic_tools(SPECS)
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "read_file")
        self.assertIn("input_schema", tools[0])
        self.assertEqual(tools[1]["input_schema"], {"type": "object", "properties": {}})


class ResponseParsingTests(unittest.TestCase):
    def test_parse_openai_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                        ],
                    },
                }
            ]
        }
        response = parse_openai_response(payload)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0], ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}))
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_parse_openai_plain_text(self) -> None:
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]}
        response = parse_openai_response(payload)
        self.assertFalse(response.has_tool_calls)
        self.assertEqual(response.text, "hello")

    def test_parse_anthropic_tool_use(self) -> None:
        payload = {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "let me read it"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        }
        response = parse_anthropic_response(payload)
        self.assertEqual(response.text, "let me read it")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "a.py"})


class AnthropicMessageTranslationTests(unittest.TestCase):
    def test_tool_result_and_tool_use_translation(self) -> None:
        messages = [
            {"role": "user", "content": "read a.py"},
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [{"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file body"},
        ]
        translated = model_client._openai_messages_to_anthropic(messages)
        self.assertEqual(translated[0]["role"], "user")
        self.assertEqual(translated[1]["role"], "assistant")
        tool_use = translated[1]["content"][1]
        self.assertEqual(tool_use["type"], "tool_use")
        self.assertEqual(tool_use["input"], {"path": "a.py"})
        tool_result = translated[2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call_1")

    def test_adjacent_same_role_merged(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ]
        translated = model_client._openai_messages_to_anthropic(messages)
        self.assertEqual(len(translated), 1)
        self.assertEqual(len(translated[0]["content"]), 2)


class ProviderClientTests(unittest.TestCase):
    def test_openai_client_builds_tools_payload(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"id": "c1", "function": {"name": "final_answer", "arguments": "{}"}}]}}]}

        client = model_client.OpenAICompatibleModelClient()
        object.__setattr__(client, "_settings", _fake_settings(provider="openai-compatible", model="gpt-4o", base_url="http://x/v1", api_key="k"))
        original = model_client._post_json
        model_client._post_json = fake_post  # type: ignore[assignment]
        try:
            response = client.generate_with_tools(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=SPECS,
                tool_choice="auto",
            )
        finally:
            model_client._post_json = original  # type: ignore[assignment]
        payload = captured["payload"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(len(payload["tools"]), 2)
        self.assertEqual(response.tool_calls[0].name, "final_answer")

    def test_anthropic_client_builds_native_payload(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(**kwargs):
            captured.update(kwargs)
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]}

        client = model_client.AnthropicModelClient()
        object.__setattr__(client, "_settings", _fake_settings(provider="anthropic", model="claude-3.5", base_url="https://api.anthropic.com/v1", api_key="k"))
        original = model_client._post_json
        model_client._post_json = fake_post  # type: ignore[assignment]
        try:
            client.generate_with_tools(system_prompt="sys", messages=[{"role": "user", "content": "hi"}], tools=SPECS)
        finally:
            model_client._post_json = original  # type: ignore[assignment]
        headers = captured["headers"]
        self.assertIn("x-api-key", headers)
        self.assertIn("anthropic-version", headers)
        payload = captured["payload"]
        # System prompt and the last tool carry cache breakpoints so the
        # stable prefix (tools + system) is served from Anthropic's prompt cache.
        self.assertEqual(payload["system"], [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
        self.assertIn("max_tokens", payload)
        self.assertEqual(len(payload["tools"]), 2)
        self.assertNotIn("cache_control", payload["tools"][0])
        self.assertEqual(payload["tools"][-1]["cache_control"], {"type": "ephemeral"})

    def test_anthropic_message_breakpoint_skips_trailing_status(self) -> None:
        messages = model_client._with_message_cache_breakpoint(
            model_client._openai_messages_to_anthropic(
                [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "body"},
                    {"role": "user", "content": "[run status] {}"},
                ]
            )
        )
        last = messages[-1]["content"]
        # tool_result (stable) is the breakpoint; the per-step status after it is not cached.
        self.assertEqual(last[0]["type"], "tool_result")
        self.assertEqual(last[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", last[1])

    def test_usage_normalizes_cache_hits(self) -> None:
        openai = ToolCallResponse(raw={"usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 900}}})
        self.assertEqual(openai.usage, {"input_tokens": 1000, "cached_input_tokens": 900, "output_tokens": 20})
        anthropic = ToolCallResponse(raw={"usage": {"input_tokens": 50, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 100, "output_tokens": 20}})
        self.assertEqual(
            anthropic.usage,
            {"input_tokens": 1050, "cached_input_tokens": 900, "cache_write_tokens": 100, "output_tokens": 20},
        )
        self.assertEqual(ToolCallResponse(raw=None).usage, {})

    def test_build_model_client_selects_provider(self) -> None:
        self.assertIsInstance(
            model_client.build_model_client(_fake_settings(provider="anthropic", model="claude-3.5")),
            model_client.AnthropicModelClient,
        )
        self.assertIsInstance(
            model_client.build_model_client(_fake_settings(provider="ollama", model="llama3")),
            model_client.OpenAICompatibleModelClient,
        )
        # claude model name without provider still routes to Anthropic.
        self.assertEqual(model_client.resolve_model_provider(_fake_settings(provider="", model="claude-3-opus")), "anthropic")


class ModelProfileTests(unittest.TestCase):
    def test_tool_calls_enabled_for_modern_providers(self) -> None:
        for provider, model in [("openai", "gpt-4o"), ("anthropic", "claude-3.5"), ("ollama", "llama3")]:
            profile = detect_model_profile(provider=provider, model_name=model)
            self.assertTrue(profile.supports_tool_calls, f"{provider}/{model} should support tool calls")

    def test_metadata_override_wins(self) -> None:
        profile = detect_model_profile(provider="openai", model_name="gpt-4o", provider_metadata={"supports_tool_calls": False})
        self.assertFalse(profile.supports_tool_calls)


def _fake_settings(*, provider: str, model: str, base_url: str | None = None, api_key: str | None = None):
    class _S:
        configured_model_provider = provider
        configured_model_name = model
        openai_model = model
        openai_base_url = base_url
        openai_api_key = api_key
        configured_model_connection_mode = "local-runtime" if provider in {"ollama", "vllm"} else "remote-api"
        model_request_timeout_seconds = 30

    return _S()


if __name__ == "__main__":
    unittest.main()
