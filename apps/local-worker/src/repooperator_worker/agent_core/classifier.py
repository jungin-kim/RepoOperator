"""
Compatibility shim — do not add logic here.

All real implementation lives in request_understanding.py.
This module re-exports the old classifier symbols so existing direct imports
keep working without old routing fields.
"""
from __future__ import annotations

from repooperator_worker.agent_core.request_understanding import (
    RequestUnderstanding,
    _parse_json as _parse_classifier_json,
    request_understanding_to_classifier_result,
    understand_request,
)
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.agent_core.state import ClassifierResult


def classify_intent(request: AgentRunRequest) -> ClassifierResult:
    """Backward-compat wrapper: understand_request + ClassifierResult adapter."""
    ru = understand_request(request)
    return request_understanding_to_classifier_result(ru, request)


# Legacy name used by some test helpers — maps to validate logic in request_understanding.
def validate_classifier_payload(payload: dict, request: AgentRunRequest) -> ClassifierResult:
    from repooperator_worker.agent_core.request_understanding import _build_understanding
    ru = _build_understanding(payload, request)
    return request_understanding_to_classifier_result(ru, request)


__all__ = [
    "classify_intent",
    "validate_classifier_payload",
    "_parse_classifier_json",
    "RequestUnderstanding",
    "request_understanding_to_classifier_result",
    "understand_request",
]
