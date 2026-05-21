from __future__ import annotations

from typing import Iterable

from repooperator_worker.agent_core.tools.base import ToolSpec
from repooperator_worker.services.json_safe import json_safe


class ToolSearch:
    """Search registered tool contracts without executing tools."""

    def __init__(self, registry) -> None:
        self.registry = registry

    def search(
        self,
        *,
        query: str | None = None,
        capability: str | None = None,
        capabilities: Iterable[str] | None = None,
        names: Iterable[str] | None = None,
        keywords: Iterable[str] | None = None,
        limit: int = 12,
        model_specs: bool = True,
    ) -> list[dict]:
        requested_capabilities = _normalized_terms([capability, *(capabilities or [])])
        requested_names = _normalized_terms(names or [])
        requested_keywords = _normalized_terms(keywords or [])
        query_terms = _normalized_terms((query or "").replace("-", "_").split())
        scored: list[tuple[int, int, ToolSpec]] = []
        for index, spec in enumerate(self.registry.specs()):
            score = self._score(
                spec,
                query_terms=query_terms,
                requested_capabilities=requested_capabilities,
                requested_names=requested_names,
                requested_keywords=requested_keywords,
            )
            if score > 0:
                scored.append((score, -index, spec))
        scored.sort(reverse=True)
        specs = [spec for _, _, spec in scored[: max(1, int(limit or 12))]]
        if model_specs:
            names_for_model = [spec.name for spec in specs]
            return self.registry.specs_for_model(tool_names=names_for_model, include_default=False)
        return json_safe([spec.model_dump() for spec in specs])

    def _score(
        self,
        spec: ToolSpec,
        *,
        query_terms: set[str],
        requested_capabilities: set[str],
        requested_names: set[str],
        requested_keywords: set[str],
    ) -> int:
        haystack = {
            spec.name.lower(),
            spec.operation.lower(),
            *[item.lower() for item in spec.capability_names],
            *[item.lower() for item in spec.tool_search_keywords],
            *_normalized_terms(spec.description.replace("-", "_").split()),
            *_normalized_terms(spec.prompt_summary.replace("-", "_").split()),
        }
        score = 0
        if requested_names and spec.name.lower() in requested_names:
            score += 100
        capability_matches = requested_capabilities.intersection({item.lower() for item in spec.capability_names})
        if capability_matches:
            score += 80 + len(capability_matches) * 5
        keyword_matches = requested_keywords.intersection(haystack)
        if keyword_matches:
            score += 40 + len(keyword_matches) * 3
        query_matches = query_terms.intersection(haystack)
        if query_matches:
            score += 20 + len(query_matches) * 2
        if not (requested_names or requested_capabilities or requested_keywords or query_terms):
            score = 1 if spec.always_load else 0
        return score


def _normalized_terms(items: Iterable[str | None]) -> set[str]:
    result: set[str] = set()
    for item in items:
        text = str(item or "").strip().lower()
        if text:
            result.add(text)
    return result
