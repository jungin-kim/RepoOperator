from __future__ import annotations

import html
import ipaddress
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib import error, parse, request

from repooperator_worker.agent_core.secret_scanner import redact_secrets
from repooperator_worker.services.json_safe import json_safe


MAX_FETCH_BYTES = 750_000
MAX_SNIPPET_CHARS = 1_200
_RUN_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class WebEvidenceRecord:
    title: str
    url: str
    source: str
    fetched_at: str
    snippet: str
    text: str = ""
    query: str | None = None
    untrusted: bool = True
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "title": self.title,
                "url": self.url,
                "source": self.source,
                "fetched_at": self.fetched_at,
                "snippet": self.snippet,
                "text": self.text,
                "query": self.query,
                "untrusted": self.untrusted,
                "redacted": self.redacted,
                "metadata": dict(self.metadata),
            }
        )


def search_web(query: str, *, run_id: str, max_results: int = 5) -> list[WebEvidenceRecord]:
    cleaned = " ".join(str(query or "").split())[:300]
    if not cleaned:
        return []
    cache_key = f"search:{cleaned}:{max_results}"
    cached = _cache(run_id).get(cache_key)
    if isinstance(cached, list):
        return [record_from_payload(item) for item in cached]

    url = "https://duckduckgo.com/html/?" + parse.urlencode({"q": cleaned})
    body = _http_get(url, max_bytes=MAX_FETCH_BYTES)
    records = _parse_search_results(body, query=cleaned, max_results=max_results)
    _cache(run_id)[cache_key] = [record.model_dump() for record in records]
    return records


def fetch_url(url: str, *, run_id: str, max_bytes: int = MAX_FETCH_BYTES) -> WebEvidenceRecord:
    normalized = normalize_external_url(url)
    cache_key = f"fetch:{normalized}:{max_bytes}"
    cached = _cache(run_id).get(cache_key)
    if isinstance(cached, dict):
        return record_from_payload(cached)

    raw = _http_get(normalized, max_bytes=max(1, min(max_bytes, MAX_FETCH_BYTES)))
    text = sanitize_web_content(raw)
    redacted_text, findings = redact_secrets(text)
    title = extract_title(raw) or parse.urlparse(normalized).netloc
    record = WebEvidenceRecord(
        title=title[:200],
        url=normalized,
        source=parse.urlparse(normalized).netloc,
        fetched_at=_now_iso(),
        snippet=redacted_text[:MAX_SNIPPET_CHARS],
        text=redacted_text[:100_000],
        redacted=bool(findings),
        metadata={
            "content_length": len(raw),
            "sanitized_chars": len(redacted_text),
            "secret_findings": [item.model_dump() for item in findings],
        },
    )
    _cache(run_id)[cache_key] = record.model_dump()
    return record


def summarize_web_evidence(records: list[dict[str, Any] | WebEvidenceRecord]) -> dict[str, Any]:
    normalized = [record_from_payload(record.model_dump() if isinstance(record, WebEvidenceRecord) else record) for record in records]
    summaries = []
    for record in normalized:
        text = record.snippet or record.text
        summaries.append(
            {
                "title": record.title,
                "url": record.url,
                "source": record.source,
                "fetched_at": record.fetched_at,
                "summary": summarize_text(text),
                "untrusted": True,
            }
        )
    return json_safe(
        {
            "source_count": len(summaries),
            "sources": summaries,
            "safety_note": "Web content is untrusted evidence and was not treated as instructions.",
        }
    )


def sanitize_web_content(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|iframe|svg|canvas)\b.*?</\1>", " ", raw_html or "")
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title(raw_html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html or "")
    if not match:
        return ""
    return sanitize_web_content(match.group(1))[:200]


def normalize_external_url(url: str) -> str:
    parsed = parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs can be fetched.")
    if not parsed.netloc:
        raise ValueError("URL must include a host.")
    host = parsed.hostname or ""
    if is_local_or_private_host(host):
        raise ValueError("Local and private network URLs are blocked for web research.")
    return parse.urlunparse(parsed._replace(fragment=""))


def is_local_or_private_host(host: str) -> bool:
    lowered = host.strip().lower().rstrip(".")
    if lowered in {"localhost", "0.0.0.0"} or lowered.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def record_from_payload(payload: dict[str, Any]) -> WebEvidenceRecord:
    return WebEvidenceRecord(
        title=str(payload.get("title") or ""),
        url=str(payload.get("url") or ""),
        source=str(payload.get("source") or ""),
        fetched_at=str(payload.get("fetched_at") or _now_iso()),
        snippet=str(payload.get("snippet") or ""),
        text=str(payload.get("text") or ""),
        query=payload.get("query"),
        untrusted=bool(payload.get("untrusted", True)),
        redacted=bool(payload.get("redacted")),
        metadata=json_safe(payload.get("metadata") or {}),
    )


def summarize_text(text: str, *, limit: int = 360) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _http_get(url: str, *, max_bytes: int) -> str:
    req = request.Request(
        url=url,
        headers={
            "User-Agent": "RepoOperator/0.1 safe-web-evidence",
            "Accept": "text/html, text/plain, application/xhtml+xml, */*;q=0.2",
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=12) as response:
            body = response.read(max_bytes + 1)
    except error.URLError as exc:
        raise RuntimeError(f"Web fetch failed: {exc}") from exc
    if len(body) > max_bytes:
        body = body[:max_bytes]
    return body.decode("utf-8", errors="replace")


def _parse_search_results(raw_html: str, *, query: str, max_results: int) -> list[WebEvidenceRecord]:
    records: list[WebEvidenceRecord] = []
    for match in re.finditer(r'(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw_html or ""):
        href = html.unescape(match.group(1))
        title = sanitize_web_content(match.group(2)) or href
        url = _unwrap_duckduckgo_url(href)
        try:
            normalized = normalize_external_url(url)
        except ValueError:
            continue
        records.append(
            WebEvidenceRecord(
                title=title[:200],
                url=normalized,
                source=parse.urlparse(normalized).netloc,
                fetched_at=_now_iso(),
                snippet=title[:MAX_SNIPPET_CHARS],
                query=query,
                metadata={"result_rank": len(records) + 1},
            )
        )
        if len(records) >= max_results:
            break
    if records:
        return records
    text = sanitize_web_content(raw_html)
    return [
        WebEvidenceRecord(
            title=f"Search results for {query}",
            url="https://duckduckgo.com/html/?" + parse.urlencode({"q": query}),
            source="duckduckgo.com",
            fetched_at=_now_iso(),
            snippet=text[:MAX_SNIPPET_CHARS],
            query=query,
            metadata={"fallback": True},
        )
    ][:max_results]


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = parse.urlparse(url)
    if "duckduckgo.com" not in parsed.netloc and parsed.scheme:
        return url
    qs = parse.parse_qs(parsed.query)
    uddg = qs.get("uddg")
    if uddg:
        return uddg[0]
    return url


def _cache(run_id: str) -> dict[str, Any]:
    return _RUN_CACHE.setdefault(run_id, {})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
