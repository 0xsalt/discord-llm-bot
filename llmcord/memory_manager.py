#!/usr/bin/env python3
"""
memory_manager.py
Async HTTP client for MCP Knowledge Graph service.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Tuple

import httpx  # type: ignore
import structlog  # type: ignore


class EntityTD(TypedDict):
    name: str
    entityType: str
    observations: List[str]


class RelationTD(TypedDict):
    from_: str
    to: str
    relationType: str


class ObservationRecordTD(TypedDict):
    entityName: str
    contents: List[str]


class MemoryItemTD(TypedDict, total=False):
    type: str
    payload: Dict[str, Any]
    score: float
    id: str


@dataclass
class MemoryManager:
    cfg_state: Any
    # Circuit breaker state (optional resilience)
    _cb_fail_count: int = 0
    _cb_open_until: float = 0.0

    def _cfg(self) -> Any:
        return self.cfg_state.read()

    def is_enabled(self) -> bool:
        cfg = self._cfg()
        ms = getattr(cfg, "memory_service", None)
        return bool(ms and ms.enabled and ms.base_url)

    def _base_url(self) -> str:
        cfg = self._cfg()
        ms = getattr(cfg, "memory_service", None)
        return getattr(ms, "base_url", "http://mcp-server:8080")

    def _timeout(self) -> float:
        cfg = self._cfg()
        ms = getattr(cfg, "memory_service", None)
        return float(getattr(ms, "timeout_s", 2))

    def _retrieve_limit(self) -> int:
        cfg = self._cfg()
        ms = getattr(cfg, "memory_service", None)
        return int(getattr(ms, "retrieve_limit", 5))

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "mcp-api-version": "1",
        }

    def _cb_params(self) -> Tuple[int, float]:
        """
        Circuit breaker parameters:
        - threshold: consecutive failures to open breaker
        - cooldown_s: seconds to keep breaker open
        Defaults are safe; optionally read overrides from config.memory_service.*
        """
        cfg = self._cfg()
        ms = getattr(cfg, "memory_service", None)
        threshold = int(getattr(ms, "cb_failure_threshold", 5))
        cooldown_s = float(getattr(ms, "cb_cooldown_s", 30.0))
        return threshold, cooldown_s

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[float] = None,
        max_attempts: int = 3,
        base_delay: float = 0.2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
        """
        Returns: (json_body, error_message, duration_ms)
        """
        url = self._base_url().rstrip("/") + path
        timeout = timeout_sec if timeout_sec is not None else self._timeout()
        logger = structlog.get_logger()
        start = time.time()
        last_err: Optional[str] = None

        # Circuit breaker: short-circuit calls during cool-down
        now = time.time()
        if self._cb_open_until and now < self._cb_open_until:
            took_ms = (now - start) * 1000.0
            logger.warning("memory_http_circuit_open", url=url, open_until=self._cb_open_until, took_ms=round(took_ms, 2))
            return None, "circuit_open", took_ms

        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(method, url, headers=self._headers(), json=json)
                    took_ms = (time.time() - start) * 1000.0
                    if resp.status_code >= 200 and resp.status_code < 300:
                        try:
                            # Success: close breaker and reset fail count
                            self._cb_open_until = 0.0
                            self._cb_fail_count = 0
                            return resp.json(), None, took_ms
                        except Exception:
                            self._cb_open_until = 0.0
                            self._cb_fail_count = 0
                            return {}, None, took_ms
                    else:
                        # 4xx do not retry except 429; 5xx retry
                        body_text = ""
                        try:
                            body_text = resp.text
                        except Exception:
                            body_text = ""
                        if resp.status_code == 429 or 500 <= resp.status_code < 600:
                            last_err = f"http_{resp.status_code}"
                        else:
                            # Non-retryable: do not increment breaker fail count; return immediately
                            return None, f"http_{resp.status_code}:{body_text[:200]}", took_ms
            except Exception as e:
                last_err = str(e)

            # backoff
            delay = base_delay * (2 ** (attempt - 1))
            delay = delay + random.uniform(0, delay * 0.25)
            await asyncio.sleep(delay)

        took_ms = (time.time() - start) * 1000.0
        logger.warning("memory_http_retry_exhausted", url=url, attempts=max_attempts, error=last_err, took_ms=round(took_ms, 2))
        # Circuit breaker open decision on retry exhaustion (only for retryable categories)
        try:
            thr, cooldown = self._cb_params()
        except Exception:
            thr, cooldown = 5, 30.0
        # Increment fail count for breaker on retry-exhausted errors
        self._cb_fail_count = (self._cb_fail_count or 0) + 1
        if self._cb_fail_count >= max(1, thr):
            self._cb_open_until = time.time() + max(1.0, cooldown)
            logger.warning("memory_http_circuit_opened", url=url, fail_count=self._cb_fail_count, open_until=self._cb_open_until, threshold=thr, cooldown_s=cooldown)
            # reset fail count after opening to count post-cooldown failures anew
            self._cb_fail_count = 0
        return None, last_err, took_ms

    async def health_check(self) -> Dict[str, Any]:
        logger = structlog.get_logger()
        if not self.is_enabled():
            logger.info("memory.health", enabled=False)
            return {"ok": False, "enabled": False}
        body, err, took_ms = await self._request("GET", "/health", None)
        ok = bool(body and body.get("status") == "ok" and not err)
        logger.info("memory.health", ok=ok, took_ms=round(took_ms, 2), error=err)
        return {"ok": ok, "data": body or {}, "error": err, "took_ms": took_ms}

    async def upsert_memory(
        self,
        *,
        entities: Optional[List[EntityTD]] = None,
        relations: Optional[List[RelationTD]] = None,
        observations: Optional[List[ObservationRecordTD]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fire-and-forget safe: caller may ignore errors; method returns created counts or error.
        """
        logger = structlog.get_logger()
        if not self.is_enabled():
            logger.info("memory.upsert", enabled=False)
            return {"created": {"entities": 0, "relations": 0, "observations": 0}, "skipped": True}

        payload: Dict[str, Any] = {}
        if context:
            payload["context"] = context
        if entities:
            # map RelationTD uses from_ workaround; convert to server field names when needed
            payload["entities"] = entities
        if relations:
            payload["relations"] = [{"from": r["from_"], "to": r["to"], "relationType": r["relationType"]} for r in relations]
        if observations:
            payload["observations"] = observations
        if provenance:
            payload["provenance"] = provenance

        body, err, took_ms = await self._request("POST", "/v1/memory/upsert", payload)
        ok = bool(body and not err)
        logger.info("memory.upsert", ok=ok, took_ms=round(took_ms, 2), error=err)
        return body or {"error": err or "unknown"}

    async def query_relevant(
        self,
        *,
        terms: Optional[List[str]] = None,
        context_hint: Optional[str] = None,
        retrieve_limit: Optional[int] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns: { items: MemoryItemTD[], total: int, took_ms: number }
        """
        logger = structlog.get_logger()
        if not self.is_enabled():
            logger.info("memory.query", enabled=False)
            return {"items": [], "total": 0, "took_ms": 0}

        limit = int(retrieve_limit) if retrieve_limit is not None else self._retrieve_limit()
        payload: Dict[str, Any] = {
            "terms": terms or [],
            "retrieve_limit": limit,
        }
        if context_hint:
            payload["context_hint"] = context_hint
        # optional scope hints for future server-side filtering
        if user_id:
            payload["user_id"] = user_id
        if channel_id:
            payload["channel_id"] = channel_id

        body, err, took_ms = await self._request("POST", "/v1/memory/query", payload)
        ok = bool(body and not err)
        logger.info("memory.query", ok=ok, took_ms=round(took_ms, 2), error=err, retrieved=(len(body.get("items", [])) if body else 0))
        return body or {"items": [], "total": 0, "took_ms": took_ms, "error": err or "unknown"}


# Convenience constructors for typed inputs
def entity(name: str, entity_type: str, observations: Optional[List[str]] = None) -> EntityTD:
    return {"name": name, "entityType": entity_type, "observations": observations or []}


def relation(from_name: str, to_name: str, relation_type: str) -> RelationTD:
    return {"from_": from_name, "to": to_name, "relationType": relation_type}


def observation(entity_name: str, contents: List[str]) -> ObservationRecordTD:
    return {"entityName": entity_name, "contents": contents}


__all__ = [
    "MemoryManager",
    "EntityTD",
    "RelationTD",
    "ObservationRecordTD",
    "MemoryItemTD",
    "entity",
    "relation",
    "observation",
]