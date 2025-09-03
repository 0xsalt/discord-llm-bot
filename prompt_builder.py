#!/usr/bin/env python3
"""
prompt_builder.py
Composable prompt construction utilities for llmcord.
- identity_block: bot identity/persona and addressing behavior
- memory_context_block: optional memory summary block
- combine blocks with conversation messages to form final prompt

This module is intentionally dependency-light and does not import discord-llm-bot.py to avoid cycles.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import re


def _escape(s: str) -> str:
    return s if s is not None else ""

def _mask_label(s: str) -> str:
    """
    Mask internal identifiers like 'user:123', 'bot:...', 'channel:...' to friendly labels
    so the model does not echo numeric IDs back to users.
    """
    s = str(s or "")
    if s.startswith("user:"):
        return "user"
    if s.startswith("bot:"):
        return "bot"
    if s.startswith("channel:"):
        return "channel"
    return s


def identity_block(trigger_msg: Any, cfg: Any) -> List[Dict[str, Any]]:
    """
    Build the identity header using cfg.app.identity and cfg.prompts.enable_identity
    Includes the requesting user's display name and id for downstream tools/memory.
    """
    if not getattr(cfg, "prompts", None) or not getattr(cfg.prompts, "enable_identity", True):
        return []

    ident = getattr(cfg.app, "identity", None)
    name = getattr(ident, "name", "varuna") if ident else "varuna"
    persona = getattr(ident, "persona", "") if ident else ""
    aliases = getattr(ident, "addressing_aliases", ["varuna"]) if ident else ["varuna"]
    aliases_text = ", ".join(aliases) if aliases else name

    user_display = getattr(trigger_msg.author, "display_name", None) or getattr(trigger_msg.author, "name", "")
    user_id = str(getattr(trigger_msg.author, "id", ""))

    header_lines: List[str] = [
        f"You are {name}. When addressed as any of [{aliases_text}], treat it as a direct address."
    ]
    if persona:
        header_lines.append(f"Persona: {persona}")
    header_lines.append(f"Requesting user: {user_display}.")
    # Memory behavior hint (only if memory context is enabled)
    if getattr(getattr(cfg, "prompts", object()), "enable_memory_context", True):
        header_lines.append("You maintain a lightweight per-guild memory of explicit 'remember/record' requests and notable facts. Acknowledge 'remember' requests and use the Memory Context block when present.")

    header = "\n".join(header_lines)
    return [{"role": "system", "content": [{"type": "text", "text": header}]}]


def _format_memory_context(items: List[Dict[str, Any]], limit: int = 5, max_chars: int = 800) -> str:
    """
    Render retrieved memory items into a concise bullet list, capped by limit and total chars.
    Mirrors the representation used in llmcord for consistency.
    """
    lines: List[str] = []
    count = 0
    for it in items:
        if count >= max(limit, 0):
            break
        t = it.get("type")
        payload = it.get("payload", {}) or {}
        if t == "entity":
            raw_name = payload.get("name", "entity")
            name = _mask_label(_escape(raw_name))
            et = _escape(payload.get("entityType", "unknown"))
            lines.append(f"- entity: {name} ({et})")
        elif t == "relation":
            frm = _mask_label(payload.get("from", payload.get("from_", "")) or "")
            to = _mask_label(_escape(payload.get("to", "")))
            rt = _escape(payload.get("relationType", ""))
            lines.append(f"- relation: {frm} -[{rt}]-> {to}")
        elif t == "observation":
            en = _mask_label(_escape(payload.get("entityName", "unknown")))
            ct = _escape(payload.get("content", ""))
            lines.append(f"- obs({en}): {ct}")
        else:
            # fallback
            lines.append(f"- {t}: {str(payload)[:120]}")
        count += 1
        if sum(len(l) + 1 for l in lines) >= max_chars:
            break
    txt = "\n".join(lines)
    if len(txt) > max_chars:
        txt = txt[: max_chars - 3] + "..."
    return txt


def memory_context_block(cfg: Any, memory_items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build the Memory Context system block if enabled and items exist.
    """
    if not getattr(cfg, "prompts", None) or not getattr(cfg.prompts, "enable_memory_context", True):
        return []
    items = memory_items or []
    if not items:
        return []
    limit = getattr(getattr(cfg, "memory_service", object()), "retrieve_limit", 5)
    ctx_text = _format_memory_context(items, limit=limit)
    return [{"role": "system", "content": [{"type": "text", "text": f"Memory Context:\n{ctx_text}"}]}]


def policy_block(cfg: Any) -> List[Dict[str, Any]]:
    """
    Optional global policy/governance block configured via cfg.prompts.policy.
    """
    prmpts = getattr(cfg, "prompts", None)
    if not prmpts or not getattr(prmpts, "enable_policy", False):
        return []
    text = getattr(prmpts, "policy", "") or ""
    if not text.strip():
        return []
    return [{"role": "system", "content": [{"type": "text", "text": f"Policy:\n{text}"}]}]


def build_prompt_messages(
    trigger_msg: Any,
    cfg: Any,
    memory_items: Optional[List[Dict[str, Any]]],
    conversation_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compose the final messages list:
    [ identity_block?, memory_context_block?, ...conversation_messages ]
    - identity and memory blocks are optional controlled by cfg.prompts flags
    - conversation_messages should already include user/assistant content and attachments
    """
    final: List[Dict[str, Any]] = []
    final.extend(identity_block(trigger_msg, cfg))
    final.extend(policy_block(cfg))
    final.extend(memory_context_block(cfg, memory_items))
    final.extend(conversation_messages)
    return final