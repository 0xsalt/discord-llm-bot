#!/usr/bin/env python3
"""
llmcord.py
Core single-file application for the Discord LLM bot MVP.

Capabilities (per PRD and Architecture):
- Public channels: reply on @mention (configurable)
- Direct messages: reply to all
- Reply-chain context building up to configured depth
- Streaming responses with chunking to fit Discord limits
- View-only /model command showing active LLM model details
- Permissions gating (users/roles/channels) via config.yaml
- Hot reload of configuration (no restart)
- Structured logging via structlog, with rotation policy
- Docker-only runtime; tokens supplied via environment
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml
import structlog  # type: ignore
import logging
from logging.handlers import RotatingFileHandler

import discord  # type: ignore
from discord import app_commands, Intents, Message  # type: ignore
from discord.ext import commands  # type: ignore

from openai import OpenAI  # type: ignore
from openai.types.chat import ChatCompletionChunk  # type: ignore

# Memory service client (support both package and script execution)
try:
    from .memory_manager import MemoryManager, entity, relation, observation  # type: ignore
except Exception:  # pragma: no cover
    from memory_manager import MemoryManager, entity, relation, observation  # type: ignore

# Prompt builder (modular prompt construction)
try:
    from .prompt_builder import build_prompt_messages  # type: ignore
except Exception:  # pragma: no cover
    from prompt_builder import build_prompt_messages  # type: ignore

# ---------------------------
# Slash command group: /memory summarize
# ---------------------------
memory_group = app_commands.Group(name="memory", description="Memory commands")

@memory_group.command(name="summarize", description="Summarize top N memory items about you in this context")
@app_commands.describe(limit="Number of items to include (1-20)")
async def memory_summarize(interaction: discord.Interaction, limit: int = 5):
    """
    Provide a concise memory summary for the requesting user in the current guild/channel context.
    Uses MemoryManager.query_relevant() with user/channel scoped terms and formats with format_memory_context().
    """
    # Safely access the bot instance and config
    bot = cast("LlmCord", interaction.client)  # type: ignore
    cfg = bot.cfg_state.read()

    # Bound limit to a sane range
    lim = max(1, min(int(limit), 20))

    # Build scope
    context_hint = str(interaction.guild.id) if interaction.guild else None
    channel_id = str(getattr(interaction.channel, "id", "")) if interaction.channel else None
    user_id = str(interaction.user.id)

    # Drive simple scoring by including canonical entity names used in upserts
    terms = [f"user:{user_id}"]
    if channel_id:
        terms.append(f"channel:{channel_id}")

    try:
        res = await bot.memory.query_relevant(
            terms=terms,
            context_hint=context_hint,
            retrieve_limit=lim,
            user_id=user_id,
            channel_id=channel_id,
        )
        items = res.get("items", [])
        if not items:
            await interaction.response.send_message(
                "No relevant memory found for you in this context, or memory is disabled.",
                ephemeral=True,
            )
            return

        ctx_text = format_memory_context(items, limit=lim)
        await interaction.response.send_message(
            f"Memory Summary (top {lim}):\n{ctx_text}",
            ephemeral=True,
        )
    except Exception as e:
        # Non-fatal behavior
        bot.logger.warning("memory_summarize_error", error=str(e))
        await interaction.response.send_message(
            "Unable to retrieve memory summary right now.",
            ephemeral=True,
        )


# ---------------------------
# Config Management
# ---------------------------

DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH") or str(Path(__file__).with_name("config.yaml"))
LOG_DIR = Path(os.getenv("LOG_DIR") or Path(__file__).with_name("logs"))

@dataclass
class AppResponseCfg:
    stream: bool = True
    max_chunk_chars: int = 1800
    code_block_boundary_safe: bool = True

@dataclass
class AppIdentityCfg:
    name: str = "varuna"
    addressing_aliases: List[str] = field(default_factory=lambda: ["varuna"])
    persona: str = ""

@dataclass
class AppCfg:
    status_message: str = "llmcord: community AI assistant"
    max_context_messages: int = 12
    reply_preamble: bool = True
    response: AppResponseCfg = field(default_factory=AppResponseCfg)
    identity: AppIdentityCfg = field(default_factory=AppIdentityCfg)

@dataclass
class DiscordCfg:
    token_env: str = "DISCORD_BOT_TOKEN"
    require_mention_in_guilds: bool = True
    allow_dm: bool = True

@dataclass
class LLMCfg:
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 1024
    request_timeout_secs: int = 60

@dataclass
class PermissionsList:
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)

@dataclass
class PermissionsCfg:
    users: PermissionsList = field(default_factory=PermissionsList)
    roles: PermissionsList = field(default_factory=PermissionsList)
    channels: PermissionsList = field(default_factory=PermissionsList)

@dataclass
class LoggingCfg:
    level: str = "INFO"
    json: bool = True
    include_context: List[str] = field(default_factory=lambda: ["guild_id","channel_id","user_id","message_id"])
    rotation: str = "session"  # none | size | session
    max_bytes: int = 1048576
    backups: int = 3

@dataclass
class HotReloadCfg:
    enabled: bool = True
    watch_paths: List[str] = field(default_factory=lambda: ["config.yaml"])
    debounce_ms: int = 200

@dataclass
class PromptsCfg:
    enable_identity: bool = True
    enable_memory_context: bool = True
    enable_policy: bool = False
    policy: str = ""


@dataclass
class MemoryServiceCfg:
    enabled: bool = False
    base_url: Optional[str] = "http://mcp-server:8080"
    timeout_s: int = 2
    retrieve_limit: int = 5

@dataclass
class AttachTextCfg:
    max_chars: int = 50000
    summarize_overflow: bool = True

@dataclass
class AttachImagesCfg:
    max_count: int = 5

@dataclass
class AttachmentsCfg:
    allow_images: bool = True
    allow_text: bool = True
    max_total_mb: float = 10.0
    text: AttachTextCfg = field(default_factory=AttachTextCfg)
    images: AttachImagesCfg = field(default_factory=AttachImagesCfg)

@dataclass
class SecurityCfg:
    allowed_guilds: List[str] = field(default_factory=list)
    blocked_guilds: List[str] = field(default_factory=list)

@dataclass
class Config:
    app: AppCfg = field(default_factory=AppCfg)
    discord: DiscordCfg = field(default_factory=DiscordCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    memory_service: MemoryServiceCfg = field(default_factory=MemoryServiceCfg)
    permissions: PermissionsCfg = field(default_factory=PermissionsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    hot_reload: HotReloadCfg = field(default_factory=HotReloadCfg)
    attachments: AttachmentsCfg = field(default_factory=AttachmentsCfg)
    security: SecurityCfg = field(default_factory=SecurityCfg)
    prompts: PromptsCfg = field(default_factory=PromptsCfg)


class ConfigState:
    def __init__(self, path: str):
        self.path = path
        self.raw: Dict[str, Any] = {}
        self.cfg = Config()
        self._mtime = 0.0
        self._lock = threading.RLock()
        self._load_initial()

    def _load_initial(self):
        self._load()

    def _load(self):
        try:
            p = Path(self.path)
            if not p.exists():
                structlog.get_logger().warning("config_missing", path=self.path)
                return
            mtime = p.stat().st_mtime
            if mtime <= self._mtime and self.raw:
                return
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.raw = data
            old_cfg = self.cfg
            new_cfg = self._from_dict(data)
            # Log selected diffs on hot-reload (non-secret keys)
            try:
                self._log_cfg_diff(old_cfg, new_cfg)
            except Exception:
                pass
            self.cfg = new_cfg
            self._mtime = mtime
            structlog.get_logger().info("config_loaded", path=self.path, mtime=self._mtime)
        except Exception as e:
            structlog.get_logger().error("config_load_error", error=str(e))

    def _from_dict(self, d: Dict[str, Any]) -> Config:
        # shallow-to-dataclass mapping
        def map_obj(cls, obj):
            if isinstance(obj, dict):
                return cls(**{
                    k: map_field(getattr(cls, k, None), v)
                    for k, v in obj.items()
                    if hasattr(cls, k)
                })
            return cls()

        def map_field(template, value):
            if isinstance(template, AppResponseCfg):
                return AppResponseCfg(**value)
            if isinstance(template, AppCfg):
                res = value.get("response", {})
                ident = value.get("identity", {})
                return AppCfg(
                    status_message=value.get("status_message", "llmcord: community AI assistant"),
                    max_context_messages=value.get("max_context_messages", 12),
                    reply_preamble=value.get("reply_preamble", True),
                    response=AppResponseCfg(**res) if isinstance(res, dict) else AppResponseCfg(),
                    identity=AppIdentityCfg(**ident) if isinstance(ident, dict) else AppIdentityCfg(),
                )
            if isinstance(template, DiscordCfg):
                return DiscordCfg(**value)
            if isinstance(template, LLMCfg):
                return LLMCfg(**value)
            if isinstance(template, MemoryServiceCfg):
                return MemoryServiceCfg(**value)
            if isinstance(template, PermissionsList):
                return PermissionsList(**value)
            if isinstance(template, PermissionsCfg):
                users = value.get("users", {})
                roles = value.get("roles", {})
                channels = value.get("channels", {})
                return PermissionsCfg(
                    users=PermissionsList(**users) if isinstance(users, dict) else PermissionsList(),
                    roles=PermissionsList(**roles) if isinstance(roles, dict) else PermissionsList(),
                    channels=PermissionsList(**channels) if isinstance(channels, dict) else PermissionsList(),
                )
            if isinstance(template, LoggingCfg):
                return LoggingCfg(**value)
            if isinstance(template, HotReloadCfg):
                return HotReloadCfg(**value)
            if isinstance(template, AttachTextCfg):
                return AttachTextCfg(**value)
            if isinstance(template, AttachImagesCfg):
                return AttachImagesCfg(**value)
            if isinstance(template, AttachmentsCfg):
                text = value.get("text", {})
                images = value.get("images", {})
                return AttachmentsCfg(
                    allow_images=value.get("allow_images", True),
                    allow_text=value.get("allow_text", True),
                    max_total_mb=value.get("max_total_mb", 10.0),
                    text=AttachTextCfg(**text) if isinstance(text, dict) else AttachTextCfg(),
                    images=AttachImagesCfg(**images) if isinstance(images, dict) else AttachImagesCfg(),
                )
            if isinstance(template, SecurityCfg):
                return SecurityCfg(**value)
            if isinstance(template, PromptsCfg):
                return PromptsCfg(**value)
            return value

        cfg = Config()
        # manual expand to apply mapping
        cfg.app = cast(AppCfg, map_field(cfg.app, d.get("app", {})))
        cfg.discord = cast(DiscordCfg, map_field(cfg.discord, d.get("discord", {})))
        cfg.llm = cast(LLMCfg, map_field(cfg.llm, d.get("llm", {})))
        cfg.memory_service = cast(MemoryServiceCfg, map_field(cfg.memory_service, d.get("memory_service", {})))
        cfg.permissions = cast(PermissionsCfg, map_field(cfg.permissions, d.get("permissions", {})))
        cfg.logging = cast(LoggingCfg, map_field(cfg.logging, d.get("logging", {})))
        cfg.hot_reload = cast(HotReloadCfg, map_field(cfg.hot_reload, d.get("hot_reload", {})))
        cfg.attachments = cast(AttachmentsCfg, map_field(cfg.attachments, d.get("attachments", {})))
        cfg.security = cast(SecurityCfg, map_field(cfg.security, d.get("security", {})))
        cfg.prompts = cast(PromptsCfg, map_field(cfg.prompts, d.get("prompts", {})))
        return cfg

    def read(self) -> Config:
        with self._lock:
            return self.cfg

    def _log_cfg_diff(self, old: Config, new: Config):
        logger = structlog.get_logger()

        def _section_diff(old_obj: Any, new_obj: Any, keys: List[str]) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for k in keys:
                ov = getattr(old_obj, k, None)
                nv = getattr(new_obj, k, None)
                if ov != nv:
                    out[k] = {"old": ov, "new": nv}
            return out

        try:
            mem_old = getattr(old, "memory_service", object())
            mem_new = getattr(new, "memory_service", object())
            prm_old = getattr(old, "prompts", object())
            prm_new = getattr(new, "prompts", object())

            mem_changes = _section_diff(mem_old, mem_new, ["enabled", "base_url", "timeout_s", "retrieve_limit"])
            prm_changes = _section_diff(prm_old, prm_new, ["enable_identity", "enable_memory_context", "enable_policy"])

            # Do not log policy text; just log length delta
            try:
                old_len = len(getattr(prm_old, "policy", "") or "")
                new_len = len(getattr(prm_new, "policy", "") or "")
                if old_len != new_len:
                    prm_changes["policy_len"] = {"old": old_len, "new": new_len}
            except Exception:
                pass

            if mem_changes or prm_changes:
                logger.info("config_hot_reload_diff", memory_service=mem_changes, prompts=prm_changes)
        except Exception:
            # Best-effort diff; never break reload
            pass

    def watch(self, stop_event: threading.Event):
        if not self.read().hot_reload.enabled:
            return
        debounce = max(50, self.read().hot_reload.debounce_ms) / 1000.0
        last_check = 0.0
        while not stop_event.is_set():
            now = time.time()
            if now - last_check >= debounce:
                self._load()
                last_check = now
            time.sleep(0.1)


# ---------------------------
# Logging
# ---------------------------

def setup_logging(cfg: LoggingCfg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = []
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    if cfg.rotation == "size":
        handler = RotatingFileHandler(
            filename=str(LOG_DIR / "llmcord.log"),
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backups,
            encoding="utf-8"
        )
        handlers.append(handler)
    elif cfg.rotation == "session":
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        handler = logging.FileHandler(str(LOG_DIR / f"llmcord-{ts}.log"), encoding="utf-8")
        handlers.append(handler)

    # Always include stderr handler for visibility
    stderr = logging.StreamHandler(sys.stderr)
    handlers.append(stderr)

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")

    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if cfg.json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer(key_order=["event", *cfg.include_context]))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------
# Permissions
# ---------------------------

def _id_in_list(target_id: str, allow: List[str], deny: List[str]) -> Tuple[bool, Optional[str]]:
    # Deny precedence
    if target_id in set(deny):
        return False, "deny"
    if allow and target_id not in set(allow):
        return False, "not_in_allow"
    return True, None

async def is_allowed(ctx_message: Message, cfg: Config, bot_user_id: int) -> Tuple[bool, Optional[str]]:
    user_id = str(ctx_message.author.id)
    channel_id = str(ctx_message.channel.id)

    ok, reason = _id_in_list(user_id, cfg.permissions.users.allow, cfg.permissions.users.deny)
    if not ok:
        return False, f"user_{reason}"

    ok, reason = _id_in_list(channel_id, cfg.permissions.channels.allow, cfg.permissions.channels.deny)
    if not ok:
        return False, f"channel_{reason}"

    # Role checks (guild only)
    if hasattr(ctx_message.author, "roles"):
        role_ids = {str(r.id) for r in getattr(ctx_message.author, "roles", [])}
        if cfg.permissions.roles.deny and role_ids.intersection(set(cfg.permissions.roles.deny)):
            return False, "role_deny"
        if cfg.permissions.roles.allow and not role_ids.intersection(set(cfg.permissions.roles.allow)):
            return False, "role_not_in_allow"

    # Guild security allow/deny
    if ctx_message.guild:
        guild_id = str(ctx_message.guild.id)
        if cfg.security.blocked_guilds and guild_id in set(cfg.security.blocked_guilds):
            return False, "guild_blocked"
        if cfg.security.allowed_guilds and guild_id not in set(cfg.security.allowed_guilds):
            return False, "guild_not_in_allow"

    return True, None


# ---------------------------
# Context Building
# ---------------------------

async def collect_reply_chain(message: Message, max_depth: int) -> List[Message]:
    """Collect reply chain from the message backwards up to max_depth. Oldest first."""
    chain: List[Message] = []
    current = message
    depth = 0
    try:
        while current and depth < max_depth and current.reference and current.reference.message_id:
            ref_id = current.reference.message_id
            try:
                parent = await current.channel.fetch_message(ref_id)
            except Exception:
                break
            chain.append(parent)
            current = parent
            depth += 1
    except Exception:
        pass
    return list(reversed(chain))


def attachments_to_llm_content(msg: Message, cfg: Config) -> List[Dict[str, Any]]:
    """Convert Discord attachments to OpenAI chat content blocks."""
    blocks: List[Dict[str, Any]] = []
    if not msg.attachments:
        return blocks

    total_mb = 0.0
    images_count = 0

    for att in msg.attachments:
        total_mb += (att.size or 0) / (1024.0 * 1024.0)

    if total_mb > cfg.attachments.max_total_mb:
        # Skip attachments if too large; caller should notify via warning
        return blocks

    for att in msg.attachments:
        mime = att.content_type or ""
        if cfg.attachments.allow_images and mime.startswith("image/"):
            if images_count < cfg.attachments.images.max_count:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": att.url}
                })
                images_count += 1
        elif cfg.attachments.allow_text and (mime.startswith("text/") or att.filename.lower().endswith((".txt", ".md"))):
            # For simplicity, include a pointer to the URL. In a richer flow, fetch content and truncate.
            blocks.append({
                "type": "text",
                "text": f"[Attached text file: {att.filename}] {att.url}"
            })

    return blocks


async def build_messages_for_llm(trigger_msg: Message, cfg: Config) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with multimodal content (conversation only).
    System identity and memory context are added by the PromptBuilder layer.
    """
    messages: List[Dict[str, Any]] = []

    # Accumulate reply chain
    chain = await collect_reply_chain(trigger_msg, cfg.app.max_context_messages)

    # Include chain messages and the final trigger message
    conversation: List[Message] = chain + [trigger_msg]

    for m in conversation:
        role = "user" if not m.author.bot else "assistant"
        content_blocks: List[Dict[str, Any]] = []

        if m.content:
            content_blocks.append({"type": "text", "text": m.content})

        # Attachments
        content_blocks.extend(attachments_to_llm_content(m, cfg))

        if not content_blocks:
            # Avoid empty content
            content_blocks.append({"type": "text", "text": ""})

        messages.append({
            "role": role,
            "content": content_blocks
        })

    return messages


# ---------------------------
# LLM Streaming and Chunking
# ---------------------------

def chunk_text(s: str, max_len: int, code_safe: bool = True) -> List[str]:
    """Chunk text to max_len, trying to avoid splitting code blocks."""
    if len(s) <= max_len:
        return [s]

    chunks: List[str] = []
    current = s

    def balanced_code(text: str) -> bool:
        # Count occurrences of triple backticks
        return text.count("```") % 2 == 0

    while current:
        if len(current) <= max_len:
            chunks.append(current)
            break
        # find split point near max_len at a boundary
        split = current.rfind("\n\n", 0, max_len)
        if split == -1:
            split = current.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len

        candidate = current[:split]
        if code_safe:
            # If chunk would end unbalanced, try earlier boundary
            while not balanced_code(candidate) and split > 0:
                split = current.rfind("\n", 0, split - 1)
                if split == -1:
                    break
                candidate = current[:split]
            if split == -1:
                candidate = current[:max_len]

        chunks.append(candidate)
        current = current[len(candidate):].lstrip("\n")

    return chunks


def build_preamble(cfg: Config) -> str:
    """Build the model identification preamble required by architecture."""
    model = cfg.llm.model
    # Best-effort fields; can be refined by querying provider metadata later.
    model_type = "LLM"
    model_size = os.getenv("LLM_MODEL_SIZE", "n/a")
    model_rev = os.getenv("LLM_MODEL_REV", datetime.utcnow().strftime("%Y-%m"))
    return f"[model={model} | size={model_size} | type={model_type} | rev={model_rev}] "


# -------- Memory helpers (Story 2.3/2.4) --------

def _mask_label(s: str) -> str:
    """
    Mask internal identifiers like 'user:123', 'bot:...', 'channel:...' to friendly labels
    so user-facing summaries and context never expose numeric IDs.
    """
    s = str(s or "")
    if s.startswith("user:"):
        return "user"
    if s.startswith("bot:"):
        return "bot"
    if s.startswith("channel:"):
        return "channel"
    return s

def extract_terms(text: str, max_terms: int = 8) -> List[str]:
    """Very lightweight term extraction: split on non-alnum, filter short tokens, dedupe, keep first N."""
    if not text:
        return []
    import re
    tokens = re.split(r"[^A-Za-z0-9_]+", text.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    # keep unique in order
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def format_memory_context(items: List[Dict[str, Any]], limit: int = 5, max_chars: int = 800) -> str:
    """Render retrieved memory items into a concise bullet list, capped by limit and chars. Masks internal IDs."""
    lines: List[str] = []
    count = 0
    for it in items:
        if count >= max(limit, 0):
            break
        t = it.get("type")
        payload = it.get("payload", {}) or {}
        if t == "entity":
            raw_name = payload.get("name", "entity")
            name = _mask_label(raw_name)
            et = payload.get("entityType", "unknown")
            lines.append(f"- entity: {name} ({et})")
        elif t == "relation":
            frm = _mask_label(payload.get("from", payload.get("from_", "")))
            to = _mask_label(payload.get("to", ""))
            rt = payload.get("relationType", "")
            lines.append(f"- relation: {frm} -[{rt}]-> {to}")
        elif t == "observation":
            en = _mask_label(payload.get("entityName", "unknown"))
            ct = payload.get("content", "")
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

# Partitioning helper for memory context selection
def compute_partition_context(message: Message, strategy: str = "guild") -> Optional[str]:
    """
    Default: 'guild' — one file per Discord server (memory-<guild_id>.jsonl).
    Future options (commented for easy extension):
      - 'channel': one file per channel (memory-<channel_id>.jsonl)
      - 'global' : single file for everything (memory.jsonl)
    Switching the strategy should be applied to BOTH retrieval and upserts.
    """
    if strategy == "guild" and message.guild:
        return str(message.guild.id)
    if strategy == "channel":
        return str(getattr(message.channel, "id", ""))
    if strategy == "global":
        return None
    # Fallback to guild or global
    return str(message.guild.id) if message.guild else None

async def stream_completion_and_send(
    client: OpenAI,
    cfg: Config,
    channel: discord.abc.Messageable,
    messages: List[Dict[str, Any]],
):
    logger = structlog.get_logger()
    # Create chat completion with streaming
    try:
        stream = client.chat.completions.create(
            model=cfg.llm.model,
            messages=messages,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_tokens,
            stream=cfg.app.response.stream,
        )
    except Exception as e:
        logger.error("llm_request_error", error=str(e))
        await channel.send("An error occurred while contacting the language model.")
        return ""

    collected = []
    try:
        for chunk in stream:
            if isinstance(chunk, ChatCompletionChunk):
                for choice in chunk.choices:
                    delta = choice.delta.content or ""
                    if delta:
                        collected.append(delta)
    except Exception as e:
        logger.error("llm_stream_error", error=str(e))

    full_text = "".join(collected).strip()
    if not full_text:
        await channel.send("The model returned no content.")
        return ""

    # Preamble on first chunk only
    if cfg.app.reply_preamble:
        preamble = build_preamble(cfg).rstrip()
        full_text = f"{preamble}\n---\n{full_text}"

    pieces = chunk_text(
        full_text,
        max_len=cfg.app.response.max_chunk_chars,
        code_safe=cfg.app.response.code_block_boundary_safe
    )

    for piece in pieces:
        await channel.send(piece)
    return full_text


# ---------------------------
# Discord Bot
# ---------------------------

class LlmCord(commands.Bot):
    def __init__(self, cfg_state: ConfigState, *args, **kwargs):
        intents = kwargs.pop("intents", None)
        if intents is None:
            intents = Intents.default()
            intents.message_content = True
            intents.members = False

        super().__init__(*args, intents=intents, **kwargs)
        self.cfg_state = cfg_state
        self.logger = structlog.get_logger()
        self.client: Optional[OpenAI] = None
        # Initialize memory manager (used by later stories; safe no-op if disabled)
        self.memory = MemoryManager(cfg_state=self.cfg_state)


    async def setup_hook(self):
        cfg = self.cfg_state.read()
        # OpenAI client
        api_key = os.getenv(cfg.llm.api_key_env)
        client_kwargs = {}
        if cfg.llm.base_url:
            client_kwargs["base_url"] = cfg.llm.base_url
        self.client = OpenAI(api_key=api_key, **client_kwargs)

        # Sync command tree
        async def _register():
            try:
                self.tree.add_command(self.model_cmd)
                await self.tree.sync()
            except Exception as e:
                self.logger.error("slash_sync_error", error=str(e))
        await _register()

        # Register memory group
        try:
            self.tree.add_command(memory_group)
        except Exception as e:
            self.logger.warning("slash_group_add_error", error=str(e))

        # Status
        try:
            await self.change_presence(activity=discord.Game(name=cfg.app.status_message))
        except Exception:
            pass

        # Memory service startup health probe (non-fatal)
        try:
            res = await self.memory.health_check()
            self.logger.info(
                "memory_startup_health",
                ok=res.get("ok"),
                took_ms=round(res.get("took_ms", 0.0), 2),
            )
        except Exception as e:
            self.logger.warning("memory_startup_health_error", error=str(e))

    async def on_ready(self):
        self.logger.info("bot_ready", user=str(self.user), id=self.user.id if self.user else None)

    async def on_message(self, message: Message):
        # Ignore own messages
        if self.user and message.author.id == self.user.id:
            return

        cfg = self.cfg_state.read()
        bot_id = self.user.id if self.user else 0
        allowed, reason = await is_allowed(message, cfg, bot_id)
        if not allowed:
            self.logger.info(
                "permission_denied",
                reason=(reason or "not_allowed"),
                guild_id=(str(message.guild.id) if message.guild else None),
                channel_id=str(message.channel.id),
                author_id=str(message.author.id),
                message_id=message.id,
            )
            await self._warn_permissions(message, reason or "not_allowed")
            return
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentioned = bool(self.user and self.user in message.mentions)

        # Robust mention detection: also recognize raw ID tokens (<@id>, <@!id>)
        raw_id_tokens: set[str] = set()
        if self.user:
            raw_id_tokens = {f"<@{self.user.id}>", f"<@!{self.user.id}>"}
        if message.content and raw_id_tokens and any(tok in message.content for tok in raw_id_tokens):
            mentioned = True

        # Treat replies to the bot as implicit mentions (reply chain UX)
        reply_to_bot = False
        if message.reference and message.reference.message_id:
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
                reply_to_bot = bool(self.user and ref_msg.author and ref_msg.author.id == self.user.id)
            except Exception:
                reply_to_bot = False

        # Diagnostic log for guild/gating investigations
        self.logger.info(
            "msg_received",
            guild_id=(str(message.guild.id) if message.guild else None),
            channel_id=str(message.channel.id),
            author_id=str(message.author.id),
            message_id=message.id,
            mentioned=mentioned,
            reply_to_bot=reply_to_bot,
            mention_ids=[str(u.id) for u in message.mentions],
            bot_id=(str(self.user.id) if self.user else None),
            content_has_id_mention=bool(self.user and any(tok in (message.content or "") for tok in raw_id_tokens)),
            is_dm=is_dm,
            require_mention_in_guilds=cfg.discord.require_mention_in_guilds,
            allow_dm=cfg.discord.allow_dm,
            content_preview=(message.content or "")[:120],
        )


        if message.guild:
            if cfg.discord.require_mention_in_guilds and not (mentioned or reply_to_bot):
                self.logger.info(
                    "ignore_no_mention",
                    guild_id=str(message.guild.id),
                    channel_id=str(message.channel.id),
                    author_id=str(message.author.id),
                    message_id=message.id,
                    mentioned=mentioned,
                    reply_to_bot=reply_to_bot,
                )
                return

        if is_dm and not cfg.discord.allow_dm:
            self.logger.info(
                "ignore_dm_disabled",
                author_id=str(message.author.id),
                message_id=message.id,
            )
            return

        # Build LLM messages and stream
        self.logger.info(
            "llm_request_start",
            guild_id=(str(message.guild.id) if message.guild else None),
            channel_id=str(message.channel.id),
            author_id=str(message.author.id),
            message_id=message.id,
            mentioned=mentioned,
            is_dm=is_dm,
        )
        try:
            llm_messages = await build_messages_for_llm(message, cfg)
        except Exception as e:
            self.logger.error("context_build_error", error=str(e))
            await message.channel.send("Failed to build conversation context.")
            return

        # Attachment size check warnings (if skipped due to size)
        # Simplified: notify if too large was detected in attachments_to_llm_content by emptying despite attachments
        # (Omitted: full size computation path feedback to user)

        # Memory retrieval (Story 2.4) — gather Memory Context items (no prompt mutation here)
        context_hint = compute_partition_context(message, "guild")
        retrieved_items: List[Dict[str, Any]] = []
        try:
            # Seed retrieval with: message terms, user/channel canonical tags, display name tokens, and a 'memory' anchor
            base_terms = extract_terms(message.content or "")
            user_id = str(message.author.id)
            channel_id = str(getattr(message.channel, "id", ""))
            user_display = getattr(message.author, "display_name", None) or getattr(message.author, "name", "")
            display_terms = extract_terms(user_display)
            # Deduplicate while preserving order
            terms = list(dict.fromkeys(base_terms + display_terms + [f"user:{user_id}", f"channel:{channel_id}", "memory"]))
            res = await self.memory.query_relevant(
                terms=terms,
                context_hint=context_hint,
            )
            retrieved_items = res.get("items", []) or []
            # Metrics
            retrieve_limit = max(0, int(self.cfg_state.read().memory_service.retrieve_limit))
            used_count = min(len(retrieved_items), retrieve_limit)
            self.logger.info(
                "memory_retrieved",
                retrieved_count=len(retrieved_items),
                used_count=used_count,
                took_ms=res.get("took_ms") if isinstance(res, dict) else None,
                terms_preview=terms[:6],
            )
        except Exception as e:
            self.logger.warning("memory_retrieval_error", error=str(e))

        if not self.client:
            await message.channel.send("LLM client not initialized.")
            return

        # Build final prompt messages using PromptBuilder (identity + memory + conversation)
        final_messages = build_prompt_messages(
            trigger_msg=message,
            cfg=cfg,
            memory_items=retrieved_items,
            conversation_messages=llm_messages,
        )

        out_text = await stream_completion_and_send(self.client, cfg, message.channel, final_messages)

        # Automatic memory extraction and upsert (Story 2.3)
        try:
            provenance = {
                "guild_id": (str(message.guild.id) if message.guild else None),
                "channel_id": str(message.channel.id),
                "user_id": str(message.author.id),
                "message_id": str(message.id),
                "direction": "in",
                "source": "discord",
            }
            # Basic entity for user and relation to channel
            # Add truncation indicator if input was sliced
            _raw_content = message.content or ""
            _text_slice = _raw_content[:500]
            if _raw_content and len(_raw_content) > 500:
                _text_slice = f"{_text_slice} [truncated]"
            entities = [
                entity(
                    name=f"user:{message.author.id}",
                    entity_type="user",
                    observations=(([_text_slice] if _raw_content else []) + [f"display_name:{getattr(message.author, 'display_name', None) or getattr(message.author, 'name', '')}"]),
                )
            ]
            relations = [
                relation(
                    from_name=f"user:{message.author.id}",
                    to_name=f"channel:{message.channel.id}",
                    relation_type="messaged_in",
                )
            ]

            # Extract explicit "remember/record/note (that) X..." requests into a normalized memory observation
            import re  # local import to avoid polluting module scope
            clean_content = (message.content or "")
            # Remove direct mention tokens to avoid confusing the extractor
            if raw_id_tokens:
                for tok in raw_id_tokens:
                    clean_content = clean_content.replace(tok, "").strip()
            remember_match = re.search(r"\b(?:remember|record|note)\b(?:\s+that)?\s+(?P<fact>.+)", clean_content, re.IGNORECASE)
            observations_list = None
            if remember_match:
                fact = remember_match.group("fact").strip()[:500]
                mem_entity_name = f"memory:{message.guild.id}" if message.guild else "memory"
                observations_list = [observation(entity_name=mem_entity_name, contents=[fact])]

            # Fire and forget upsert for inbound (include any extracted memory observation)
            asyncio.create_task(self.memory.upsert_memory(
                entities=entities,
                relations=relations,
                observations=observations_list,
                provenance=provenance,
                context=context_hint,
            ))

            # Also record bot's outbound observation when available
            if out_text:
                provenance_out = dict(provenance)
                provenance_out["direction"] = "out"
                obs = [
                    observation(
                        entity_name=(f"bot:{self.user.id}" if self.user else "bot"),
                        contents=[(out_text[:500] + (" [truncated]" if len(out_text) > 500 else ""))],
                    )
                ]
                asyncio.create_task(self.memory.upsert_memory(
                    entities=None,
                    relations=None,
                    observations=obs,
                    provenance=provenance_out,
                    context=context_hint,
                ))
        except Exception as e:
            self.logger.warning("memory_upsert_error", error=str(e))

    async def _warn_permissions(self, message: Message, reason: str):
        text = f"Permission check did not pass ({reason}). If you believe this is an error, ask the admin to update config.yaml."
        try:
            await message.channel.send(text)
        except Exception:
            pass

    # Slash Commands
    @app_commands.command(name="model", description="View the active LLM model (view-only)")
    async def model_cmd(self, interaction: discord.Interaction):
        cfg = self.cfg_state.read()
        preamble = build_preamble(cfg)
        await interaction.response.send_message(f"{preamble}Active model is set by the administrator and cannot be changed by users.")

# ---------------------------
# Entrypoint
# ---------------------------

def ensure_tokens(cfg: Config) -> Tuple[str, str]:
    discord_token = os.getenv(cfg.discord.token_env)
    api_key = os.getenv(cfg.llm.api_key_env)

    if not discord_token:
        structlog.get_logger().error("missing_discord_token", env_var=cfg.discord.token_env)
        raise SystemExit(
            f"Missing Discord token. Provide environment variable {cfg.discord.token_env}. "
            "The project owner indicated they will provide the Discord API token when requested."
        )
    if not api_key:
        structlog.get_logger().error("missing_openai_key", env_var=cfg.llm.api_key_env)
        raise SystemExit(
            f"Missing LLM API key. Provide environment variable {cfg.llm.api_key_env}. "
            "The project owner indicated they will provide the OpenAI-compatible API key when requested."
        )
    return discord_token, api_key


def main():
    # Initial config load (logging configuration requires config)
    cfg_state = ConfigState(DEFAULT_CONFIG_PATH)
    setup_logging(cfg_state.read().logging)
    logger = structlog.get_logger()

    # Hot reload watcher
    stop_event = threading.Event()
    watcher = threading.Thread(target=cfg_state.watch, args=(stop_event,), daemon=True)
    watcher.start()

    # Instantiate bot
    bot = LlmCord(cfg_state=cfg_state, command_prefix="!")

    # Graceful shutdown
    loop = asyncio.get_event_loop()

    def _signal_handler(sig, frame):
        logger.info("shutdown_signal", signal=str(sig))
        stop_event.set()
        try:
            loop.create_task(bot.close())
        except Exception:
            pass

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Ensure tokens present (surface request to user/admin if missing)
    try:
        discord_token, _ = ensure_tokens(cfg_state.read())
    except SystemExit as e:
        logger.error("startup_missing_secret", detail=str(e))
        sys.exit(1)

    logger.info("starting_bot")
    bot.run(discord_token)


if __name__ == "__main__":
    main()