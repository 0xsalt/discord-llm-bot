# discord-llm-bot

A Discord bot that provides a collaborative LLM experience using a single pre-configured model. Docker-only runtime. Core logic resides in a single file: discord-llm-bot.py.

Status
- MVP scope per PRD shards (docs/shards/*.jsonl).
- Docker-only per architecture.
- /model slash command is view-only (no switching).
- Secrets provided via environment variables (see below).

Quick Start (Docker-only)
1) Prepare environment variables (do NOT commit secrets):
   - DISCORD_BOT_TOKEN=<your discord bot token>
   - OPENAI_API_KEY=<your openai-compatible api key>

2) Create config.yaml from the example:
   - cp config-example.yaml config.yaml
   - Adjust values to your needs.

3) Build and run:
   - docker compose up --build

Configuration
- See config-example.yaml for a complete reference.
- memory_service configuration controls the MCP Knowledge Graph integration:
  - Keys: enabled (bool), base_url (string), timeout_s (int), retrieve_limit (int)
  - Example:
    ```
    memory_service:
      enabled: true
      base_url: "http://mcp-server:8080"
      timeout_s: 2
      retrieve_limit: 5
    ```
- A JSON Schema for config is available: docs/config/schema.json (optional validation in tools/CI).

Key Features (MVP)
- Public channel: reply when mentioned (@botname).
- Direct messages: reply to all DMs.
- Reply-chain context: construct conversation history up to a configured depth.
- Streaming responses with chunking to fit Discord limits.
- View-only /model command that returns the active model details.
- Permissions via config.yaml (users/roles/channels allow/deny).
- Hot reload of configuration without restart (file watcher).

Runtime Environment
- CONFIG_PATH=/app/config.yaml (mounted by docker-compose).
- LOG_DIR=/app/logs (mounted directory for logs).
- DISCORD_BOT_TOKEN and OPENAI_API_KEY must be set in the environment.

Development Notes
- Use pytest for tests (run inside container).
- Logging is via structlog; no print statements in production code.
- Every user-facing response should begin with a model identification preamble:
  - name, size, type, revision date (best effort, configurable).

Repository Structure
- .gitignore
- config-example.yaml
- config.yaml (user-created, not tracked)
- docker-compose.yaml
- docker-compose.dev.yaml
- Dockerfile
- LICENSE.md
- discord-llm-bot.py (core application)
- memory_manager.py (memory service client)
- prompt_builder.py (prompt construction)
- README.md
- requirements.txt
- mcp-knowledge-graph/ (MCP server)
  - server.js
- mcp-data/ (runtime data, not tracked)
- logs/ (runtime logs, not tracked)
- docs/
  - shards/ (sharded brief, prd, architecture with index)
  - backlog/engineering.jsonl
  - tests/plan.jsonl
  - config/schema.json

Secrets Policy
- Never store tokens/keys in the repo or shards.
- Provide DISCORD_BOT_TOKEN and OPENAI_API_KEY through environment only.
- If you need to rotate or switch providers, update environment and config.yaml.

Next Steps
- Implement discord-llm-bot.py with:
  - structlog setup and rotation policy.
  - config loader with hot reload.
  - Discord client with intents, handlers, and /model command.
  - LLM client call with streaming and chunking.
  - Permissions gating and helpful warnings.
- Add pytest skeletons aligned with docs/tests/plan.jsonl.


## Prompt Customization

You can customize the bot’s identity and enable/disable system prompt modules without touching code.

- Identity settings (name, addressing aliases, persona) live under `app.identity` in your config.
- Prompt modules are toggled under the top-level `prompts` section.
- System blocks are composed centrally in `prompt_builder.py`.

Example configuration (config.yaml or config-example.yaml):
```yaml
app:
  identity:
    name: "varuna"
    addressing_aliases: ["varuna"]   # add nicknames here to recognize as direct addresses
    persona: ""                      # optional free text to shape tone/behavior

prompts:
  enable_identity: true              # toggle identity header
  enable_memory_context: true        # toggle memory summary block when available
  enable_policy: true                # toggle an optional global Policy block
  policy: |                          # free text; appears as a "Policy:" system block
    Respond concisely. Prefer bullet lists for enumeration.
    Do not reveal internal IDs or secrets.
    When Memory Context is present, leverage it explicitly and avoid repetition.
```

Where these are used:
- Identity header, Policy block, and Memory Context are generated in prompt_builder.py:
  - identity_block(), policy_block(), memory_context_block() are composed by build_prompt_messages().
- Conversation content (user/assistant messages + attachments) is built in discord-llm-bot.py and intentionally excludes system blocks:
  - build_messages_for_llm() produces conversation-only messages.
- on_message collects any Memory Context items and passes them (without mutating the conversation) to PromptBuilder.

Add your own system blocks:
- Extend prompt_builder.py with a new function that returns a system message block (role="system").
- Wire it into build_prompt_messages() behind a new toggle in the `prompts` section of your config.

Operational notes:
- If `prompts.enable_identity` is false, no identity header is added.
- If `prompts.enable_memory_context` is false or no items are retrieved, the Memory Context block is omitted.
- If `prompts.enable_policy` is true and `prompts.policy` is non-empty, a “Policy:” system block is included after Identity.
- Per-guild partitioning for memory retrieval/upserts is the default to keep server knowledge separate.
- MCP data persists in the repository at `mcp-data` (bind-mounted to the server). Ensure host permissions allow UID 1000 to write:
  - `mkdir -p ./mcp-data && sudo chown -R 1000:1000 ./mcp-data && sudo chmod 775 ./mcp-data`
