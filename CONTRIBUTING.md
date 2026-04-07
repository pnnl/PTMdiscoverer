# Contributing

Keep changes focused, reproducible, and aligned with the two main surfaces in this repo:

1. `ptmdiscoverer`: standalone CLI, inference library, and lightweight MCP server.
2. `agentic_framework_pkg`: ADEPT integration layer, Streamlit app, and HTTP MCP server.

## Setup

Core development:

```bash
uv sync --group dev
```

If your change touches ADEPT code, app code, or ADEPT-facing tests:

```bash
uv sync --extra adept --group dev
```

If your change touches the standalone MCP server only:

```bash
uv sync --extra mcp --group dev
```

## Quality Gate

Run before every commit:

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
```

If you changed ADEPT code or dependencies, also run:

```bash
uv run --extra adept pytest -q
```

If you changed generated API docs or public module structure, regenerate:

```bash
uv run --extra adept pdoc -o docs/api ptmdiscoverer agentic_framework_pkg
```

## Change Expectations

1. Add or update tests for any changed behavior.
2. Update README, docs, and docstrings when user-facing behavior or workflows change.
3. Keep top-level files limited to primary project entry points and core metadata.
4. Put examples, auxiliary configs, and reference material under `docs/` unless they must live at repo root.
5. Do not commit secrets or personal environment settings.
6. Avoid unrelated refactors while making a targeted fix.

## MCP Notes

The standalone MCP client example config lives at `docs/examples/mcp_config.example.json`.

Use it as a reference for MCP Inspector or other MCP clients. If you change the standalone MCP launch command or required environment variables, update that file too.

## Pull Requests

Open a PR with:

1. What changed and why in 1 to 3 sentences.
2. Any notable trade-offs or scope limits.
3. Follow-up work, if any.

Small, clear commits are preferred.
