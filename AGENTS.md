# AGENTS.md

## Purpose

This repository exists to discover and improve trading strategies with the goal of maximizing profit.

When evaluating alternatives, prefer implementations that improve strategy performance while keeping results reproducible and code maintainable.

## Scope

These instructions apply to all agent work in this repository.

## Project Map

- Entry scripts:
  - `run_backtest.py`: run one configured backtest and show stats/plot.
  - `run_hyperparameter_search.py`: run parameter search using configured ranges.
  - `alpaca_stream.py`: live stream demo.
- Core modules:
  - `src/config.py`: central configuration values.
  - `src/data.py`: Alpaca historical data loading.
  - `src/strategies/`: strategy implementations.

## Non-Negotiable Runtime Rule

Do not rely on runtime parameters when running scripts.

- Scripts should run as plain commands such as:
  - `python run_backtest.py`
  - `python run_hyperparameter_search.py`
- Operational parameters must be read from dedicated config files.
- If new parameters are needed, add them to config modules/files and consume them from code.
- Do not introduce required CLI flags for normal operation.

## Configuration Policy

- Keep configuration in dedicated config files only.
- Use `src/config.py` as the default location, or split into additional config files if the domain grows (for example strategy config, data config, execution config).
- Avoid hardcoded tunable values in entry scripts and strategy logic.
- Keep defaults explicit and documented in config.

## Coding Standards

Use these as the canonical coding rules for this repository:

- Follow PEP 8.
- Use type hints on function signatures and variable declarations.
- Prefer readable functional patterns (for example list comprehensions/map/filter) over deep nested loops.
- Use f-strings for formatting.
- Prefer `pathlib` over `os.path`.
- Use Google-style docstrings for public modules, classes, and functions.
- Keep comments focused on why, not what.

Tooling and tests:

- Prefer `ruff` for linting/formatting.
- Use `pytest` for tests with Arrange-Act-Assert structure.
- Assume dependency management via `uv` or `poetry` unless task context requires otherwise.

Error handling and safety:

- Never use bare `except`; catch specific exceptions.
- Prefer custom exception classes for domain-specific failures.
- Keep `try-except` scopes narrow around likely failure points.
- For I/O-bound concurrency, prefer `asyncio`.
- Avoid mutable global state.
- Use `secrets` (not `random`) for sensitive randomness.

## Architecture Guidance

- Keep entry scripts thin orchestration layers.
- Put reusable business logic in `src/` modules.
- Add new strategy code under `src/strategies/`.
- Keep data acquisition concerns inside data modules.
- Preserve backward-compatible behavior unless the task explicitly requires change.

## Change Workflow

- Before editing, read relevant files and match existing style.
- Make the smallest safe change that solves the task.
- Do not revert unrelated local changes.
- Avoid destructive git operations unless explicitly requested.
- After edits, run targeted checks/tests when possible.

## Validation Expectations

For strategy or backtest changes:

- Verify scripts still run without runtime params.
- Verify config is the single source of operational values.
- Verify output remains understandable for iterative research.

## Preferred Task Patterns

### Add or tune a strategy

1. Add or modify strategy implementation in `src/strategies/`.
2. Put tunable windows/thresholds in config.
3. Keep entry scripts as orchestrators only.

### Add a new operational setting

1. Add the setting to a dedicated config file.
2. Read it in the module that needs it.
3. Do not require passing it via CLI at runtime.

### Improve observability

- Prefer clear progress/status prints in script flow.
- Keep logs concise and useful for backtest iteration.
