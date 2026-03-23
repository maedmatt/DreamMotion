# Development

## Setup

```bash
uv sync --all-extras
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Linting and type checking

```bash
make          # uv sync + ruff + basedpyright
make lint     # ruff + basedpyright only
```

The lint pipeline runs:

1. `ruff check --fix` — linting with auto-fix
2. `ruff format` — code formatting
3. `basedpyright` — type checking

### Configuration

- **Ruff**: configured in `pyproject.toml` under `[tool.ruff]`, targeting Python 3.11
- **basedpyright**: configured in `pyproject.toml` under `[tool.basedpyright]`, standard mode

### Pre-commit hooks

The `.pre-commit-config.yaml` runs:

- `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending` (LF)
- The full lint pipeline (`ruff` + `basedpyright`)

## Project layout

```
src/            # all source code
devtools/       # lint.py
docs/           # documentation
output/         # generated motion files (gitignored)
```

All source lives under `src/` and is packaged via hatchling. Entry points (`agent`, `web`, `deploy`, `robot-audio-test`) are defined in `pyproject.toml` under `[project.scripts]`.
