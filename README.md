# MLflow Codex Tracing (Minimal)

Minimal tracing utility for `codex exec --json` with MLflow.

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Required env

Use `.env` (or system env) with one of these modes:

- Databricks:
  - `MLFLOW_TRACKING_URI=databricks`
  - `DATABRICKS_HOST`
  - `DATABRICKS_TOKEN`
- Local MLflow:
  - `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`

## Python usage (recommended)

```python
import mlflow_codex as codex

# Enable tracing for subprocess codex calls in this Python process
codex.autolog(patch_subprocess=True)

# Keep your original style
import subprocess
subprocess.run(["codex", "exec", "--json", "幫我從 grafana mcp 確認資源"], text=True, capture_output=True)
```

## Example

Run the provided sample:

```bash
uv run python3 eval.py
```
