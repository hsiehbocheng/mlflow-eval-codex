# Codex Batch Skill Evaluation (Minimal + MLflow)

This is a minimal local benchmark harness to batch-evaluate Codex behavior across skill treatments, inspired by:
- https://github.com/langchain-ai/skills-benchmarks
- https://blog.langchain.com/evaluating-skills/
- https://developers.openai.com/codex/noninteractive

## What this does

- Runs `task x treatment x repetition` in batch.
- Executes each case with `codex exec --json` (no Docker).
- Evaluates each case with a simple evaluator (`file_equals` in this sample).
- Logs per-case metrics/artifacts to MLflow (instead of LangSmith).

## Files

- `evaluate_skills.py`: benchmark runner.
- `benchmark.json`: benchmark/task/treatment config (default).
- `benchmark.yaml`: optional YAML version of the same config.
- `tasks/...`: benchmark task prompts + optional template workspace.
- `treatments/...`: treatment-specific `AGENTS.md` instructions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure `.env` (or environment) contains:
- `OPENAI_API_KEY`
- `MLFLOW_TRACKING_URI` (optional if you pass `--tracking-uri`)
- `MLFLOW_EXPERIMENT_ID` (optional if you pass `--experiment-id`)

Codex auth modes in `benchmark.json`:
- `codex.auth = "login"`: use local `codex login` session (no API key required).
- `codex.auth = "api_key"`: use `CODEX_API_KEY` (or fallback from `OPENAI_API_KEY`).
- `codex.codex_home = "default"` uses your normal `~/.codex`.
- `codex.codex_home = "isolated"` writes state under `runs/.codex-home`.

## Run

Real run:

`benchmark.json` is the default:

```bash
python3 evaluate_skills.py
```

Offline validation run (no API call):

```bash
python3 evaluate_skills.py --dry-run --tracking-uri file:./mlruns
```

Offline flow-only run without MLflow dependency:

```bash
python3 evaluate_skills.py --dry-run --no-mlflow
```

## Notes

- Set `codex.codex_home = "isolated"` if you want Codex state under `runs/.codex-home`; default is your normal `~/.codex`.
- Output summaries are written to `results/summary_*.json` and `results/summary_*.csv`.
