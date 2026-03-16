import json
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from tqdm import tqdm

from codex_session_ingest import ingest_session_by_id


@dataclass
class BatchEvalConfig:
    input_file: str | None = None
    dataset_run_id: str | None = None
    dataset_artifact_path: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    run_name: str = "codex-batch-eval"
    max_workers: int = 4
    keep_existing_session_traces: bool = False
    codex_model: str | None = None
    codex_reasoning_effort: str | None = None
    codex_cd: str | None = None
    codex_config_overrides: list[str] | None = None
    codex_skip_git_repo_check: bool = False
    dotenv_path: str = ".env"
    output_path: str | None = None
    log_results_artifact: bool = True

    def validate(self) -> None:
        if bool(self.input_file) == bool(self.dataset_run_id):
            raise RuntimeError("Provide exactly one of input_file or dataset_run_id.")
        if self.dataset_run_id and not self.dataset_artifact_path:
            raise RuntimeError("dataset_artifact_path is required when dataset_run_id is provided.")
        if self.max_workers <= 0:
            raise RuntimeError("max_workers must be > 0.")
        if self.codex_reasoning_effort and self.codex_reasoning_effort not in {"low", "medium", "high"}:
            raise RuntimeError("codex_reasoning_effort must be one of: low, medium, high")


def load_config_from_yaml(path: str) -> BatchEvalConfig:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"Config file not found: {src}")
    raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("YAML config root must be an object.")
    cfg = BatchEvalConfig(
        input_file=raw.get("input_file"),
        dataset_run_id=raw.get("dataset_run_id"),
        dataset_artifact_path=raw.get("dataset_artifact_path"),
        experiment_id=raw.get("experiment_id"),
        experiment_name=raw.get("experiment_name"),
        run_name=str(raw.get("run_name", "codex-batch-eval")),
        max_workers=int(raw.get("max_workers", 4)),
        keep_existing_session_traces=bool(raw.get("keep_existing_session_traces", False)),
        codex_model=raw.get("codex_model"),
        codex_reasoning_effort=raw.get("codex_reasoning_effort"),
        codex_cd=raw.get("codex_cd"),
        codex_config_overrides=raw.get("codex_config_overrides"),
        codex_skip_git_repo_check=bool(raw.get("codex_skip_git_repo_check", False)),
        dotenv_path=str(raw.get("dotenv_path", ".env")),
        output_path=str(raw["output_path"]) if raw.get("output_path") else None,
        log_results_artifact=bool(raw.get("log_results_artifact", True)),
    )
    cfg.validate()
    return cfg


def _build_distinct_run_name(base: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{stamp}-{suffix}"


def _load_cases_from_file(path: str) -> list[dict[str, Any]]:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"Input file not found: {src}")

    suffix = src.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(src).to_dict(orient="records")
    elif suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        data = rows
    elif suffix == ".json":
        raw = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            data = raw["data"]
        else:
            raise RuntimeError("JSON must be a list or an object with key `data` as list.")
    else:
        raise RuntimeError(f"Unsupported file type: {suffix}. Use csv/json/jsonl.")

    if not data:
        raise RuntimeError("No cases found in input file.")
    return data


def _normalize_cases(raw_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    norm: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_cases):
        if not isinstance(row, dict):
            raise RuntimeError(f"Case at index {idx} must be a JSON object.")
        if "inputs" not in row or "expectations" not in row:
            raise RuntimeError(
                f"Case at index {idx} missing required keys: `inputs` and/or `expectations`."
            )
        case_id = str(row.get("case_id") or f"case-{idx+1:04d}")
        user_input = row.get("inputs")
        expectation = row.get("expectations")
        norm.append(
            {
                "case_id": case_id,
                "inputs": user_input if isinstance(user_input, str) else json.dumps(user_input, ensure_ascii=False),
                "expectations": expectation
                if isinstance(expectation, str)
                else json.dumps(expectation, ensure_ascii=False),
            }
        )
    return norm


def _extract_session_id_from_codex_stdout(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        et = str(event.get("type") or "")
        if et == "thread.started":
            sid = str(event.get("thread_id") or "")
            if sid:
                return sid
        if et == "session_meta":
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            sid = str(payload.get("id") or "")
            if sid:
                return sid
        sid = str(event.get("session_id") or "")
        if sid:
            return sid
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        sid = str(payload.get("session_id") or payload.get("thread_id") or "")
        if sid:
            return sid
    return ""


def _build_codex_exec_cmd(prompt: str, cfg: BatchEvalConfig) -> list[str]:
    cmd = ["codex", "exec", "--json"]
    if cfg.codex_model:
        cmd.extend(["-m", cfg.codex_model])
    if cfg.codex_reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{cfg.codex_reasoning_effort}"'])
    if cfg.codex_config_overrides:
        for item in cfg.codex_config_overrides:
            cmd.extend(["-c", str(item)])
    if cfg.codex_cd:
        cwd_value = os.getcwd() if cfg.codex_cd == "pwd" else cfg.codex_cd
        cmd.extend(["-C", str(cwd_value)])
    if cfg.codex_skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    cmd.append(prompt)
    return cmd


def _run_one_case(case: dict[str, Any], cfg: BatchEvalConfig) -> dict[str, Any]:
    prompt = str(case["inputs"])
    cmd = _build_codex_exec_cmd(prompt, cfg)
    started = time.time_ns()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    ended = time.time_ns()
    merged = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    expectation = str(case["expectations"]).lower()
    auto_pass = expectation in merged if expectation else proc.returncode == 0
    return {
        "case_id": case["case_id"],
        "prompt": prompt,
        "expectations": case["expectations"],
        "returncode": int(proc.returncode),
        "latency_ms": int((ended - started) / 1_000_000),
        "stdout_preview": (proc.stdout or "")[:1000],
        "stderr_preview": (proc.stderr or "")[:1000],
        "session_id": _extract_session_id_from_codex_stdout(proc.stdout or ""),
        "auto_pass": bool(auto_pass),
        "command": cmd,
    }


def _download_dataset_artifact(dataset_run_id: str, artifact_path: str) -> str:
    client = mlflow.MlflowClient()
    dst_dir = tempfile.mkdtemp(prefix="mlflow_dataset_")
    return client.download_artifacts(dataset_run_id, artifact_path, dst_path=dst_dir)


def _load_cases_from_config(cfg: BatchEvalConfig) -> tuple[list[dict[str, Any]], str]:
    if cfg.input_file:
        raw = _load_cases_from_file(cfg.input_file)
        return _normalize_cases(raw), str(Path(cfg.input_file).expanduser().resolve())
    downloaded = _download_dataset_artifact(str(cfg.dataset_run_id), str(cfg.dataset_artifact_path))
    raw = _load_cases_from_file(downloaded)
    return _normalize_cases(raw), downloaded


def _setup_experiment(cfg: BatchEvalConfig) -> str:
    if cfg.experiment_id:
        exp = mlflow.set_experiment(experiment_id=cfg.experiment_id)
    elif cfg.experiment_name:
        exp = mlflow.set_experiment(experiment_name=cfg.experiment_name)
    else:
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip() or "codex-batch-eval"
        exp = mlflow.set_experiment(experiment_name=exp_name)
    return exp.experiment_id


def run_batch_eval(cfg: BatchEvalConfig) -> dict[str, Any]:
    cfg.validate()
    load_dotenv(cfg.dotenv_path)
    experiment_id = _setup_experiment(cfg)
    os.environ[MLFLOW_EXPERIMENT_ID.name] = experiment_id

    cases, dataset_source = _load_cases_from_config(cfg)
    dataset_df = pd.DataFrame(cases)
    dataset = mlflow.data.from_pandas(
        dataset_df,
        source=dataset_source,
        name=f"codex_eval_dataset_{datetime.now().strftime('%Y%m%d')}",
    )

    run_name = _build_distinct_run_name(cfg.run_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_input(dataset, context="evaluation")
        mlflow.log_param("case_count", len(cases))
        mlflow.log_param("max_workers", cfg.max_workers)
        if cfg.codex_model:
            mlflow.log_param("codex_model", cfg.codex_model)
        if cfg.codex_reasoning_effort:
            mlflow.log_param("codex_reasoning_effort", cfg.codex_reasoning_effort)
        if cfg.codex_cd:
            mlflow.log_param("codex_cd", cfg.codex_cd)
        if cfg.codex_config_overrides:
            mlflow.log_param(
                "codex_config_overrides",
                json.dumps(cfg.codex_config_overrides, ensure_ascii=False),
            )
        mlflow.log_param("codex_skip_git_repo_check", cfg.codex_skip_git_repo_check)
        if cfg.dataset_run_id:
            mlflow.log_param("dataset_run_id", cfg.dataset_run_id)
            mlflow.log_param("dataset_artifact_path", cfg.dataset_artifact_path)
        else:
            mlflow.log_param("input_file", dataset_source)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
            futures = [pool.submit(_run_one_case, case, cfg) for case in cases]
            with tqdm(total=len(futures), desc="codex eval", unit="case") as pbar:
                for fut in as_completed(futures):
                    results.append(fut.result())
                    pbar.update(1)

        session_ids = sorted(set(r["session_id"] for r in results if r["session_id"]))
        ingest_results: list[dict[str, Any]] = []
        for sid in session_ids:
            ingest_results.append(
                ingest_session_by_id(
                    sid,
                    delete_existing_session=not cfg.keep_existing_session_traces,
                )
            )

        auto_pass_count = sum(1 for r in results if r["auto_pass"])
        auto_pass_rate = auto_pass_count / len(results) if results else 0.0
        avg_latency_ms = sum(int(r["latency_ms"]) for r in results) / len(results) if results else 0.0
        session_found_rate = len(session_ids) / len(results) if results else 0.0

        mlflow.log_metric("avg_latency_ms", avg_latency_ms)

        out = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "case_count": len(results),
            "session_count": len(session_ids),
            "results": results,
            "session_ingest": ingest_results,
        }
        if cfg.log_results_artifact:
            mlflow.log_dict(out, "batch_eval_results.json")
        if cfg.output_path:
            out_path = Path(cfg.output_path).expanduser().resolve()
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def run_batch_eval_from_yaml(config_path: str) -> dict[str, Any]:
    cfg = load_config_from_yaml(config_path)
    return run_batch_eval(cfg)
