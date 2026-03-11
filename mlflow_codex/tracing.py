import json
import os
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager

from .config import CodexTraceConfig

NANOSECONDS_PER_MS = 1_000_000
NANOSECONDS_PER_S = 1_000_000_000
MAX_PREVIEW_LENGTH = 1000


def setup_tracer(config: CodexTraceConfig) -> None:
    # Keep signature for CLI compatibility. We use MLflow native tracing APIs.
    tracking_uri = config.endpoint
    if "/api/2.0/otel/v1/traces" in tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    elif tracking_uri.endswith("/v1/traces"):
        tracking_uri = tracking_uri[: -len("/v1/traces")]
    mlflow.set_tracking_uri(tracking_uri)

    local_experiment_id = config.headers.get("x-mlflow-experiment-id")
    is_local_tracking = tracking_uri.startswith("http://127.0.0.1:") or tracking_uri.startswith(
        "http://localhost:"
    )
    if is_local_tracking and local_experiment_id:
        mlflow.set_experiment(experiment_id=local_experiment_id)
        return

    experiment_id = os.getenv(MLFLOW_EXPERIMENT_ID.name, "").strip()
    experiment_name = os.getenv(MLFLOW_EXPERIMENT_NAME.name, "").strip()
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    elif experiment_name:
        mlflow.set_experiment(experiment_name=experiment_name)


def _safe_text(value: Any, max_len: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_len]
    try:
        return json.dumps(value, ensure_ascii=False)[:max_len]
    except Exception:
        return str(value)[:max_len]


def _parse_events(raw_stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (raw_stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _event_ns(event: dict[str, Any]) -> int | None:
    for key in ("timestamp_ns", "ts_ns"):
        val = event.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    for key in ("timestamp_ms", "ts_ms"):
        val = event.get(key)
        if isinstance(val, (int, float)):
            return int(val * NANOSECONDS_PER_MS)
    for key in ("timestamp", "time", "created_at"):
        raw = event.get(key)
        if isinstance(raw, str) and raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return int(dt.timestamp() * NANOSECONDS_PER_S)
            except ValueError:
                continue
    return None


def _extract_session_turn_ids(events: list[dict[str, Any]]) -> tuple[str, str]:
    session_id = ""
    turn_id = ""
    for event in events:
        event_type = _safe_text(event.get("type"), 80)
        if event_type == "thread.started" and not session_id:
            session_id = _safe_text(event.get("thread_id"), 120)
        if event_type == "turn.started" and not turn_id:
            turn_id = _safe_text(event.get("turn_id") or event.get("id"), 120)
    return session_id, turn_id


def _normalize_error(error_text: str, exit_code: int | None, return_code: int | None) -> str:
    if error_text:
        lowered = error_text.lower()
        if "timeout" in lowered:
            return "timeout"
        if "forbidden" in lowered or "permission" in lowered or "unauthorized" in lowered:
            return "permission"
        if "not found" in lowered or "no such file" in lowered:
            return "not_found"
        return "runtime_error"
    rc = return_code if return_code is not None else exit_code
    if rc in (None, 0):
        return "none"
    return "process_exit_nonzero"


def _read_codex_config() -> dict[str, Any]:
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return {}
    try:
        return tomllib.loads(config_path.read_text())
    except Exception:
        return {}


def _extract_config_runtime_info(cfg: dict[str, Any]) -> dict[str, Any]:
    model = _safe_text(cfg.get("model"), 120)
    reasoning_effort = _safe_text(cfg.get("model_reasoning_effort"), 120)
    model_display = f"{model} ({reasoning_effort or 'unknown'})" if model else ""
    mcp_servers_cfg = cfg.get("mcp_servers", {})
    skills_cfg = cfg.get("skills", {})

    mcp_servers_enabled: list[str] = []
    if isinstance(mcp_servers_cfg, dict):
        for name, val in mcp_servers_cfg.items():
            if isinstance(val, dict) and val.get("enabled") is True:
                mcp_servers_enabled.append(name)

    skills_enabled: list[str] = []
    if isinstance(skills_cfg, dict):
        configs = skills_cfg.get("config", [])
        if isinstance(configs, list):
            for item in configs:
                if isinstance(item, dict) and item.get("enabled") is True:
                    path = _safe_text(item.get("path"), 300)
                    skills_enabled.append(path)

    skills_installed_sources: list[dict[str, str]] = []
    skill_name_set: set[str] = set()
    scan_roots = [
        ("codex", Path.home() / ".codex" / "skills"),
        ("agents", Path.home() / ".agents" / "skills"),
    ]
    for source, skills_dir in scan_roots:
        if skills_dir.exists() and skills_dir.is_dir():
            for child in skills_dir.iterdir():
                if child.is_dir():
                    skill_name = child.name
                    skill_name_set.add(skill_name)
                    skills_installed_sources.append({"name": skill_name, "source": source})

    skills_installed = sorted(skill_name_set)
    skills_effective = sorted(set(skills_installed) | set(skills_enabled))

    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_display": model_display,
        "mcp_servers_enabled": sorted(mcp_servers_enabled),
        "skills_enabled": skills_enabled,
        "skills_installed": skills_installed,
        "skills_installed_sources": sorted(
            skills_installed_sources, key=lambda x: (x["name"], x["source"])
        ),
        "skills_effective": skills_effective,
    }


def _resolve_effective_model(
    events: list[dict[str, Any]], cli_model: str | None, config_model: str
) -> str:
    # Prefer model reported by runtime events if available.
    for event in events:
        model = _safe_text(event.get("model"), 160)
        if model:
            return model
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_model = _safe_text(item.get("model"), 160)
        if item_model:
            return item_model
    # Then use explicit CLI override used for this execution.
    if cli_model:
        return _safe_text(cli_model, 160)
    # Finally fallback to config.toml default.
    return _safe_text(config_model, 160)


def _extract_turn_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    usage = {}
    for event in reversed(events):
        if _safe_text(event.get("type"), 80) == "turn.completed":
            raw = event.get("usage")
            if isinstance(raw, dict):
                usage = raw
                break
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = input_tokens + output_tokens
    token_count = total_tokens + cached_input_tokens
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "token_count": token_count,
    }


def _usage_for_chat_attribute(usage: dict[str, int]) -> dict[str, int]:
    return {
        TokenUsageKey.INPUT_TOKENS: int(usage.get("input_tokens", 0)),
        TokenUsageKey.OUTPUT_TOKENS: int(usage.get("output_tokens", 0)),
        TokenUsageKey.TOTAL_TOKENS: int(usage.get("token_count", 0)),
    }


def _first_last_event_ns(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    timestamps = [ts for e in events if (ts := _event_ns(e)) is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _is_tool_item(item: dict[str, Any], event_type: str) -> bool:
    item_type = _safe_text(item.get("type"), 120).lower()
    if item_type in ("command_execution", "tool_call", "tool_result", "function_call", "function_result"):
        return True
    if item_type.startswith("mcp") or "mcp" in item_type:
        return True
    if "tool" in item_type:
        return True
    if _safe_text(item.get("tool_name"), 120):
        return True
    if _safe_text(item.get("function_name"), 120):
        return True
    if _safe_text(item.get("command"), 120):
        return True
    if event_type.endswith("item.failed") and (
        item.get("error") is not None or item.get("exit_code") is not None
    ):
        return True
    return False


def _extract_tool_name(item: dict[str, Any]) -> str:
    name = _safe_text(
        item.get("tool_name")
        or item.get("name")
        or item.get("function_name")
        or item.get("command_name")
        or item.get("mcp_tool")
        or "",
        160,
    )
    if name:
        return name
    item_type = _safe_text(item.get("type"), 120).lower()
    if item_type:
        return item_type
    return "unknown_tool"


def _extract_tool_input(start_item: dict[str, Any], end_item: dict[str, Any], turn_id: str) -> dict[str, Any]:
    payload = (
        end_item.get("input")
        or end_item.get("arguments")
        or end_item.get("params")
        or end_item.get("request")
        or end_item.get("payload")
        or start_item.get("input")
        or start_item.get("arguments")
        or start_item.get("params")
        or start_item.get("request")
        or start_item.get("payload")
    )
    if payload is None:
        payload = {}
    command = end_item.get("command") or start_item.get("command")
    if command is not None and isinstance(payload, dict):
        payload = {**payload, "command": command}
    return {"turn_id": turn_id, "input": payload}


def _extract_tool_output(end_item: dict[str, Any], start_item: dict[str, Any]) -> dict[str, Any]:
    result = (
        end_item.get("output")
        or end_item.get("result")
        or end_item.get("response")
        or end_item.get("content")
        or end_item.get("aggregated_output")
        or start_item.get("output")
        or start_item.get("result")
    )
    exit_code = end_item.get("exit_code")
    error = end_item.get("error") or start_item.get("error")
    status = _safe_text(end_item.get("status") or start_item.get("status"), 80)
    return {"result": result, "exit_code": exit_code, "status": status, "error": error}


def _build_turn_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    started_map: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    for event in events:
        item = event.get("item") if isinstance(event.get("item"), dict) else None
        if not item:
            continue
        item_id = _safe_text(item.get("id"), 120)
        if not item_id:
            continue
        event_type = _safe_text(event.get("type"), 120)
        if event_type.endswith("item.started"):
            started_map[item_id] = {"event": event, "item": item}
        elif event_type.endswith("item.completed") or event_type.endswith("item.failed"):
            started = started_map.get(item_id)
            completed.append({"started": started, "completed": {"event": event, "item": item}})
    return completed


def _extract_runtime_loaded_skills(events: list[dict[str, Any]]) -> list[str]:
    loaded: set[str] = set()
    for event in events:
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        text_blob = " ".join(
            [
                _safe_text(item.get("name"), 200),
                _safe_text(item.get("type"), 120),
                _safe_text(item.get("text"), 500),
                _safe_text(item.get("command"), 500),
                _safe_text(item.get("aggregated_output"), 500),
            ]
        ).lower()
        if "skill" in text_blob:
            # Keep a compact marker instead of dumping long content.
            loaded.add(_safe_text(item.get("name") or item.get("type") or "skill_hit", 120))
    return sorted(x for x in loaded if x)


def _create_child_spans(
    parent_span,
    pairs: list[dict[str, Any]],
    turn_id: str,
    usage: dict[str, int],
    model_display: str,
) -> dict[str, int]:
    counts = {"tool": 0, "reasoning": 0, "response": 0}

    for pair in pairs:
        started = pair.get("started") or {}
        completed = pair["completed"]
        start_event = started.get("event") if isinstance(started, dict) else None
        end_event = completed["event"]
        start_item = started.get("item") if isinstance(started, dict) else {}
        end_item = completed["item"]

        start_ns = _event_ns(start_event) if isinstance(start_event, dict) else None
        end_ns = _event_ns(end_event) if isinstance(end_event, dict) else None
        item_type = _safe_text(end_item.get("type"), 80).lower()
        item_id = _safe_text(end_item.get("id"), 120) or "unknown"
        event_type = _safe_text(end_event.get("type"), 120)

        if _is_tool_item(end_item, event_type):
            tool_name = _extract_tool_name(end_item)
            tool_inputs = _extract_tool_input(start_item, end_item, turn_id)
            tool_outputs = _extract_tool_output(end_item, start_item)
            exit_code = tool_outputs.get("exit_code")
            err_text = _safe_text(tool_outputs.get("error"), 500)
            err_type = _normalize_error(
                err_text,
                exit_code if isinstance(exit_code, int) else None,
                None,
            )
            span = mlflow.start_span_no_context(
                name=f"tool.{tool_name}",
                parent_span=parent_span,
                span_type=SpanType.TOOL,
                start_time_ns=start_ns,
                inputs={**tool_inputs, "tool_name": tool_name, "raw_item": end_item},
                attributes={
                    "item.id": item_id,
                    "item.event_type": event_type,
                    "tool_name": tool_name,
                    "tool_status": _safe_text(tool_outputs.get("status"), 80),
                    "tool_error_type": err_type,
                },
            )
            span.set_outputs({**tool_outputs, "raw_output": end_item})
            span.end(end_time_ns=end_ns)
            counts["tool"] += 1
            continue

        if item_type in ("reasoning", "agent_message"):
            text = _safe_text(end_item.get("text"), 4000)
            span_name = "reasoning" if item_type == "reasoning" else "llm"
            span = mlflow.start_span_no_context(
                name=span_name,
                parent_span=parent_span,
                span_type=SpanType.LLM,
                start_time_ns=start_ns,
                inputs={
                    "turn_id": turn_id,
                    "item_type": item_type,
                    "model": model_display,
                },
                attributes={
                    "item.id": item_id,
                    "item.event_type": event_type,
                    "model": model_display,
                    "model_display": model_display,
                },
            )
            span.set_outputs({"text": text, "raw_item": end_item})
            span.end(end_time_ns=end_ns)
            if item_type == "reasoning":
                counts["reasoning"] += 1
            else:
                counts["response"] += 1

    return counts


def run_codex_exec(prompt: str, model: str | None = None) -> dict[str, Any]:
    cmd = ["codex", "exec", "--json"]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    started_at_ns = time.time_ns()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    ended_at_ns = time.time_ns()
    latency_ms = int((ended_at_ns - started_at_ns) / NANOSECONDS_PER_MS)
    events = _parse_events(proc.stdout or "")
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "latency_ms": latency_ms,
        "started_at_ns": started_at_ns,
        "ended_at_ns": ended_at_ns,
        "events": events,
    }


def _find_final_response_text(pairs: list[dict[str, Any]]) -> str:
    for pair in reversed(pairs):
        end_item = pair["completed"]["item"]
        if _safe_text(end_item.get("type"), 80).lower() == "agent_message":
            text = _safe_text(end_item.get("text"), 4000)
            if text.strip():
                return text
    return ""


def _trace_codex_result(
    prompt: str,
    case_id: str,
    model: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    codex_cfg = _read_codex_config()
    runtime_info = _extract_config_runtime_info(codex_cfg)

    events = result["events"]
    effective_model = _resolve_effective_model(events, model, runtime_info["model"])
    runtime_info["model"] = effective_model
    runtime_info["model_display"] = (
        f"{effective_model} ({runtime_info['reasoning_effort'] or 'unknown'})"
        if effective_model
        else ""
    )
    pairs = _build_turn_items(events)
    runtime_loaded_skills = _extract_runtime_loaded_skills(events)
    usage = _extract_turn_usage(events)
    event_start_ns, event_end_ns = _first_last_event_ns(events)
    run_start_ns = event_start_ns or result["started_at_ns"]
    run_end_ns = event_end_ns or result["ended_at_ns"]
    session_id, turn_id = _extract_session_turn_ids(events)
    err_text = _safe_text(result["stderr"], 500)
    err_type = _normalize_error(
        err_text,
        None,
        result["returncode"] if isinstance(result["returncode"], int) else None,
    )
    turn_id = turn_id or "turn-unknown"

    root_span = mlflow.start_span_no_context(
        name="codex_conversation",
        span_type=SpanType.AGENT,
        start_time_ns=run_start_ns,
        inputs={"prompt": prompt, "case_id": case_id},
        attributes={
            "session_id": session_id,
            "turn_id": turn_id,
            "error_type": err_type,
            "model": runtime_info["model"],
            "reasoning_effort": runtime_info["reasoning_effort"],
            "model_display": runtime_info["model_display"],
            "mcp_servers_enabled": ",".join(runtime_info["mcp_servers_enabled"]),
            "skills_enabled": json.dumps(runtime_info["skills_enabled"], ensure_ascii=False),
            "skills_installed": json.dumps(runtime_info["skills_installed"], ensure_ascii=False),
            "skills_effective": json.dumps(runtime_info["skills_effective"], ensure_ascii=False),
            "skills_installed_sources": json.dumps(
                runtime_info["skills_installed_sources"], ensure_ascii=False
            ),
            "skills_runtime_loaded": json.dumps(runtime_loaded_skills, ensure_ascii=False),
        },
    )

    turn_span = mlflow.start_span_no_context(
        name="agent_turn",
        parent_span=root_span,
        span_type=SpanType.CHAIN,
        start_time_ns=run_start_ns,
        inputs={"event_count": len(events), "turn_id": turn_id},
    )

    counts = _create_child_spans(
        turn_span, pairs, turn_id, usage, runtime_info["model_display"]
    )
    turn_span.set_outputs(
        {
            "tool_count": counts["tool"],
            "reasoning_count": counts["reasoning"],
            "response_count": counts["response"],
            "usage": usage,
            "token_count": usage["token_count"],
        }
    )
    turn_span.set_attribute("usage.input_tokens", usage["input_tokens"])
    turn_span.set_attribute("usage.cached_input_tokens", usage["cached_input_tokens"])
    turn_span.set_attribute("usage.output_tokens", usage["output_tokens"])
    turn_span.set_attribute("usage.total_tokens", usage["total_tokens"])
    turn_span.set_attribute("usage.token_count", usage["token_count"])
    turn_span.end(end_time_ns=run_end_ns)

    final_response = _find_final_response_text(pairs)
    root_span.set_outputs(
        {
            "status": "completed" if result["returncode"] == 0 else "error",
            "response": final_response,
            "stderr": err_text,
            "model": runtime_info["model"],
            "reasoning_effort": runtime_info["reasoning_effort"],
            "model_display": runtime_info["model_display"],
            "usage": usage,
            "token_count": usage["token_count"],
        }
    )
    root_span.set_attribute("usage.input_tokens", usage["input_tokens"])
    root_span.set_attribute("usage.cached_input_tokens", usage["cached_input_tokens"])
    root_span.set_attribute("usage.output_tokens", usage["output_tokens"])
    root_span.set_attribute("usage.total_tokens", usage["total_tokens"])
    root_span.set_attribute("usage.token_count", usage["token_count"])
    root_span.set_attribute(SpanAttributeKey.CHAT_USAGE, _usage_for_chat_attribute(usage))
    root_span.end(end_time_ns=run_end_ns)

    try:
        with InMemoryTraceManager.get_instance().get_trace(root_span.trace_id) as in_memory_trace:
            in_memory_trace.info.request_preview = prompt[:MAX_PREVIEW_LENGTH]
            if final_response:
                in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
            in_memory_trace.info.trace_metadata = {
                **in_memory_trace.info.trace_metadata,
                TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                TraceMetadataKey.TRACE_SESSION: session_id,
                "mlflow.codex.model": runtime_info["model"],
                "mlflow.codex.reasoning_effort": runtime_info["reasoning_effort"],
                "mlflow.codex.model_display": runtime_info["model_display"],
                "mlflow.codex.mcp_servers_enabled": json.dumps(
                    runtime_info["mcp_servers_enabled"], ensure_ascii=False
                ),
                "mlflow.codex.skills_enabled": json.dumps(
                    runtime_info["skills_enabled"], ensure_ascii=False
                ),
                "mlflow.codex.skills_installed": json.dumps(
                    runtime_info["skills_installed"], ensure_ascii=False
                ),
                "mlflow.codex.skills_effective": json.dumps(
                    runtime_info["skills_effective"], ensure_ascii=False
                ),
                "mlflow.codex.skills_installed_sources": json.dumps(
                    runtime_info["skills_installed_sources"], ensure_ascii=False
                ),
                "mlflow.codex.skills_runtime_loaded": json.dumps(
                    runtime_loaded_skills, ensure_ascii=False
                ),
                TraceMetadataKey.TOKEN_USAGE: json.dumps(
                    {
                        TokenUsageKey.INPUT_TOKENS: usage["input_tokens"],
                        TokenUsageKey.OUTPUT_TOKENS: usage["output_tokens"],
                        TokenUsageKey.TOTAL_TOKENS: usage["token_count"],
                    }
                ),
            }
    except Exception:
        pass

    return {
        "case_id": case_id,
        "trace_id": root_span.trace_id,
        "returncode": result["returncode"],
        "latency_ms": result["latency_ms"],
        "event_count": len(events),
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "token_count": usage["token_count"],
        "model": runtime_info["model"],
        "reasoning_effort": runtime_info["reasoning_effort"],
        "model_display": runtime_info["model_display"],
        "mcp_servers_enabled": runtime_info["mcp_servers_enabled"],
        "skills_enabled": runtime_info["skills_enabled"],
        "skills_installed": runtime_info["skills_installed"],
        "skills_installed_sources": runtime_info["skills_installed_sources"],
        "skills_effective": runtime_info["skills_effective"],
        "skills_runtime_loaded": runtime_loaded_skills,
        "tool_span_count": counts["tool"],
        "reasoning_span_count": counts["reasoning"],
        "agent_message_span_count": counts["response"],
        "session_id": session_id,
        "turn_id": turn_id,
    }


def trace_codex_prompt(
    _tracer: None, prompt: str, case_id: str, model: str | None = None
) -> dict[str, Any]:
    result = run_codex_exec(prompt, model=model)
    return _trace_codex_result(prompt=prompt, case_id=case_id, model=model, result=result)


def trace_codex_completed_process(
    *,
    prompt: str,
    case_id: str,
    model: str | None,
    returncode: int,
    stdout: str,
    stderr: str,
    started_at_ns: int,
    ended_at_ns: int,
) -> dict[str, Any]:
    result = {
        "returncode": returncode,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "latency_ms": int((ended_at_ns - started_at_ns) / NANOSECONDS_PER_MS),
        "started_at_ns": started_at_ns,
        "ended_at_ns": ended_at_ns,
        "events": _parse_events(stdout or ""),
    }
    return _trace_codex_result(prompt=prompt, case_id=case_id, model=model, result=result)
