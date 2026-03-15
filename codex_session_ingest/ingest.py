import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey

NANOSECONDS_PER_S = 1_000_000_000


@dataclass
class SessionTurn:
    index: int
    turn_id: str
    events: list[dict[str, Any]]
    user_text: str
    model: str
    reasoning_effort: str
    cwd: str
    approval_policy: str
    sandbox_policy: str
    skills: list[str]


def _safe_text(value: Any, max_len: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_len]
    try:
        return json.dumps(value, ensure_ascii=False)[:max_len]
    except Exception:
        return str(value)[:max_len]


def _parse_iso_ns(raw: Any) -> int | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp() * NANOSECONDS_PER_S)
    except ValueError:
        return None


def _event_ns(event: dict[str, Any]) -> int | None:
    return _parse_iso_ns(event.get("timestamp"))


def _parse_skills_from_user_instructions(text: str) -> list[str]:
    if not text:
        return []
    start = text.find("### Available skills")
    if start < 0:
        return []
    block = text[start:]
    names: list[str] = []
    for line in block.splitlines():
        if line.startswith("### ") and "Available skills" not in line:
            break
        match = re.match(r"-\s+([A-Za-z0-9._-]+)\s*:", line.strip())
        if match:
            names.append(match.group(1))
    return sorted(set(names))


def _is_user_boundary(event: dict[str, Any]) -> tuple[bool, str]:
    et = _safe_text(event.get("type"), 80)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

    if et == "event_msg" and _safe_text(payload.get("type"), 80) == "user_message":
        return True, _safe_text(payload.get("message"), 4000)

    if et == "response_item":
        if _safe_text(payload.get("type"), 80) == "message" and _safe_text(
            payload.get("role"), 30
        ) == "user":
            content = payload.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = _safe_text(item.get("text") or item.get("input_text"), 4000)
                        if text:
                            return True, text
            return True, ""
    return False, ""


def _is_event_msg_user_boundary(event: dict[str, Any]) -> tuple[bool, str]:
    et = _safe_text(event.get("type"), 80)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if et == "event_msg" and _safe_text(payload.get("type"), 80) == "user_message":
        text = _safe_text(payload.get("message"), 4000).strip()
        if text:
            return True, text
    return False, ""


def _is_noise_user_text(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    return normalized.startswith("# agents.md instructions") or normalized.startswith(
        "<environment_context>"
    )


def _event_kind(event: dict[str, Any]) -> str:
    et = _safe_text(event.get("type"), 80)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

    if et == "event_msg":
        ptype = _safe_text(payload.get("type"), 80)
        if ptype == "agent_reasoning":
            return "reasoning"
        if ptype == "agent_message":
            return "assistant_message"
        if ptype == "token_count":
            return "token_usage"
        if ptype == "user_message":
            return "user_message"
        if "tool" in ptype or "mcp" in ptype:
            return "tool"
        return "event"

    if et == "response_item":
        ptype = _safe_text(payload.get("type"), 120).lower()
        role = _safe_text(payload.get("role"), 20).lower()
        if ptype == "reasoning":
            return "reasoning"
        if ptype == "message" and role == "assistant":
            return "assistant_message"
        if ptype == "message" and role == "user":
            return "user_message"
        if (
            "function_call" in ptype
            or "tool" in ptype
            or "mcp" in ptype
            or "command" in ptype
            or "computer_call" in ptype
            or "web_search" in ptype
        ):
            return "tool"
        return "event"

    if et == "turn_context":
        return "turn_context"
    if et == "session_meta":
        return "session_meta"
    return "event"


def _extract_text_preview(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    et = _safe_text(event.get("type"), 80)
    if et == "event_msg":
        text = payload.get("message") or payload.get("text")
        return _safe_text(text, 4000)
    if et == "response_item":
        if _safe_text(payload.get("type"), 80) == "message":
            content = payload.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        val = _safe_text(item.get("text") or item.get("input_text"), 1200)
                        if val:
                            parts.append(val)
                return "\n".join(parts)[:4000]
    return ""


def _load_session_events(session_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in session_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


def _split_turns(events: list[dict[str, Any]]) -> tuple[list[SessionTurn], dict[str, Any]]:
    session_meta = {}
    turns: list[SessionTurn] = []
    current: list[dict[str, Any]] = []
    current_turn_id = ""
    current_user_text = ""
    current_model = ""
    current_effort = ""
    current_cwd = ""
    current_approval = ""
    current_sandbox = ""
    current_skills: list[str] = []
    last_boundary_text = ""
    last_boundary_ts = -1
    turn_idx = 0
    has_event_user_messages = any(_is_event_msg_user_boundary(e)[0] for e in events)

    for event in events:
        et = _safe_text(event.get("type"), 80)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        ts = _event_ns(event) or -1

        if et == "session_meta" and not session_meta:
            session_meta = payload

        if et == "turn_context":
            current_turn_id = _safe_text(payload.get("turn_id"), 120) or current_turn_id
            current_cwd = _safe_text(payload.get("cwd"), 500)
            current_approval = _safe_text(payload.get("approval_policy"), 120)
            current_sandbox = _safe_text(payload.get("sandbox_policy"), 2000)
            current_model = _safe_text(payload.get("model"), 120)
            collab = payload.get("collaboration_mode")
            if isinstance(collab, dict):
                settings = collab.get("settings")
                if isinstance(settings, dict):
                    current_effort = _safe_text(settings.get("reasoning_effort"), 80)
            current_effort = current_effort or _safe_text(payload.get("effort"), 80)
            user_instructions = _safe_text(payload.get("user_instructions"), 50000)
            parsed_skills = _parse_skills_from_user_instructions(user_instructions)
            if parsed_skills:
                current_skills = parsed_skills

        boundary, boundary_text = _is_event_msg_user_boundary(event)
        if not boundary and not has_event_user_messages:
            fallback_boundary, fallback_text = _is_user_boundary(event)
            if fallback_boundary and not _is_noise_user_text(fallback_text):
                boundary, boundary_text = True, fallback_text
        duplicate_boundary = (
            boundary
            and current
            and (
                last_boundary_text == boundary_text
                or not last_boundary_text
                or not boundary_text
            )
            and ts > 0
            and last_boundary_ts > 0
            and abs(ts - last_boundary_ts) <= 2 * NANOSECONDS_PER_S
        )
        if boundary and current and not duplicate_boundary:
            if current_user_text.strip():
                turns.append(
                    SessionTurn(
                        index=turn_idx,
                        turn_id=current_turn_id or f"turn-{turn_idx}",
                        events=current,
                        user_text=current_user_text,
                        model=current_model,
                        reasoning_effort=current_effort,
                        cwd=current_cwd,
                        approval_policy=current_approval,
                        sandbox_policy=current_sandbox,
                        skills=current_skills[:],
                    )
                )
                turn_idx += 1
                current_turn_id = ""
            # Drop preamble events before the first real user message.
            current = []

        if boundary:
            current_user_text = boundary_text
            last_boundary_text = boundary_text
            last_boundary_ts = ts

        current.append(event)

    if current:
        if current_user_text.strip():
            turns.append(
                SessionTurn(
                    index=turn_idx,
                    turn_id=current_turn_id or f"turn-{turn_idx}",
                    events=current,
                    user_text=current_user_text,
                    model=current_model,
                    reasoning_effort=current_effort,
                    cwd=current_cwd,
                    approval_policy=current_approval,
                    sandbox_policy=current_sandbox,
                    skills=current_skills[:],
                )
            )
    return turns, session_meta


def _collect_turn_usage(turn_events: list[dict[str, Any]]) -> dict[str, int]:
    # Legacy fallback: sum per-event last_token_usage within this turn.
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "token_count": 0,
    }
    for event in turn_events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if _safe_text(event.get("type"), 80) != "event_msg" or _safe_text(payload.get("type"), 80) != "token_count":
            continue
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        last = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
        if not last:
            continue
        usage["input_tokens"] += int(last.get("input_tokens", 0) or 0)
        usage["cached_input_tokens"] += int(last.get("cached_input_tokens", 0) or 0)
        usage["output_tokens"] += int(last.get("output_tokens", 0) or 0)
        usage["total_tokens"] += int(
            last.get(
                "total_tokens",
                int(last.get("input_tokens", 0) or 0) + int(last.get("output_tokens", 0) or 0),
            )
            or 0
        )
    usage["token_count"] = usage["total_tokens"] + usage["cached_input_tokens"]
    return usage


def _token_usage_from_raw(raw: dict[str, Any] | None) -> dict[str, int]:
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "token_count": 0,
    }
    if not isinstance(raw, dict):
        return usage
    usage["input_tokens"] = int(raw.get("input_tokens", 0) or 0)
    usage["cached_input_tokens"] = int(raw.get("cached_input_tokens", 0) or 0)
    usage["output_tokens"] = int(raw.get("output_tokens", 0) or 0)
    usage["total_tokens"] = int(
        raw.get("total_tokens", usage["input_tokens"] + usage["output_tokens"]) or 0
    )
    usage["token_count"] = usage["total_tokens"] + usage["cached_input_tokens"]
    return usage


def _subtract_token_usage(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    usage = {
        "input_tokens": max(0, current["input_tokens"] - previous["input_tokens"]),
        "cached_input_tokens": max(
            0, current["cached_input_tokens"] - previous["cached_input_tokens"]
        ),
        "output_tokens": max(0, current["output_tokens"] - previous["output_tokens"]),
        "total_tokens": max(0, current["total_tokens"] - previous["total_tokens"]),
        "token_count": 0,
    }
    usage["token_count"] = usage["total_tokens"] + usage["cached_input_tokens"]
    return usage


def _extract_turn_total_token_usage(turn_events: list[dict[str, Any]]) -> dict[str, int] | None:
    for event in reversed(turn_events):
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if _safe_text(event.get("type"), 80) != "event_msg" or _safe_text(payload.get("type"), 80) != "token_count":
            continue
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        total = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else None
        if total:
            return _token_usage_from_raw(total)
    return None


def _resolve_turn_usage(
    turn_events: list[dict[str, Any]], previous_turn_total: dict[str, int] | None
) -> tuple[dict[str, int], dict[str, int] | None]:
    current_total = _extract_turn_total_token_usage(turn_events)
    if current_total is not None:
        if previous_turn_total is None:
            return current_total, current_total
        return _subtract_token_usage(current_total, previous_turn_total), current_total

    # If total_token_usage is unavailable, fallback to event-level increments.
    fallback = _collect_turn_usage(turn_events)
    return fallback, previous_turn_total


def _resolve_turn_time_window(turn_events: list[dict[str, Any]]) -> tuple[int, int]:
    timestamps = [ts for ts in (_event_ns(e) for e in turn_events) if ts is not None]
    if not timestamps:
        raise RuntimeError("Turn has no valid timestamps")

    start_ns = None
    end_ns = None
    last_agent_message_ns = None

    for event in turn_events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if _safe_text(event.get("type"), 80) != "event_msg":
            continue
        ptype = _safe_text(payload.get("type"), 80)
        ts = _event_ns(event)
        if ts is None:
            continue
        if ptype == "user_message" and start_ns is None:
            start_ns = ts
        if ptype == "agent_message":
            last_agent_message_ns = ts
        if ptype == "task_complete":
            end_ns = ts

    if start_ns is None:
        start_ns = min(timestamps)
    if end_ns is None:
        end_ns = last_agent_message_ns or max(timestamps)
    if end_ns < start_ns:
        end_ns = start_ns
    return start_ns, end_ns


def _setup_mlflow() -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("Missing required env: MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_id = os.getenv(MLFLOW_EXPERIMENT_ID.name, "").strip()
    experiment_name = os.getenv(MLFLOW_EXPERIMENT_NAME.name, "").strip()
    if experiment_id:
        exp = mlflow.set_experiment(experiment_id=experiment_id)
        return exp.experiment_id
    if experiment_name:
        exp = mlflow.set_experiment(experiment_name=experiment_name)
        return exp.experiment_id
    return None


def _delete_existing_session_traces(*, session_id: str, experiment_id: str | None) -> int:
    if not experiment_id:
        return 0

    client = MlflowClient()
    filter_string = f"metadata.`{TraceMetadataKey.TRACE_SESSION}` = '{session_id}'"
    page_token = None
    deleted = 0
    while True:
        page = client.search_traces(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=200,
            include_spans=False,
            page_token=page_token,
        )
        trace_ids = [trace.info.trace_id for trace in page]
        if trace_ids:
            deleted += int(
                client.delete_traces(
                    experiment_id=experiment_id,
                    trace_ids=trace_ids,
                )
                or 0
            )
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return deleted


def _is_tool_like_event(event: dict[str, Any]) -> bool:
    et = _safe_text(event.get("type"), 80)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if et == "event_msg":
        ptype = _safe_text(payload.get("type"), 120).lower()
        return any(key in ptype for key in ("tool", "mcp", "command", "computer_call", "web_search"))
    if et == "response_item":
        ptype = _safe_text(payload.get("type"), 120).lower()
        return any(
            key in ptype
            for key in ("tool", "mcp", "function_call", "command_execution", "computer_call", "web_search")
        )
    return False


def _iter_traceable_events(turn_events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for event in turn_events:
        et = _safe_text(event.get("type"), 80)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if et == "event_msg":
            ptype = _safe_text(payload.get("type"), 120)
            if ptype == "agent_reasoning":
                items.append(("reasoning", event))
            elif ptype == "agent_message":
                items.append(("assistant_message", event))
            elif _is_tool_like_event(event):
                items.append(("tool", event))
        elif et == "response_item":
            # Avoid duplicate LLM spans for mirrored response_item message/reasoning events.
            if _is_tool_like_event(event):
                items.append(("tool", event))
    return items


def _extract_tool_name(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    for key in ("tool_name", "name", "function_name", "type"):
        val = _safe_text(payload.get(key), 120)
        if val:
            return val
    return "unknown"


def _json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return value


def _extract_tool_input_output(event: dict[str, Any]) -> tuple[Any, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    et = _safe_text(event.get("type"), 80)
    ptype = _safe_text(payload.get("type"), 120).lower()
    if et == "response_item":
        if ptype == "function_call":
            return _json_or_text(payload.get("arguments")), None
        if ptype in ("function_call_output", "tool_result", "tool_output"):
            return None, _json_or_text(payload.get("output") or payload.get("result"))
        if "tool" in ptype or "command" in ptype or "mcp" in ptype:
            tool_input = payload.get("input") or payload.get("arguments") or payload.get("params")
            tool_output = payload.get("output") or payload.get("result") or payload.get("content")
            return _json_or_text(tool_input), _json_or_text(tool_output)
    if et == "event_msg":
        tool_input = payload.get("input") or payload.get("arguments") or payload.get("request")
        tool_output = payload.get("output") or payload.get("result") or payload.get("response")
        return _json_or_text(tool_input), _json_or_text(tool_output)
    return None, None


def _build_tool_calls(turn_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    for event in turn_events:
        et = _safe_text(event.get("type"), 80)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        ptype = _safe_text(payload.get("type"), 120).lower()
        if et == "response_item" and ptype == "function_call":
            call_id = _safe_text(payload.get("call_id"), 160) or f"call-{len(pending)}"
            tool_name = _safe_text(payload.get("name"), 160) or "function_call"
            tool_input, _ = _extract_tool_input_output(event)
            pending[call_id] = {
                "call_id": call_id,
                "tool_name": tool_name,
                "input": tool_input,
                "output": None,
                "start_ns": _event_ns(event),
                "end_ns": _event_ns(event),
                "status": "ok",
                "error": "",
            }
            continue

        if et == "response_item" and ptype in ("function_call_output", "tool_result", "tool_output"):
            call_id = _safe_text(payload.get("call_id"), 160)
            _, tool_output = _extract_tool_input_output(event)
            error_text = _safe_text(payload.get("error"), 2000)
            status = "error" if error_text else "ok"
            if call_id and call_id in pending:
                item = pending.pop(call_id)
                item["output"] = tool_output
                item["end_ns"] = _event_ns(event) or item["end_ns"]
                item["status"] = status
                item["error"] = error_text
                completed.append(item)
            else:
                completed.append(
                    {
                        "call_id": call_id or "",
                        "tool_name": "function_call_output",
                        "input": None,
                        "output": tool_output,
                        "start_ns": _event_ns(event),
                        "end_ns": _event_ns(event),
                        "status": status,
                        "error": error_text,
                    }
                )
            continue

        if _is_tool_like_event(event):
            tool_name = _extract_tool_name(event)
            tool_input, tool_output = _extract_tool_input_output(event)
            completed.append(
                {
                    "call_id": "",
                    "tool_name": tool_name,
                    "input": tool_input,
                    "output": tool_output,
                    "start_ns": _event_ns(event),
                    "end_ns": _event_ns(event),
                    "status": "ok",
                    "error": "",
                }
            )

    completed.extend(pending.values())
    return completed


def _trace_single_turn(
    *,
    turn: SessionTurn,
    session_id: str,
    user_id: str,
    session_path: Path,
    source: str,
    previous_turn_total: dict[str, int] | None,
) -> dict[str, Any]:
    turn_start_ns, turn_end_ns = _resolve_turn_time_window(turn.events)
    usage, current_turn_total = _resolve_turn_usage(turn.events, previous_turn_total)
    model_display = (
        f"{turn.model} ({turn.reasoning_effort or 'unknown'})"
        if turn.model
        else f"unknown ({turn.reasoning_effort or 'unknown'})"
    )
    traceable_events = _iter_traceable_events(turn.events)
    tool_calls = _build_tool_calls(turn.events)
    last_assistant_message = ""

    root_span = mlflow.start_span_no_context(
        name="codex_turn",
        span_type=SpanType.AGENT,
        start_time_ns=turn_start_ns,
        inputs=turn.user_text,
        attributes={
            "session_id": session_id,
            "turn_id": turn.turn_id,
            "turn.index": turn.index,
            "model": turn.model,
            "reasoning_effort": turn.reasoning_effort,
            "model_display": model_display,
            "cwd": turn.cwd,
            "approval_policy": turn.approval_policy,
            "sandbox_policy": turn.sandbox_policy,
            "skills.available": json.dumps(turn.skills, ensure_ascii=False),
        },
        metadata={
            TraceMetadataKey.TRACE_SESSION: session_id,
            TraceMetadataKey.TRACE_USER: user_id,
            "mlflow.codex.turn_id": turn.turn_id,
            "mlflow.codex.turn_index": str(turn.index),
            "mlflow.codex.session_source": source,
            "mlflow.codex.session_path": str(session_path),
        },
        tags={
            TraceMetadataKey.TRACE_SESSION: session_id,
            TraceMetadataKey.TRACE_USER: user_id,
        },
    )

    llm_spans = []
    tool_count = 0
    reasoning_count = 0
    response_count = 0

    for tool in tool_calls:
        start_ns = tool.get("start_ns") or turn_start_ns
        end_ns = tool.get("end_ns") or start_ns
        tool_name = _safe_text(tool.get("tool_name"), 160) or "unknown_tool"
        span = mlflow.start_span_no_context(
            name=f"tool.{tool_name}",
            parent_span=root_span,
            span_type=SpanType.TOOL,
            start_time_ns=start_ns,
            inputs={
                "tool_name": tool_name,
                "call_id": _safe_text(tool.get("call_id"), 160),
                "input": tool.get("input"),
            },
            attributes={
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "tool_status": _safe_text(tool.get("status"), 80),
            },
        )
        span.set_outputs(
            {
                "output": tool.get("output"),
                "error": _safe_text(tool.get("error"), 2000),
            }
        )
        span.end(end_time_ns=end_ns + 1)
        tool_count += 1

    for kind, event in traceable_events:
        ev_ns = _event_ns(event) or turn_start_ns
        if kind == "tool":
            continue

        span_name = "reasoning" if kind == "reasoning" else "llm"
        span = mlflow.start_span_no_context(
            name=span_name,
            parent_span=root_span,
            span_type=SpanType.LLM,
            start_time_ns=ev_ns,
            inputs={
                "turn_id": turn.turn_id,
                "model": model_display,
            },
            attributes={
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "model": model_display,
                "model_display": model_display,
                "usage.input_tokens": usage["input_tokens"],
                "usage.cached_input_tokens": usage["cached_input_tokens"],
                "usage.output_tokens": usage["output_tokens"],
                "usage.total_tokens": usage["total_tokens"],
                "usage.token_count": usage["token_count"],
            },
        )
        preview = _extract_text_preview(event)
        if span_name == "llm" and preview.strip():
            last_assistant_message = preview
        span.set_outputs({"text": preview, "event": event})
        span.end(end_time_ns=ev_ns + 1)
        llm_spans.append(span)
        if span_name == "reasoning":
            reasoning_count += 1
        else:
            response_count += 1

    root_span.set_outputs(
        {
            "response": last_assistant_message,
            "usage": usage,
            "event_count": len(turn.events),
            "llm_span_count": len(llm_spans),
            "tool_span_count": tool_count,
        }
    )
    root_span.set_attribute("usage.input_tokens", usage["input_tokens"])
    root_span.set_attribute("usage.cached_input_tokens", usage["cached_input_tokens"])
    root_span.set_attribute("usage.output_tokens", usage["output_tokens"])
    root_span.set_attribute("usage.total_tokens", usage["total_tokens"])
    root_span.set_attribute("usage.token_count", usage["token_count"])
    root_span.set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: usage["input_tokens"],
            TokenUsageKey.OUTPUT_TOKENS: usage["output_tokens"],
            TokenUsageKey.TOTAL_TOKENS: usage["token_count"],
        },
    )
    root_span.end(end_time_ns=turn_end_ns)

    return {
        "trace_id": root_span.trace_id,
        "session_id": session_id,
        "turn_id": turn.turn_id,
        "turn_index": turn.index,
        "request": turn.user_text,
        "response_preview": last_assistant_message[:200],
        "event_count": len(turn.events),
        "usage": usage,
        "tool_span_count": tool_count,
        "reasoning_span_count": reasoning_count,
        "llm_span_count": response_count,
        "duration_ms": int((turn_end_ns - turn_start_ns) / 1_000_000),
        "_turn_total_usage": current_turn_total,
    }


def ingest_session_file(
    session_path: str,
    *,
    case_id_prefix: str = "codex-session",
    delete_existing_session: bool = True,
) -> dict[str, Any]:
    _ = case_id_prefix
    experiment_id = _setup_mlflow()
    path = Path(session_path).expanduser().resolve()
    events = _load_session_events(path)
    if not events:
        raise RuntimeError(f"Session file has no valid events: {path}")

    turns, session_meta = _split_turns(events)
    session_id = _safe_text(session_meta.get("id"), 120) or path.stem
    deleted_trace_count = 0
    if delete_existing_session:
        deleted_trace_count = _delete_existing_session_traces(
            session_id=session_id, experiment_id=experiment_id
        )
    source = _safe_text(session_meta.get("source"), 80)
    user_id = os.getenv("USER", "")
    trace_results: list[dict[str, Any]] = []
    source = _safe_text(session_meta.get("source"), 80)
    previous_turn_total: dict[str, int] | None = None
    for turn in turns:
        turn_result = _trace_single_turn(
            turn=turn,
            session_id=session_id,
            user_id=user_id,
            session_path=path,
            source=source,
            previous_turn_total=previous_turn_total,
        )
        previous_turn_total = turn_result.get("_turn_total_usage") or previous_turn_total
        trace_results.append(turn_result)

    total_usage = {
        "input_tokens": sum(r["usage"]["input_tokens"] for r in trace_results),
        "cached_input_tokens": sum(r["usage"]["cached_input_tokens"] for r in trace_results),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in trace_results),
        "total_tokens": sum(r["usage"]["total_tokens"] for r in trace_results),
        "token_count": sum(r["usage"]["token_count"] for r in trace_results),
    }
    total_event_count = sum(r["event_count"] for r in trace_results)

    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "event_count": total_event_count,
        "usage": total_usage,
        "deleted_trace_count": deleted_trace_count,
        "session_path": str(path),
        "traces": [
            {k: v for k, v in item.items() if k != "_turn_total_usage"} for item in trace_results
        ],
    }


def ingest_session_by_id(
    session_id: str,
    *,
    sessions_root: str | None = None,
    case_id_prefix: str = "codex-session",
    delete_existing_session: bool = True,
) -> dict[str, Any]:
    root = Path(sessions_root).expanduser() if sessions_root else Path.home() / ".codex" / "sessions"
    pattern = f"*{session_id}.jsonl"
    candidates = sorted(root.rglob(pattern))
    if not candidates:
        raise RuntimeError(f"Cannot find session_id={session_id} under {root}")
    return ingest_session_file(
        str(candidates[-1]),
        case_id_prefix=case_id_prefix,
        delete_existing_session=delete_existing_session,
    )


def ingest_session_files(
    session_paths: list[str],
    *,
    case_id_prefix: str = "codex-session",
    delete_existing_session: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in session_paths:
        results.append(
            ingest_session_file(
                path,
                case_id_prefix=case_id_prefix,
                delete_existing_session=delete_existing_session,
            )
        )
    return results
