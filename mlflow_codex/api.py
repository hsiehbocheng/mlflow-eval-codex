import time
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from .config import CodexTraceConfig
from .tracing import setup_tracer, trace_codex_completed_process, trace_codex_prompt

_PATCH_LOCK = threading.Lock()
_PATCHED = False
_ORIGINAL_RUN = None
_STATE = threading.local()


def _is_guarded() -> bool:
    return bool(getattr(_STATE, "guarded", False))


def _set_guarded(value: bool) -> None:
    _STATE.guarded = value


def _normalize_command(command: Any) -> list[str]:
    if isinstance(command, (list, tuple)):
        return [str(x) for x in command]
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except Exception:
            return command.split()
    return []


def _is_codex_exec(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    first = tokens[0].split("/")[-1]
    return first == "codex" and "exec" in tokens[1:]


def _extract_model_from_tokens(tokens: list[str]) -> str | None:
    for i, tok in enumerate(tokens):
        if tok in ("-m", "--model") and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _extract_prompt_from_tokens(tokens: list[str]) -> str:
    # Best effort: for codex exec, prompt is usually the last positional token.
    for tok in reversed(tokens):
        if not tok.startswith("-"):
            if tok in ("exec", "resume", "review", "help"):
                continue
            return tok
    return ""


@dataclass
class CodexAutoLogger:
    case_id_prefix: str = "codex"

    def run(
        self,
        prompt: str,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        final_case_id = case_id or f"{self.case_id_prefix}-{int(time.time() * 1000)}"
        was_guarded = _is_guarded()
        _set_guarded(True)
        try:
            return trace_codex_prompt(None, prompt, final_case_id, model=None)
        finally:
            _set_guarded(was_guarded)

    def evaluate_batch(
        self,
        cases: list[str | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for idx, case in enumerate(cases, start=1):
            if isinstance(case, str):
                prompt = case
                case_id = f"{self.case_id_prefix}-batch-{idx}"
            else:
                prompt = str(case.get("prompt", ""))
                case_id = str(case.get("case_id") or f"{self.case_id_prefix}-batch-{idx}")
            results.append(self.run(prompt=prompt, case_id=case_id))
        return results


def _enable_subprocess_patch(case_id_prefix: str) -> None:
    global _PATCHED, _ORIGINAL_RUN
    with _PATCH_LOCK:
        if _PATCHED:
            return
        _ORIGINAL_RUN = subprocess.run

        def _wrapped_run(*args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            tokens = _normalize_command(command)
            should_trace = _is_codex_exec(tokens) and not _is_guarded()
            started_ns = time.time_ns()
            result = _ORIGINAL_RUN(*args, **kwargs)
            ended_ns = time.time_ns()

            if should_trace:
                try:
                    model = _extract_model_from_tokens(tokens)
                    prompt = _extract_prompt_from_tokens(tokens)
                    case_id = f"{case_id_prefix}-subprocess-{int(time.time() * 1000)}"
                    stdout = result.stdout if isinstance(result.stdout, str) else ""
                    stderr = result.stderr if isinstance(result.stderr, str) else ""
                    if "--json" in tokens and stdout:
                        was_guarded = _is_guarded()
                        _set_guarded(True)
                        try:
                            trace_codex_completed_process(
                                prompt=prompt,
                                case_id=case_id,
                                model=model,
                                returncode=int(result.returncode),
                                stdout=stdout,
                                stderr=stderr,
                                started_at_ns=started_ns,
                                ended_at_ns=ended_ns,
                            )
                        finally:
                            _set_guarded(was_guarded)
                except Exception:
                    # Never break user command execution due to tracing side effects.
                    pass

            return result

        subprocess.run = _wrapped_run
        _PATCHED = True


def autolog(
    *,
    case_id_prefix: str = "codex",
    load_env: bool = True,
    patch_subprocess: bool = False,
) -> CodexAutoLogger:
    if load_env:
        load_dotenv()
    config = CodexTraceConfig.from_env()
    setup_tracer(config)
    if patch_subprocess:
        _enable_subprocess_patch(case_id_prefix=case_id_prefix)
    return CodexAutoLogger(case_id_prefix=case_id_prefix)


def run(
    prompt: str,
    *,
    case_id: str | None = None,
    load_env: bool = True,
) -> dict[str, Any]:
    logger = autolog(load_env=load_env)
    return logger.run(prompt=prompt, case_id=case_id)


def evaluate_batch(
    cases: list[str | dict[str, Any]],
    *,
    case_id_prefix: str = "codex",
    load_env: bool = True,
) -> list[dict[str, Any]]:
    logger = autolog(case_id_prefix=case_id_prefix, load_env=load_env)
    return logger.evaluate_batch(cases=cases)
