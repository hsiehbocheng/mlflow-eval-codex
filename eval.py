import json
import subprocess

from dotenv import load_dotenv

import mlflow_codex as codex


def run_case(case_id: str, prompt: str) -> dict:
    # Keep your existing codex usage style in Python.
    cmd = ["codex", "exec", "--json", prompt]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "case_id": case_id,
        "returncode": proc.returncode,
        "stdout_lines": len((proc.stdout or "").splitlines()),
        "stderr_preview": (proc.stderr or "")[:200],
    }


def main() -> None:
    load_dotenv()

    # No model control in mlflow_codex layer.
    # Trace is automatically generated when subprocess executes `codex exec --json`.
    codex.autolog(patch_subprocess=True)

    cases = [
        ("eval-1", "幫我從 grafana mcp 確認 tools"),
        ("eval-2", "請列出目前可用的 mcp servers"),
    ]

    results = []
    for case_id, prompt in cases:
        results.append(run_case(case_id, prompt))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
