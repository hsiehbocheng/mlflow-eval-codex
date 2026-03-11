import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

import mlflow_codex as codex

# Define manual review fields here.
# You can add/remove keys to match your evaluation workflow.
HUMAN_REVIEW_FIELDS = {
    "human_pass": None,
    "human_feedback": "",
}


def load_cases(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_human_pass(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "t", "1", "yes", "y", "pass"):
        return True
    if text in ("false", "f", "0", "no", "n", "fail"):
        return False
    return None


def main() -> None:
    load_dotenv()
    codex.autolog(patch_subprocess=True)

    cases_path = Path("batch_cases.jsonl")
    cases = load_cases(str(cases_path))
    df = pd.DataFrame(cases)
    dataset = mlflow.data.from_pandas(df, source=str(cases_path), name="codex_batch_cases")

    with mlflow.start_run(run_name="codex-batch-eval-demo"):
        mlflow.log_input(dataset, context="evaluation")
        mlflow.log_param("case_count", len(cases))

        auto_pass_count = 0
        final_pass_count = 0
        results: list[dict] = []

        for case in cases:
            case_id = str(case.get("case_id"))
            prompt = str(case.get("prompt", ""))
            expected = str(case.get("expected", "")).lower()

            proc = subprocess.run(
                ["codex", "exec", "--json", prompt],
                text=True,
                capture_output=True,
            )

            merged = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
            auto_pass = (expected in merged) if expected else (proc.returncode == 0)
            auto_pass_count += int(auto_pass)

            review_values = {k: case.get(k, default) for k, default in HUMAN_REVIEW_FIELDS.items()}
            human_pass = parse_human_pass(review_values.get("human_pass"))
            human_feedback = str(review_values.get("human_feedback", "") or "")
            final_pass = human_pass if human_pass is not None else auto_pass
            final_pass_count += int(final_pass)

            result_row = {
                "case_id": case_id,
                "returncode": proc.returncode,
                "expected": expected,
                "auto_pass": auto_pass,
                "human_pass": human_pass,
                "human_feedback": human_feedback,
                "final_pass": final_pass,
            }
            # Preserve any extra manual fields defined by HUMAN_REVIEW_FIELDS.
            for key in HUMAN_REVIEW_FIELDS:
                if key not in result_row:
                    result_row[key] = review_values.get(key)
            results.append(result_row)

        auto_pass_rate = auto_pass_count / len(cases) if cases else 0.0
        final_pass_rate = final_pass_count / len(cases) if cases else 0.0
        mlflow.log_metric("auto_pass_count", auto_pass_count)
        mlflow.log_metric("auto_pass_rate", auto_pass_rate)
        mlflow.log_metric("final_pass_count", final_pass_count)
        mlflow.log_metric("final_pass_rate", final_pass_rate)

        out_path = Path("batch_results.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_path))

        print(
            json.dumps(
                {
                    "auto_pass_count": auto_pass_count,
                    "auto_pass_rate": auto_pass_rate,
                    "final_pass_count": final_pass_count,
                    "final_pass_rate": final_pass_rate,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
