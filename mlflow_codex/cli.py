import argparse
import json
import time
from uuid import uuid4

from dotenv import load_dotenv
from opentelemetry import trace

from .config import CodexTraceConfig
from .tracing import setup_tracer, trace_codex_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MLflow tracing runner for Codex exec.")
    parser.add_argument("--prompt", required=True, help="Prompt sent to `codex exec --json`.")
    parser.add_argument(
        "--case-id", default=str(uuid4()), help="Optional case id for trace attributes."
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    case_id = args.case_id or f"codex-cli-{int(time.time())}"
    config = CodexTraceConfig.from_env()
    tracer = setup_tracer(config)
    result = trace_codex_prompt(tracer, args.prompt, case_id, model=None)
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
