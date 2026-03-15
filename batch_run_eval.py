import argparse
import json

from batch_run_eval_core import (
    BatchEvalConfig,
    _build_distinct_run_name,
    _download_dataset_artifact,
    _extract_session_id_from_codex_stdout,
    _load_cases_from_file,
    _normalize_cases,
    _run_one_case,
    _setup_experiment as _setup_experiment_cfg,
    run_batch_eval,
    run_batch_eval_from_yaml,
)


def _setup_experiment(args) -> str:
    # Backward-compatible helper for existing notebook calls.
    cfg = BatchEvalConfig(
        experiment_id=getattr(args, "experiment_id", None),
        experiment_name=getattr(args, "experiment_name", None),
        input_file="__dummy__.jsonl",
    )
    return _setup_experiment_cfg(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch evaluate with codex exec and session ingest")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--input-file", help="Input file path (csv/json/jsonl) with inputs/expectations")
    parser.add_argument("--dataset-run-id", help="Existing dataset run id to load")
    parser.add_argument("--dataset-artifact-path", help="Artifact path in dataset run (e.g. datasets/cases.jsonl)")
    parser.add_argument("--experiment-id", help="Target MLflow experiment id")
    parser.add_argument("--experiment-name", help="Target MLflow experiment name")
    parser.add_argument("--run-name", default="codex-batch-eval", help="Base run name")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel worker count")
    parser.add_argument(
        "--keep-existing-session-traces",
        action="store_true",
        help="Do not delete existing traces for same session_id before ingest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config:
        if any([args.input_file, args.dataset_run_id, args.dataset_artifact_path]):
            raise SystemExit("When --config is set, do not pass input-file or dataset-run arguments.")
        out = run_batch_eval_from_yaml(args.config)
    else:
        cfg = BatchEvalConfig(
            input_file=args.input_file,
            dataset_run_id=args.dataset_run_id,
            dataset_artifact_path=args.dataset_artifact_path,
            experiment_id=args.experiment_id,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            max_workers=args.max_workers,
            keep_existing_session_traces=args.keep_existing_session_traces,
        )
        out = run_batch_eval(cfg)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
