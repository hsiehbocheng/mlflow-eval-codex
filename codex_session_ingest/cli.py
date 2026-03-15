import argparse
import json

from dotenv import load_dotenv

from .ingest import ingest_session_by_id, ingest_session_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Codex session JSONL into MLflow traces")
    parser.add_argument("--session-path", help="Path to a Codex session JSONL file")
    parser.add_argument("--session-id", help="Codex session id (auto-search in ~/.codex/sessions)")
    parser.add_argument(
        "--sessions-root",
        help="Optional root for session-id lookup (default: ~/.codex/sessions)",
    )
    parser.add_argument(
        "--case-id-prefix",
        default="codex-session",
        help="Prefix used for synthetic case_id metadata",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing traces for the same session before ingest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.session_path) == bool(args.session_id):
        raise SystemExit("Please provide exactly one of --session-path or --session-id")
    load_dotenv()
    if args.session_path:
        result = ingest_session_file(
            session_path=args.session_path,
            case_id_prefix=args.case_id_prefix,
            delete_existing_session=not args.keep_existing,
        )
    else:
        result = ingest_session_by_id(
            session_id=args.session_id,
            sessions_root=args.sessions_root,
            case_id_prefix=args.case_id_prefix,
            delete_existing_session=not args.keep_existing,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
