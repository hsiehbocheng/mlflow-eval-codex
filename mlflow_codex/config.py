import os
from dataclasses import dataclass


def _parse_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for kv in (raw or "").split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


@dataclass
class CodexTraceConfig:
    endpoint: str
    headers: dict[str, str]
    service_name: str = "codex-skill-batch-eval"
    environment: str = "dev"

    @classmethod
    def from_env(cls) -> "CodexTraceConfig":
        """
        Env-only configuration:
        - If MLFLOW_TRACKING_URI=databricks, require DATABRICKS_HOST + DATABRICKS_TOKEN.
        - Otherwise require MLFLOW_TRACKING_URI and use it directly.
        """
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))

        if not tracking_uri:
            raise RuntimeError(
                "Missing required env: MLFLOW_TRACKING_URI. "
                "Example: databricks or http://127.0.0.1:5001"
            )

        if tracking_uri == "databricks":
            databricks_host = os.getenv("DATABRICKS_HOST", "").strip()
            databricks_token = os.getenv("DATABRICKS_TOKEN", "").strip()
            if not databricks_host or not databricks_token:
                raise RuntimeError(
                    "MLFLOW_TRACKING_URI=databricks requires DATABRICKS_HOST and DATABRICKS_TOKEN."
                )
            endpoint = f"{databricks_host.rstrip('/')}/api/2.0/otel/v1/traces"
            headers.setdefault("Authorization", f"Bearer {databricks_token}")
            headers.setdefault("content-type", "application/x-protobuf")
            uc_table = os.getenv("DATABRICKS_UC_OTEL_TABLE", "")
            if uc_table:
                headers.setdefault("X-Databricks-UC-Table-Name", uc_table)
        else:
            endpoint = tracking_uri

        return cls(
            endpoint=endpoint,
            headers=headers,
            service_name=os.getenv("OTEL_SERVICE_NAME", "codex-skill-batch-eval"),
            environment=os.getenv("ENV", "dev"),
        )
