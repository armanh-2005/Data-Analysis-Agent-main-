from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _env_int(key: str, default: int) -> int:
    v = _env_str(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = _env_str(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = _env_str(key)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # Core paths
    db_path: str
    artifacts_dir: str

    # Logging
    log_level: str
    log_json: bool

    # Importer / privacy
    respondent_id_salt: str

    # Sandbox / execution safety (used by executor later)
    sandbox_timeout_seconds: int
    max_code_iterations: int
    max_quality_iterations: int

    # LLM model names (placeholders; used by llm_factory later)
    router_model: str
    mapper_model: str
    planner_model: str
    stats_params_model: str
    code_writer_model: str
    code_reviewer_model: str
    quality_review_model: str
    report_writer_model: str

    @staticmethod
    def from_env() -> "Settings":
        # Read configuration from environment variables.
        # Keep defaults safe and local-friendly.
        db_path = _env_str("APP_DB_PATH", "data/app.db")
        artifacts_dir = _env_str("APP_ARTIFACTS_DIR", "artifacts")

        # Create directories if needed (do not create DB file here).
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        return Settings(
            db_path=db_path,
            artifacts_dir=artifacts_dir,

            log_level=_env_str("APP_LOG_LEVEL", "INFO") or "INFO",
            log_json=_env_bool("APP_LOG_JSON", True),

            respondent_id_salt=_env_str("APP_RESPONDENT_ID_SALT", "CHANGE_ME_SALT") or "CHANGE_ME_SALT",

            sandbox_timeout_seconds=_env_int("APP_SANDBOX_TIMEOUT_SECONDS", 20),
            max_code_iterations=_env_int("APP_MAX_CODE_ITERATIONS", 5),
            max_quality_iterations=_env_int("APP_MAX_QUALITY_ITERATIONS", 3),

            router_model=_env_str("APP_MODEL_ROUTER", "gpt-4.1-mini") or "gpt-4.1-mini",
            mapper_model=_env_str("APP_MODEL_MAPPER", "gpt-4.1-mini") or "gpt-4.1-mini",
            planner_model=_env_str("APP_MODEL_PLANNER", "gpt-4.1") or "gpt-4.1",
            stats_params_model=_env_str("APP_MODEL_STATS_PARAMS", "gpt-4.1") or "gpt-4.1",
            code_writer_model=_env_str("APP_MODEL_CODE_WRITER", "gpt-4.1") or "gpt-4.1",
            code_reviewer_model=_env_str("APP_MODEL_CODE_REVIEWER", "gpt-4.1") or "gpt-4.1",
            quality_review_model=_env_str("APP_MODEL_QUALITY_REVIEW", "gpt-4.1") or "gpt-4.1",
            report_writer_model=_env_str("APP_MODEL_REPORT_WRITER", "gpt-4.1") or "gpt-4.1",
        )
