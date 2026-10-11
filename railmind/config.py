"""Configuration from environment / local .env (never committed)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


class Config:
    PROVIDER = os.getenv("RAILMIND_PROVIDER", "none").strip().lower()
    PROVIDER_API_KEY = os.getenv("RAILMIND_PROVIDER_API_KEY", "")
    PROVIDER_HOST = os.getenv("RAILMIND_PROVIDER_HOST", "irctc1.p.rapidapi.com")
    PROVIDER_STATUS_PATH = os.getenv("RAILMIND_PROVIDER_STATUS_PATH", "/api/v1/liveTrainStatus")
    PROVIDER_TERMS_URL = os.getenv("RAILMIND_PROVIDER_TERMS_URL", "")
    MIN_SECONDS_BETWEEN_CALLS_PER_TRAIN = int(os.getenv("RAILMIND_MIN_REFRESH_SECONDS", "120"))
    MAX_CALLS_PER_MINUTE = int(os.getenv("RAILMIND_MAX_CALLS_PER_MINUTE", "5"))
    MAX_CALLS_PER_DAY = int(os.getenv("RAILMIND_MAX_CALLS_PER_DAY", "20"))
    STALE_AFTER_SECONDS = int(os.getenv("RAILMIND_STALE_AFTER_SECONDS", "900"))
    HTTP_TIMEOUT = float(os.getenv("RAILMIND_HTTP_TIMEOUT", "10"))
    LOG_LEVEL = os.getenv("RAILMIND_LOG_LEVEL", "INFO")
    HOST = os.getenv("RAILMIND_HOST", "127.0.0.1")
    PORT = int(os.getenv("RAILMIND_PORT", "5000"))
