"""Central configuration. Read once, injected everywhere.

Optional heavy dependencies (sentence-transformers, faiss, pdfplumber, ...) are
detected at runtime; the reference implementation degrades to lightweight pure
fallbacks when they are absent, so the bundled demo always runs.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRIDFIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = "anthropic"  # "anthropic" | "openai"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Anthropic model routing.
    model_small: str = "claude-haiku-4-5-20251001"
    model_large: str = "claude-opus-4-8"
    # OpenAI model routing (used when llm_provider == "openai").
    openai_model_small: str = "gpt-4o-mini"
    openai_model_large: str = "gpt-4o-mini"
    mock_llm: bool = False

    # --- Grid fan-out ------------------------------------------------------
    max_concurrency: int = 8
    cell_max_retries: int = 3

    # --- Retrieval ---------------------------------------------------------
    retrieval_k: int = 8
    rerank_top_n: int = 4
    rrf_k: int = 60

    # --- Paths -------------------------------------------------------------
    cache_dir: Path = PROJECT_ROOT / ".gridfin_cache"
    grid_db_path: Path = PROJECT_ROOT / "gridfin_grid.duckdb"
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def provider(self) -> str:
        """Normalised LLM provider name."""
        return (self.llm_provider or "anthropic").strip().lower()

    @property
    def use_live_llm(self) -> bool:
        """True only when we have a key for the active provider and have not been
        forced into mock mode."""
        import os

        if self.mock_llm:
            return False
        if self.provider == "openai":
            return bool(self.openai_api_key or os.environ.get("OPENAI_API_KEY", ""))
        # The Claude Code gateway also injects ANTHROPIC_BASE_URL; a key is still
        # required for the anthropic SDK to authenticate.
        return bool(self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""))

    # --- Optional dependency capability flags ------------------------------
    @property
    def has_sentence_transformers(self) -> bool:
        return _has("sentence_transformers")

    @property
    def has_faiss(self) -> bool:
        return _has("faiss")

    @property
    def has_cross_encoder(self) -> bool:
        return _has("sentence_transformers")

    @property
    def has_pdf(self) -> bool:
        return _has("pdfplumber") or _has("fitz")

    @property
    def has_docx(self) -> bool:
        return _has("docx")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
