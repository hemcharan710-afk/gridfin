import os
from pathlib import Path

import pytest

# Force deterministic offline behaviour for the whole suite.
os.environ["GRIDFIN_MOCK_LLM"] = "true"

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"

ACME = "Acme_Corp_10-K_2023-de6621c9"
BETA = "Beta_Inc_10-K_2023-4daacef1"
GAMMA = "Gamma_Ltd_Annual_2023-5edb2116"


@pytest.fixture(scope="session")
def demo_dir() -> Path:
    return DEMO


@pytest.fixture
def corpus(demo_dir, tmp_path, monkeypatch):
    # Isolate the cell cache per test so cache state never leaks between tests.
    monkeypatch.setenv("GRIDFIN_CACHE_DIR", str(tmp_path / "cache"))
    from gridfin.config import get_settings

    get_settings.cache_clear()
    from gridfin.pipeline import GridFin

    return GridFin.from_path(demo_dir, cache_namespace=f"cells_{tmp_path.name}")
