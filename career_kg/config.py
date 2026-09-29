import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

ONET_VERSION = "30_0"
ONET_URL = f"https://www.onetcenter.org/dl_files/database/db_{ONET_VERSION}_text.zip"
ONET_DIR = RAW_DIR / f"db_{ONET_VERSION}_text"
MODEL_DIR = ROOT / "data" / "models"      # git-ignored (data/)


def _load_env(path=ROOT / ".env"):
    """Minimal KEY=VALUE loader so credentials live in an untracked .env, never in code."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def _load_streamlit_secrets():
    """On Streamlit Cloud credentials come from st.secrets (no .env file in the repo)."""
    try:
        import streamlit as st
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:  # streamlit missing or no secrets file: fall back to .env / environment
        pass


_load_env()
_load_streamlit_secrets()

# Hosted Neo4j (e.g. Aura): NEO4J_URI looks like neo4j+s://<id>.databases.neo4j.io
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# Aura instances use the instance id as database name, not "neo4j"
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None
