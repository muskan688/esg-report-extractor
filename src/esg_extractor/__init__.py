"""ESG / Sustainability Report Extraction Assistant."""

from pathlib import Path

from dotenv import load_dotenv

# Loaded here (the top-level package) rather than in config.py so it fires
# for every entry point - CLI, Streamlit app, tests, or a bare `import
# esg_extractor.whatever` - regardless of which submodule happens to be
# imported first. Points at the project's own .env explicitly so it works
# no matter the caller's cwd; real environment variables still take
# precedence (load_dotenv defaults to override=False).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

__version__ = "0.1.0"
