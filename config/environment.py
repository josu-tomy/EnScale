"""Load project environment variables without overriding the caller's shell."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    """Load the repository's ignored `.env` file, preserving existing env vars."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
