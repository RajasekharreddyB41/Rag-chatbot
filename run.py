"""Entry point for the RAG Chatbot application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "app" / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
