"""Streamlit-Dashboard Starter für den Bewerbungsoptimizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard.streamlit_app import main

main()
