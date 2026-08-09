"""Deep-research skill entrypoint for Claude-compatible skill discovery."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.deep_research_skill import (  # noqa: E402,F401
    deep_research,
    deep_research_sync,
    format_research_report,
)

__all__ = ["deep_research", "deep_research_sync", "format_research_report"]
