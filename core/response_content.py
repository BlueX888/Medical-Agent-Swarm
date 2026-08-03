"""Utilities for keeping structured agent metadata out of user-visible answers."""
from __future__ import annotations

import json
import re


_TRAILING_JSON_BLOCK = re.compile(
    r"\n*```json\s*(\{[\s\S]*?\})\s*```\s*$",
    re.IGNORECASE,
)
_STRUCTURED_METADATA_KEYS = {
    "suggestions",
    "disclaimer",
    "risk_level",
    "key_findings",
}


def strip_trailing_structured_metadata(answer: str) -> str:
    """Remove a trailing fenced JSON block used only for response metadata."""
    match = _TRAILING_JSON_BLOCK.search(answer)
    if not match:
        return answer
    try:
        payload = json.loads(match.group(1))
    except (json.JSONDecodeError, TypeError):
        return answer
    if not isinstance(payload, dict) or not payload:
        return answer
    keys = set(payload)
    if not keys.issubset(_STRUCTURED_METADATA_KEYS):
        return answer
    return answer[: match.start()].rstrip()
