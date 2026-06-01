"""Lenient JSON extraction from LLM responses.

Researcher/Engineer/Judge agents all receive Claude responses that may
embed JSON inside markdown fences, prose, or trailing commentary. This
helper centralizes the extraction logic that was previously duplicated
across three agent modules.

Strategy (in order):
    1. Strip whitespace.
    2. If wrapped in a ```json ... ``` (or plain ```) fence, take the
       fenced content.
    3. Find the first balanced top-level object/array using brace depth
       tracking that respects string literals and escapes.
    4. ``json.loads`` the candidate.

Returns ``None`` on any failure rather than raising — callers treat
agent output as untrusted and fall back to retry/skip logic.

stdlib-only. No anna_v2 imports.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```",
    re.DOTALL,
)


def _strip_fence(text: str) -> str:
    """Return the inside of the first ```json fence, or the original text."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group("body").strip()
    return text.strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Extract the first balanced ``open_ch``…``close_ch`` substring.

    Walks the string tracking nesting depth, but skips over characters
    inside double-quoted strings (with escape handling) so braces in
    string literals don't throw off the count.
    """
    start = text.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_json_response(
    text: str,
    *,
    expected_keys: Optional[Iterable[str]] = None,
    expect_array: bool = False,
) -> Optional[Any]:
    """Extract and parse a JSON object/array from a free-form LLM response.

    Args:
        text: Raw response from the model.
        expected_keys: When parsing an object, require these keys to be
            present. Returns None if any are missing. Ignored when
            ``expect_array`` is True.
        expect_array: If True, extract a balanced ``[ ... ]`` and require
            the result to be a list. Otherwise extract a ``{ ... }``
            and require a dict.

    Returns:
        The parsed Python value (dict or list) on success, ``None`` on
        any extraction or validation failure.
    """
    if not text or not isinstance(text, str):
        return None

    body = _strip_fence(text)

    if expect_array:
        candidate = _find_balanced(body, "[", "]")
    else:
        candidate = _find_balanced(body, "{", "}")

    if candidate is None:
        # Fall back to parsing the whole body in case it's already pure JSON
        # without surrounding prose.
        candidate = body

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.debug("parse_json_response: json.loads failed (%s)", exc)
        return None

    if expect_array:
        if not isinstance(parsed, list):
            logger.debug(
                "parse_json_response: expected array, got %s", type(parsed).__name__
            )
            return None
        return parsed

    if not isinstance(parsed, dict):
        logger.debug(
            "parse_json_response: expected object, got %s", type(parsed).__name__
        )
        return None

    if expected_keys:
        missing = [k for k in expected_keys if k not in parsed]
        if missing:
            logger.debug(
                "parse_json_response: missing required keys %s", missing
            )
            return None

    return parsed
