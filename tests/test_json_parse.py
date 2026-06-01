"""Tests for ``anna_v2.utils.json_parse.parse_json_response``.

Covers fence stripping, balanced-brace extraction (including braces inside
strings), expected_keys gating, and graceful None return on failure.
"""
from __future__ import annotations

from anna_v2.utils.json_parse import parse_json_response


def test_parse_plain_object():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_fence():
    text = """```json
{"a": 1, "b": "two"}
```"""
    assert parse_json_response(text) == {"a": 1, "b": "two"}


def test_parse_unlabelled_fence():
    text = "```\n{\"k\": 42}\n```"
    assert parse_json_response(text) == {"k": 42}


def test_parse_object_in_prose():
    text = (
        "Here is my analysis. The result is:\n"
        '{"verdict": "improved", "delta": -0.02}\n'
        "End of report."
    )
    assert parse_json_response(text) == {"verdict": "improved", "delta": -0.02}


def test_balanced_braces_inside_string_literals():
    """Braces inside JSON string values must not confuse the balance scan."""
    text = '{"why": "training loop {} broke", "ok": true}'
    parsed = parse_json_response(text)
    assert parsed == {"why": "training loop {} broke", "ok": True}


def test_expected_keys_present():
    text = '{"a": 1, "b": 2, "c": 3}'
    assert parse_json_response(text, expected_keys=["a", "b"]) == {
        "a": 1, "b": 2, "c": 3,
    }


def test_expected_keys_missing_returns_none():
    text = '{"a": 1}'
    assert parse_json_response(text, expected_keys=["a", "missing"]) is None


def test_invalid_json_returns_none():
    assert parse_json_response("this is not json at all") is None


def test_empty_and_non_string_inputs():
    assert parse_json_response("") is None
    assert parse_json_response(None) is None  # type: ignore[arg-type]
    assert parse_json_response(123) is None  # type: ignore[arg-type]


def test_expect_array():
    text = "Here is a list:\n```json\n[1, 2, 3]\n```"
    assert parse_json_response(text, expect_array=True) == [1, 2, 3]


def test_expect_array_but_got_object_returns_none():
    assert parse_json_response('{"a": 1}', expect_array=True) is None


def test_expect_object_but_got_array_returns_none():
    assert parse_json_response("[1, 2, 3]") is None
