"""Tests for ``anna_v2.utils.atomic_write``.

Covers:
* round-trip JSON/text via ``atomic_write_json`` / ``safe_load_json``.
* corruption recovery: bad JSON is quarantined to ``<path>.corrupted.<ts>``
  and the supplied default is returned.
* concurrent writes from multiple threads never leave a torn file.
"""
from __future__ import annotations

import glob
import json
import os
import threading

import pytest

from anna_v2.utils.atomic_write import (
    atomic_write_json,
    atomic_write_text,
    safe_load_json,
    append_jsonl,
)


def test_atomic_write_json_round_trip(tmp_path):
    p = str(tmp_path / "data.json")
    payload = {"a": 1, "b": ["x", "y"], "c": {"k": True}}
    atomic_write_json(p, payload)
    assert os.path.exists(p)
    loaded = safe_load_json(p, default=None)
    assert loaded == payload


def test_atomic_write_text_creates_parent(tmp_path):
    p = str(tmp_path / "sub" / "deep" / "out.txt")
    atomic_write_text(p, "hello\n")
    with open(p) as fh:
        assert fh.read() == "hello\n"


def test_safe_load_json_missing_returns_default(tmp_path):
    p = str(tmp_path / "nonexistent.json")
    assert safe_load_json(p, default={"default": True}) == {"default": True}


def test_safe_load_json_corrupt_quarantines_and_returns_default(tmp_path):
    p = str(tmp_path / "bad.json")
    with open(p, "w") as fh:
        fh.write("{not valid json,,,")
    result = safe_load_json(p, default=[])
    assert result == []
    # Original is gone, a *.corrupted.<ts> sibling should exist.
    assert not os.path.exists(p)
    quarantined = glob.glob(str(tmp_path / "bad.json.corrupted.*"))
    assert len(quarantined) == 1, f"expected 1 quarantine file, got {quarantined}"


def test_safe_load_json_corrupt_without_backup(tmp_path):
    p = str(tmp_path / "bad.json")
    with open(p, "w") as fh:
        fh.write("nope")
    result = safe_load_json(p, default={"ok": 1}, backup_corrupt=False)
    assert result == {"ok": 1}
    # No backup_corrupt -> original stays where it was.
    assert os.path.exists(p)


def test_atomic_write_json_concurrent(tmp_path):
    """Many threads writing should always leave a parseable file."""
    p = str(tmp_path / "concurrent.json")

    def writer(value: int):
        atomic_write_json(p, {"value": value})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # File must be parseable JSON with a value in [0, 19].
    loaded = safe_load_json(p, default=None)
    assert isinstance(loaded, dict)
    assert "value" in loaded
    assert 0 <= loaded["value"] < 20


def test_append_jsonl_writes_one_record_per_line(tmp_path):
    p = str(tmp_path / "ledger.jsonl")
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    with open(p) as fh:
        lines = fh.read().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]


def test_atomic_write_json_invalid_type_does_not_touch_disk(tmp_path):
    """A TypeError during json.dumps must NOT create or replace the target."""
    p = str(tmp_path / "guarded.json")
    with pytest.raises(TypeError):
        atomic_write_json(p, {"bad": {1, 2, 3}})  # set is not JSON-serializable
    assert not os.path.exists(p)
