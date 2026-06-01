"""Atomic file I/O helpers.

Every persistent state file in anna_v2 (loop_state, reflective_memory,
retrospectives, gpu_state, heartbeat) writes via these helpers. The
contract is:

* Write to ``<path>.tmp`` first, ``fsync`` the file body, then ``os.replace``
  onto the final path. ``os.replace`` is atomic within the same filesystem
  on POSIX, so a crash mid-write never leaves a partially-truncated target.
* Parent directory is created on demand. If creation fails (read-only FS),
  the exception propagates — callers are expected to surface it.
* On JSON decode failure, ``safe_load_json`` quarantines the corrupt file
  as ``<path>.corrupted.<unix_ts>`` and returns the supplied default. This
  prevents one bad write from poisoning every subsequent restart.

stdlib-only. No anna_v2 imports.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory if missing. Idempotent."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically via tmp+fsync+rename.

    On the same filesystem this is atomic: readers observe either the old
    contents or the new contents, never a partial write.

    Raises:
        OSError: on permission/IO failure (e.g. disk full). The tmp file
            is cleaned up before re-raising.
    """
    _ensure_parent_dir(path)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    # NamedTemporaryFile lives in the same dir so os.replace stays atomic
    # (cross-FS rename would silently fall back to copy on some kernels).
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; don't mask the original error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """Serialize ``data`` to JSON and write atomically.

    Convenience wrapper over ``atomic_write_text``. JSON is serialized
    before any filesystem op so a TypeError surfaces without touching disk.
    """
    payload = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
    )
    # Trailing newline keeps diffs/grep-friendliness for human inspection.
    if not payload.endswith("\n"):
        payload += "\n"
    atomic_write_text(path, payload)


def safe_load_json(
    path: str,
    default: Any,
    *,
    backup_corrupt: bool = True,
) -> Any:
    """Load JSON from ``path`` with crash-safe fallback.

    Returns ``default`` if the file is missing or corrupt. When
    ``backup_corrupt`` is True (default), a corrupt file is renamed to
    ``<path>.corrupted.<unix_ts>`` before returning the default so the
    next write doesn't clobber forensic evidence.
    """
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error(
            "Corrupt JSON at %s (%s); returning default", path, exc
        )
        if backup_corrupt:
            quarantine = f"{path}.corrupted.{int(time.time())}"
            try:
                os.replace(path, quarantine)
                logger.error("Quarantined corrupt JSON to %s", quarantine)
            except OSError as rename_err:
                logger.error("Failed to quarantine %s: %s", path, rename_err)
        return default
    except OSError as exc:
        logger.error("OS error reading %s: %s; returning default", path, exc)
        return default


def append_jsonl(path: str, entry: Any) -> None:
    """Append a single JSON record as one line to a JSONL file.

    Not atomic across multiple processes — relies on POSIX append-write
    being atomic for writes <= PIPE_BUF (4 KiB on Linux). For ledger
    entries this is sufficient; if you need stronger guarantees use a
    lock externally.
    """
    _ensure_parent_dir(path)
    line = json.dumps(entry, ensure_ascii=False)
    if not line.endswith("\n"):
        line += "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
