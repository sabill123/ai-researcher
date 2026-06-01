"""Centralized logging + observability primitives.

Replaces the ad-hoc ``print()`` calls scattered through anna_v2 with a
single configurable logger tree, plus two side-channel observability
files used by the watchdog and post-hoc cost audits:

* ``data/heartbeats/<name>.touch`` — orchestrator main loop touches this
  every iteration; watchdog inspects mtime to detect hangs that pgrep
  alone cannot catch (zombie/wedged processes still register a PID).
* ``data/claude_ledger.jsonl`` — append-only record of every Claude CLI
  call: prompt hash, model, declared budget, observed cost, duration,
  success/fail. Lets us audit spend without rerunning agents.

stdlib-only.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time
from typing import Any, Dict, Optional

from .atomic_write import append_jsonl

_ROOT_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB per rotated log file
DEFAULT_BACKUP_COUNT = 10  # keep ~100 MiB of history


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Thin wrapper around ``logging.getLogger`` so callers don't have to
    remember to use ``__name__``. Configuration is handled centrally by
    ``setup_root_logger`` — calling ``get_logger`` before setup just
    yields a logger that inherits whatever the root currently has
    (usually WARNING + stderr), which is the stdlib default.
    """
    return logging.getLogger(name)


def setup_root_logger(
    log_dir: str,
    *,
    level: str = "INFO",
    filename: str = "anna.log",
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console: bool = True,
) -> logging.Logger:
    """Idempotently configure the root logger.

    Adds a console StreamHandler (stdout) and a RotatingFileHandler at
    ``<log_dir>/<filename>``. Safe to call multiple times — the second
    call is a no-op. Returns the root logger for convenience.

    The console handler is kept so existing watchdog grep patterns
    (``*** NEW BEST!``, ``*** HUMAN REVIEW CHECKPOINT ***``) keep
    working as long as callers preserve the message text.
    """
    global _ROOT_CONFIGURED
    with _CONFIG_LOCK:
        if _ROOT_CONFIGURED:
            return logging.getLogger()

        os.makedirs(log_dir, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(getattr(logging, level.upper(), logging.INFO))

        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

        _ROOT_CONFIGURED = True
        return root


def touch_heartbeat(name: str, heartbeat_dir: str) -> None:
    """Update the mtime on ``<heartbeat_dir>/<name>.touch``.

    Called by the orchestrator main loop each iteration. Watchdog
    compares ``time.time() - os.path.getmtime(path)`` against a stale
    threshold to detect wedged processes.

    Silently swallows OS errors — failing to update a heartbeat must
    never crash the producer; the watchdog will simply observe stale
    mtime and act accordingly.
    """
    try:
        os.makedirs(heartbeat_dir, exist_ok=True)
        path = os.path.join(heartbeat_dir, f"{name}.touch")
        # Touch without rewriting body — open(..., 'a') + close updates mtime.
        with open(path, "a", encoding="utf-8"):
            pass
        now = time.time()
        os.utime(path, (now, now))
    except OSError:
        logging.getLogger(__name__).debug(
            "touch_heartbeat(%s) failed", name, exc_info=True
        )


def append_claude_ledger(ledger_path: str, entry: Dict[str, Any]) -> None:
    """Append one record to the Claude CLI cost/outcome ledger.

    Expected fields (caller's responsibility): ``ts``, ``model``,
    ``prompt_hash``, ``duration_sec``, ``success`` (bool), ``cost_usd``,
    ``agent`` (researcher/engineer/judge), plus any free-form context.

    Append-only JSONL for cheap tailing and post-hoc analysis.
    """
    enriched = dict(entry)
    enriched.setdefault("ts", time.time())
    try:
        append_jsonl(ledger_path, enriched)
    except OSError:
        logging.getLogger(__name__).warning(
            "append_claude_ledger failed for %s", ledger_path, exc_info=True
        )
