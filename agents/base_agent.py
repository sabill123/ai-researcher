"""
Base agent: Claude Code CLI calling with configurable tools.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from ..utils.json_parse import parse_json_response  # re-export
from ..utils.logging_setup import append_claude_ledger, get_logger

logger = get_logger(__name__)

# Re-export so callers can `from anna_v2.agents.base_agent import parse_json_response`.
__all__ = [
    "SAFETY_SYSTEM_PROMPT",
    "find_claude_binary",
    "call_claude_cli",
    "ClaudeCallResult",
    "parse_json_response",
]


SAFETY_SYSTEM_PROMPT = """## CRITICAL SAFETY CONSTRAINTS — ABSOLUTE RULES

You are a research agent operating on a SHARED GPU SERVER.
You MUST follow these rules with ZERO exceptions.

### FILESYSTEM RESTRICTIONS
- ALLOWED PATH: /home/jaeseokhan/2025-02/khu/** ONLY
- FORBIDDEN: Any path outside the allowed path.
- NEVER suggest deleting, modifying, or overwriting files outside experiment directories.

### PROCESS RESTRICTIONS
- This is a SHARED SERVER with multiple users.
- NEVER suggest killing, stopping, or interfering with ANY running processes.
- NEVER suggest: kill, pkill, killall, nvidia-smi --gpu-reset, shutdown, reboot, rm -rf

### SCOPE RESTRICTIONS
- You are ONLY working on KHU facial palsy severity prediction experiments.
- All code modifications must preserve the training script interface:
  Required args: --data_dir, --output_dir, --exp, --epochs, --no_wandb
  Required output: test_metrics.json with *_score_mae keys
- NEVER modify utils.py (metric computation logic)
"""


# Default ledger path. May be overridden per-call. Resolved relative to the
# anna_v2 package directory so it lives alongside the other observability
# files in <anna_v2>/data/ regardless of cwd at call time.
_DEFAULT_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "claude_ledger.jsonl",
)


@dataclass
class ClaudeCallResult:
    """Structured result of a Claude CLI invocation.

    The legacy API returned just the response text (``str | None``). Many
    call sites still do ``result.text`` or treat ``result`` as a string —
    so this dataclass:
      * keeps ``text`` as the primary payload field,
      * implements ``__str__``/``__bool__`` so it transparently behaves
        like the text in string contexts, and
      * exposes structured observability (cost, duration, success) for
        ledger entries and post-hoc analysis.
    """

    text: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    success: bool = False
    error: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)

    def __str__(self) -> str:  # backward compat: str(result) == result.text
        return self.text or ""

    def __bool__(self) -> bool:  # ``if result:`` should mean "got text"
        return bool(self.text)

    def __len__(self) -> int:
        return len(self.text or "")

    def strip(self) -> str:
        """Backward compat for callers that did ``response.strip()`` on the str."""
        return (self.text or "").strip()


def find_claude_binary() -> Optional[str]:
    """Find the claude CLI binary."""
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    candidates = [
        os.path.expanduser("~/.npm-global/bin/claude"),
        "/usr/local/bin/claude",
        os.path.expanduser("~/.local/bin/claude"),
    ]

    vscode_patterns = [
        os.path.expanduser("~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"),
        os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"),
    ]
    for pattern in vscode_patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        candidates.extend(matches)

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _prompt_hash(prompt: str) -> str:
    """Short prompt fingerprint for the ledger (avoid logging the full text)."""
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:16]


def call_claude_cli(prompt: str,
                    model: str = "sonnet",
                    max_budget_usd: float = 0.50,
                    timeout_seconds: int = 180,
                    tools: Optional[str] = "",
                    cwd: Optional[str] = None,
                    system_prompt: Optional[str] = None,
                    allowed_tools: Optional[str] = None,
                    agent: Optional[str] = None,
                    ledger_path: Optional[str] = None) -> ClaudeCallResult:
    """
    Call Claude Code CLI in non-interactive mode.

    Returns a :class:`ClaudeCallResult`. The dataclass behaves like a string
    in legacy contexts (``str(result)``, ``if result:``, ``result.strip()``)
    so existing callers that used to receive ``Optional[str]`` keep working.
    On failure, ``result.success`` is False and ``result.text`` is empty —
    ``if not result:`` still detects the failure exactly as before.

    Args:
        prompt: The prompt to send
        model: Model to use (sonnet, opus, haiku)
        max_budget_usd: Max spend per invocation
        timeout_seconds: Timeout for the CLI call
        tools: Tools configuration ("" for no tools, None for default tools)
        cwd: Working directory for the Claude process
        system_prompt: Override system prompt (defaults to SAFETY_SYSTEM_PROMPT)
        allowed_tools: Tools to allow without permission (e.g. "Edit,Read,Write")
        agent: Optional agent name for ledger attribution ("researcher" / "engineer" / "judge").
        ledger_path: Override ledger path (defaults to ``<anna_v2>/data/claude_ledger.jsonl``).
    """
    ledger_target = ledger_path or _DEFAULT_LEDGER_PATH
    start = time.monotonic()

    def _record(result: ClaudeCallResult) -> ClaudeCallResult:
        """Append one ledger entry. Never raises into the caller."""
        try:
            append_claude_ledger(
                ledger_target,
                {
                    "agent": agent,
                    "model": model,
                    "prompt_hash": _prompt_hash(prompt),
                    "duration_sec": result.duration_ms / 1000.0,
                    "duration_ms": result.duration_ms,
                    "success": result.success,
                    "cost_usd": result.cost_usd,
                    "max_budget_usd": max_budget_usd,
                    "timeout_seconds": timeout_seconds,
                    "error": result.error,
                    "text_len": len(result.text or ""),
                },
            )
        except Exception:  # ledger is best-effort
            logger.debug("ledger append failed", exc_info=True)
        return result

    claude_bin = find_claude_binary()
    if not claude_bin:
        logger.error("'claude' CLI not found.")
        return _record(ClaudeCallResult(
            duration_ms=int((time.monotonic() - start) * 1000),
            error="claude binary not found",
        ))

    cmd = [
        claude_bin,
        "-p", prompt,
        "--output-format", "json",
        "--no-session-persistence",
        "--model", model,
        "--max-budget-usd", str(max_budget_usd),
    ]

    if tools is not None:
        cmd.extend(["--tools", tools])

    if allowed_tools:
        cmd.extend(["--allowedTools", allowed_tools])

    sp = system_prompt or SAFETY_SYSTEM_PROMPT
    cmd.extend(["--system-prompt", sp])

    # Must unset CLAUDECODE env var to bypass nesting protection
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=cwd,
        )

        duration_ms = int((time.monotonic() - start) * 1000)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            logger.error("Claude CLI error (exit %s): %s", completed.returncode, stderr[:200])
            return _record(ClaudeCallResult(
                duration_ms=duration_ms,
                error=f"exit {completed.returncode}: {stderr[:200]}",
            ))

        stdout = (completed.stdout or "").strip()
        if not stdout:
            logger.warning("Claude CLI returned empty response")
            return _record(ClaudeCallResult(
                duration_ms=duration_ms,
                error="empty response",
            ))

        text = stdout
        cost = 0.0
        raw: Optional[dict] = None
        try:
            response_data = json.loads(stdout)
            if isinstance(response_data, dict):
                raw = response_data
                if response_data.get("type") == "result":
                    cost = float(response_data.get("total_cost_usd", 0) or 0)
                    text = response_data.get("result", "") or ""
                elif "result" in response_data:
                    text = response_data.get("result", "") or ""
                    cost = float(response_data.get("total_cost_usd", 0) or 0)
        except json.JSONDecodeError:
            # stdout wasn't JSON — treat the whole thing as plain text.
            text = stdout

        if cost:
            logger.info("Claude CLI cost: $%.4f", cost)

        return _record(ClaudeCallResult(
            text=text,
            cost_usd=cost,
            duration_ms=duration_ms,
            success=bool(text),
            raw=raw,
        ))

    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.error("Claude CLI timed out after %ss", timeout_seconds)
        return _record(ClaudeCallResult(
            duration_ms=duration_ms,
            error=f"timeout after {timeout_seconds}s",
        ))
    except FileNotFoundError:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.error("Claude CLI binary not found at: %s", claude_bin)
        return _record(ClaudeCallResult(
            duration_ms=duration_ms,
            error=f"binary not found: {claude_bin}",
        ))
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.exception("Claude CLI error")
        return _record(ClaudeCallResult(
            duration_ms=duration_ms,
            error=str(exc),
        ))
