"""
Base agent: Claude Code CLI calling with configurable tools.
"""

import glob
import json
import os
import shutil
import subprocess
from typing import Optional


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


def call_claude_cli(prompt: str,
                    model: str = "sonnet",
                    max_budget_usd: float = 0.50,
                    timeout_seconds: int = 180,
                    tools: Optional[str] = "",
                    cwd: Optional[str] = None,
                    system_prompt: Optional[str] = None,
                    allowed_tools: Optional[str] = None) -> Optional[str]:
    """
    Call Claude Code CLI in non-interactive mode.

    Args:
        prompt: The prompt to send
        model: Model to use (sonnet, opus, haiku)
        max_budget_usd: Max spend per invocation
        timeout_seconds: Timeout for the CLI call
        tools: Tools configuration ("" for no tools, None for default tools)
        cwd: Working directory for the Claude process
        system_prompt: Override system prompt (defaults to SAFETY_SYSTEM_PROMPT)
        allowed_tools: Tools to allow without permission (e.g. "Edit,Read,Write")
    """
    claude_bin = find_claude_binary()
    if not claude_bin:
        print("ERROR: 'claude' CLI not found.")
        return None

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=cwd,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"  Claude CLI error (exit {result.returncode}): {stderr[:200]}")
            return None

        stdout = result.stdout.strip()
        if not stdout:
            print("  Claude CLI returned empty response")
            return None

        try:
            response_data = json.loads(stdout)
            if isinstance(response_data, dict):
                if response_data.get("type") == "result":
                    cost = response_data.get("total_cost_usd", 0)
                    print(f"  Claude CLI cost: ${cost:.4f}")
                    return response_data.get("result", "")
                if "result" in response_data:
                    return response_data["result"]
            return stdout
        except json.JSONDecodeError:
            return stdout

    except subprocess.TimeoutExpired:
        print(f"  Claude CLI timed out after {timeout_seconds}s")
        return None
    except FileNotFoundError:
        print(f"  Claude CLI binary not found at: {claude_bin}")
        return None
    except Exception as e:
        print(f"  Claude CLI error: {e}")
        return None
