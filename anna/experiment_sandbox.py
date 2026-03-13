"""
Experiment sandbox: isolates each experiment with its own code copy.
Handles code copying, symlink creation, validation, and diff generation.
"""

import os
import py_compile
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

from .config import SystemConfig, assert_safe_path


# Dangerous patterns that must never appear in modified code
DANGEROUS_PATTERNS = [
    r'\bos\.system\b',
    r'\bsubprocess\.(run|call|Popen|check_output|check_call)\b',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bshutil\.rmtree\b',
    r'\b__import__\b',
    r'\bimportlib\b',
    r'\bos\.remove\b',
    r'\bos\.unlink\b',
    r'\bos\.rmdir\b',
    r'\bopen\s*\([^)]*["\']w["\']\s*\)',  # writing outside expected paths
]

# Required CLI arguments that train_cloc_v2.py must still accept
REQUIRED_CLI_ARGS = [
    "--data_dir", "--output_dir", "--exp", "--epochs", "--no_wandb",
]

# Required output: test_metrics.json must contain *_score_mae keys
REQUIRED_METRIC_PATTERN = r"_score_mae"


class ExperimentSandbox:
    def __init__(self, config: SystemConfig):
        self.config = config

    def create_sandbox(self, exp_name: str) -> str:
        """Create an isolated experiment directory with code copies.

        Returns the experiment directory path.
        """
        exp_dir = os.path.join(self.config.experiments_dir, exp_name)
        assert_safe_path(exp_dir, "experiment_dir")

        code_dir = os.path.join(exp_dir, "code")
        results_dir = os.path.join(exp_dir, "results")

        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Copy base scripts
        for filename in self.config.sandbox_files:
            src = os.path.join(self.config.scripts_dir, filename)
            dst = os.path.join(code_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Symlink pretrained checkpoints (shared, read-only)
        # backbone.py uses: os.path.join(script_dir, "..", "pretrained_ckpts", ...)
        # So the symlink must be at exp_dir/pretrained_ckpts (parent of code/)
        ckpts_link = os.path.join(exp_dir, "pretrained_ckpts")
        if not os.path.exists(ckpts_link) and os.path.exists(self.config.pretrained_ckpts_dir):
            os.symlink(self.config.pretrained_ckpts_dir, ckpts_link)

        return exp_dir

    def validate_sandbox(self, exp_dir: str) -> Tuple[bool, List[str]]:
        """Validate modified code in sandbox.

        Returns (is_valid, list_of_errors).
        """
        code_dir = os.path.join(exp_dir, "code")
        errors = []

        # 1. Syntax check all .py files
        for filename in os.listdir(code_dir):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(code_dir, filename)
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"Syntax error in {filename}: {e}")

        # 2. Dangerous pattern scan — only check ADDED lines (diff)
        for filename in self.config.sandbox_files:
            if filename in self.config.readonly_files:
                continue
            src = os.path.join(self.config.scripts_dir, filename)
            dst = os.path.join(code_dir, filename)
            if not os.path.exists(src) or not os.path.exists(dst):
                continue
            with open(src) as f:
                original_lines = set(f.readlines())
            with open(dst) as f:
                current_lines = f.readlines()
            # Only check lines that were added (not in original)
            added_content = "\n".join(
                line for line in current_lines if line not in original_lines
            )
            if not added_content.strip():
                continue
            for pattern in DANGEROUS_PATTERNS:
                matches = re.findall(pattern, added_content)
                if matches:
                    errors.append(
                        f"Dangerous pattern '{pattern}' in new code in {filename}: {matches}"
                    )

        # 3. CLI interface preservation check
        train_script = os.path.join(code_dir, "train_cloc_v2.py")
        if os.path.exists(train_script):
            with open(train_script) as f:
                content = f.read()
            for arg in REQUIRED_CLI_ARGS:
                if arg not in content:
                    errors.append(
                        f"Required CLI argument '{arg}' missing from train_cloc_v2.py"
                    )

        # 4. Read-only files must not be modified
        for readonly_file in self.config.readonly_files:
            src = os.path.join(self.config.scripts_dir, readonly_file)
            dst = os.path.join(code_dir, readonly_file)
            if os.path.exists(src) and os.path.exists(dst):
                with open(src) as f:
                    original = f.read()
                with open(dst) as f:
                    current = f.read()
                if original != current:
                    errors.append(f"Read-only file '{readonly_file}' was modified")

        # 5. Diff size check
        diff = self.generate_diff(exp_dir)
        if diff:
            diff_lines = diff.split('\n')
            changed_lines = sum(
                1 for l in diff_lines
                if l.startswith('+') or l.startswith('-')
            )
            if changed_lines > 500:
                errors.append(
                    f"Code diff too large: {changed_lines} changed lines (max 500)"
                )

        return (len(errors) == 0, errors)

    def generate_diff(self, exp_dir: str) -> Optional[str]:
        """Generate unified diff between original scripts and sandbox code (per file)."""
        code_dir = os.path.join(exp_dir, "code")
        all_diffs = []
        for filename in self.config.sandbox_files:
            if filename in self.config.readonly_files:
                continue
            src = os.path.join(self.config.scripts_dir, filename)
            dst = os.path.join(code_dir, filename)
            if not os.path.exists(src) or not os.path.exists(dst):
                continue
            try:
                result = subprocess.run(
                    ["diff", "-u", src, dst],
                    capture_output=True, text=True, timeout=10,
                )
                if result.stdout:
                    all_diffs.append(result.stdout)
            except Exception:
                continue
        return "\n".join(all_diffs) if all_diffs else None

    def save_diff(self, exp_dir: str) -> Optional[str]:
        """Save diff to file and return the diff content."""
        diff = self.generate_diff(exp_dir)
        if diff:
            diff_path = os.path.join(exp_dir, "code_diff.patch")
            with open(diff_path, "w") as f:
                f.write(diff)
        return diff

    def restore_readonly_files(self, exp_dir: str):
        """Restore read-only files to their original state."""
        code_dir = os.path.join(exp_dir, "code")
        for readonly_file in self.config.readonly_files:
            src = os.path.join(self.config.scripts_dir, readonly_file)
            dst = os.path.join(code_dir, readonly_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
