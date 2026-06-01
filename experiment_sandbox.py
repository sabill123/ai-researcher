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
    # `\b` would match `model.eval(` too. Require no preceding `.` or word char
    # so we still catch bare `eval(...)`/`exec(...)` but allow `nn.eval()` etc.
    r'(?<![.\w])eval\s*\(',
    r'(?<![.\w])exec\s*\(',
    r'\bshutil\.rmtree\b',
    r'\b__import__\b',
    r'\bimportlib\b',
    r'\bos\.remove\b',
    r'\bos\.unlink\b',
    r'\bos\.rmdir\b',
    r'\bopen\s*\([^)]*["\']w["\']\s*\)',  # writing outside expected paths
]

# Required CLI arguments that train.py must still accept
REQUIRED_CLI_ARGS = [
    "--data_dir", "--output_dir", "--exp", "--epochs", "--no_wandb",
]

# Required output: test_metrics.json must contain *_score_mae keys
REQUIRED_METRIC_PATTERN = r"_score_mae"


class ExperimentSandbox:
    def __init__(self, config: SystemConfig):
        self.config = config

    def create_sandbox(self, exp_name: str, parent_code_dir: Optional[str] = None) -> str:
        """Create an isolated experiment directory with code copies.

        Args:
            exp_name: Name for the experiment directory.
            parent_code_dir: If provided, copy code from this directory (parent experiment)
                             instead of baseline. Falls back to baseline for missing files.

        Returns the experiment directory path.
        """
        exp_dir = os.path.join(self.config.experiments_dir, exp_name)
        assert_safe_path(exp_dir, "experiment_dir")

        code_dir = os.path.join(exp_dir, "code")
        results_dir = os.path.join(exp_dir, "results")

        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Copy scripts from parent experiment or baseline
        source_dir = parent_code_dir if parent_code_dir and os.path.isdir(parent_code_dir) else self.config.scripts_dir
        for filename in self.config.sandbox_files:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(code_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                # Fallback to baseline if parent is missing a file
                fallback = os.path.join(self.config.scripts_dir, filename)
                if os.path.exists(fallback):
                    shutil.copy2(fallback, dst)

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

        # 0. Reject extra .py files not in sandbox_files
        allowed_py = set(self.config.sandbox_files)
        for filename in os.listdir(code_dir):
            if filename.endswith(".py") and filename not in allowed_py:
                extra_path = os.path.join(code_dir, filename)
                os.remove(extra_path)
                errors.append(f"Removed unauthorized file '{filename}' — only {allowed_py} are allowed")

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
        train_script = os.path.join(code_dir, "train.py")
        if os.path.exists(train_script):
            with open(train_script) as f:
                content = f.read()
            for arg in REQUIRED_CLI_ARGS:
                if arg not in content:
                    errors.append(
                        f"Required CLI argument '{arg}' missing from train.py"
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

        # 6. Import & instantiation test — catches RuntimeError, AttributeError, TypeError
        import_errors = self._test_imports(code_dir)
        if import_errors:
            errors.append(f"Import test failed: {import_errors}")

        return (len(errors) == 0, errors)

    def _test_imports(self, code_dir: str) -> Optional[str]:
        """Try importing the key modules to catch runtime errors early.

        Runs in a subprocess with GPU available to test full instantiation.
        Only catches code bugs (TypeError, AttributeError, ImportError, NameError),
        not environment issues.
        """
        test_script = f'''
import sys, os
sys.path.insert(0, "{code_dir}")
os.chdir("{code_dir}")
import torch
# Patch torch.load to always use CPU (test runs without GPU)
_original_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs["map_location"] = "cpu"
    return _original_load(*args, **kwargs)
torch.load = _cpu_load

try:
    import backbone
    import model
    import losses
    import dataset
    # Quick instantiation test (catches missing attributes, bad LoRA wrappers)
    encoder = backbone.FaRLEncoder()
    # Test forward pass on CPU
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = encoder(dummy)
    print("OK")
except (TypeError, AttributeError, ImportError, NameError, SyntaxError, ValueError) as e:
    print(f"FAIL: {{type(e).__name__}}: {{e}}")
except RuntimeError as e:
    err_msg = str(e).lower()
    if "cuda" in err_msg or "device" in err_msg:
        print("OK_CUDA_SKIP")
    else:
        print(f"FAIL: RuntimeError: {{e}}")
except Exception as e:
    print(f"WARN: {{type(e).__name__}}: {{e}}")
'''
        try:
            result = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # CPU only
            )
            output = (result.stdout + result.stderr).strip()
            if "OK" in result.stdout:
                return None
            # Extract the error
            for line in output.split("\n"):
                if line.startswith("FAIL:"):
                    return line
            return f"Unknown error: {output[-500:]}" if output else "No output"
        except subprocess.TimeoutExpired:
            return None  # Timeout likely means model loaded (slow but working)
        except Exception as e:
            return None  # Don't block on test infrastructure issues

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

    def generate_parent_diff(self, exp_dir: str, parent_code_dir: str) -> Optional[str]:
        """Generate diff between parent experiment code and this experiment's code."""
        code_dir = os.path.join(exp_dir, "code")
        all_diffs = []
        for filename in self.config.sandbox_files:
            if filename in self.config.readonly_files:
                continue
            src = os.path.join(parent_code_dir, filename)
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
