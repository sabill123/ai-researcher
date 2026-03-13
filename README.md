# ANNA - Automated Neural Network Architect

A **3-agent AI research system** that autonomously runs ML experiments with paper searching, code modification, and result analysis. Powered by [Claude Code CLI](https://github.com/anthropics/claude-code).

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Researcher  │────▶│  Engineer    │────▶│   Judge     │
│  (Agent 1)   │     │  (Agent 2)   │     │  (Agent 3)  │
│              │     │              │     │             │
│ Paper search │     │ 1. Propose   │     │ Analyze     │
│ Technique    │     │ 2. Implement │     │ Generate    │
│ discovery    │     │ 3. Validate  │     │ insights    │
│              │     │ 4. Launch    │     │ Set next    │
│              │     │              │     │ direction   │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       └───────── Feedback Loop ◀──────────────┘
```

### Agent Roles

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Searches papers for new techniques | Web search enabled |
| **Engineer** | Proposes and implements code changes | File editing in sandboxed environment |
| **Judge** | Analyzes results, generates insights, sets research direction | Pure reasoning |

### Key Features

- **Experiment Isolation**: Each experiment gets its own code copy (sandbox), preventing interference
- **Code-Level Changes**: Not just hyperparameter tuning — the system modifies model architecture, loss functions, augmentation, and training strategies
- **MCTS-Inspired Exploration**: UCB1-based tree search balances exploring new directions vs. exploiting known good ones
- **Reflective Memory**: Accumulates insights across experiments (inspired by [MARS paper](https://arxiv.org/abs/2506.11140))
- **Safety Validation**: Syntax checking, dangerous pattern scanning, CLI interface preservation, diff size limits
- **Human Review Checkpoints**: Pauses every N iterations for human oversight

## Prerequisites

- **Claude Code CLI** installed and authenticated (`npm install -g @anthropic-ai/claude-code` or VS Code extension)
- **Python 3.8+**
- **GPU server** with `nvidia-smi` available
- A training script that accepts CLI arguments and outputs `test_metrics.json`

## Quick Start

### 1. Configure for your project

Edit `anna/config.py` to match your project:

```python
@dataclass
class SystemConfig:
    project_root: str = "/path/to/your/project"
    data_dir: str = "/path/to/your/dataset"
    target_actions: str = "target"
    epochs: int = 100

    # Files to copy into each experiment sandbox
    sandbox_files: List[str] = field(default_factory=lambda: [
        "train.py",
        "model.py",
        "losses.py",
        "dataset.py",
        # ... your training scripts
    ])

    # Files the engineer cannot modify
    readonly_files: List[str] = field(default_factory=lambda: [
        "utils.py",  # e.g., metric computation
    ])
```

Also update `ALLOWED_BASE_PATH` to restrict filesystem access.

### 2. Customize agent prompts

Edit the agent prompts in `anna/agents/` to match your task:
- `researcher.py` — What kind of techniques to search for
- `engineer.py` — Your codebase structure, valid argparse choices, constraints
- `judge.py` — What metrics to analyze, what insights to generate

### 3. Bootstrap insights (optional)

Edit `anna/reflective_memory.py` → `BOOTSTRAP_INSIGHTS` with known insights about your project.

### 4. Run

```bash
# Initialize (imports existing experiment results)
python -m anna init

# Start the 3-agent loop
python -m anna run --max-iterations 50

# Monitor
python -m anna status
python -m anna report
python -m anna gpus

# Resume after human review checkpoint
python -m anna resume
```

## How It Works

Each iteration:

1. **Researcher** (every N iterations) searches for new techniques via web search
2. **Engineer** proposes 2 experiments based on research + existing insights
3. For each experiment:
   - Creates isolated sandbox (copies training code)
   - Implements code changes via Claude Code CLI (file editing)
   - Validates: syntax check, safety scan, CLI interface preservation
   - Launches training on available GPU
4. **Waits** for training completion (monitors progress)
5. **Collects** results (parses `test_metrics.json`)
6. **Judge** analyzes results, generates insights, decides next direction
7. **Checkpoint** — saves state, optionally pauses for human review

## Project Structure

```
anna/
├── __init__.py
├── __main__.py
├── cli.py                  # CLI entrypoint
├── config.py               # System configuration
├── orchestrator.py          # Main 3-agent loop
├── experiment_sandbox.py    # Code isolation & validation
├── experiment_runner.py     # Training subprocess management
├── experiment_db.py         # SQLite experiment database
├── experiment_tree.py       # MCTS-inspired direction tracking
├── gpu_manager.py           # GPU detection & reservation
├── reflective_memory.py     # Cross-experiment insight accumulation
└── agents/
    ├── base_agent.py        # Claude Code CLI wrapper
    ├── researcher.py        # Paper/technique search agent
    ├── engineer.py          # Code modification agent
    └── judge.py             # Result analysis agent
```

## Safety Features

- **Filesystem restriction**: All operations confined to `ALLOWED_BASE_PATH`
- **Sandbox validation**: Syntax check, dangerous pattern scan (no `os.system`, `subprocess`, `eval`, etc.)
- **Read-only files**: Specified files (e.g., metric utils) cannot be modified
- **CLI interface preservation**: Training script must always accept required arguments
- **Diff size limit**: Max 500 changed lines per experiment
- **Budget limits**: GPU hours and Claude API cost limits
- **Shared server safety**: Never kills processes, only uses available GPUs

## Adapting to Your Project

ANNA was originally built for facial palsy severity prediction, but the architecture is general-purpose. To adapt:

1. **Config**: Set your paths, training files, and metric targets
2. **Researcher prompts**: Describe your task and what techniques to search for
3. **Engineer prompts**: Describe your codebase structure and constraints
4. **Judge prompts**: Define what metrics matter and what analysis to perform
5. **Bootstrap insights**: Seed with known insights about your project
6. **Result parsing**: Update `experiment_runner.py` if your metrics format differs

## References

- [MARS: Modular Agent with Reflective Search](https://arxiv.org/abs/2506.11140)
- [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- [Claude Code](https://github.com/anthropics/claude-code)

## License

MIT
