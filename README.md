# ANNA - Automated Neural Network Architect

A **3-agent AI research system** that autonomously runs ML experiments with paper searching, code modification, and result analysis. Powered by [Claude Code CLI](https://github.com/anthropics/claude-code).

## Architecture

```
                    ┌──────────────────┐
                    │  Knowledge Base  │
                    │  (@knowledge/)   │
                    └────────┬─────────┘
                             │ inject domain knowledge
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                      ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Researcher  │────▶│  Engineer    │────▶│    Judge      │
│  (Agent 1)   │     │  (Agent 2)   │     │  (Agent 3)   │
│              │     │              │     │              │
│ Paper search │     │ 1. Propose   │     │ Analyze      │
│ Technique    │     │ 2. Risk gate │     │ Post-action  │
│ discovery    │     │ 3. Implement │     │ feedback     │
│              │     │ 4. Validate  │     │ Set next     │
│              │     │ 5. Launch    │     │ direction    │
└─────────────┘     └──────┬───────┘     └──────────────┘
       ▲                   │                     │
       │            ┌──────▼───────┐             │
       │            │ Approval Gate│             │
       │            │ (risky exp)  │             │
       │            └──────────────┘             │
       └────────── Feedback Loop ◀───────────────┘
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
- **Knowledge Base**: Domain knowledge files (`@knowledge/file.md#section`) auto-injected into agent prompts
- **Post-Action Feedback**: Technique performance tracking, parent lineage analysis, stagnation detection
- **Human Approval Gate**: Risk assessment blocks dangerous experiments until human approval via CLI
- **Safety Validation**: Syntax checking, dangerous pattern scanning, CLI interface preservation
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

### 2. Create knowledge files

Populate `anna_v2/knowledge/` with domain-specific markdown files:

```
knowledge/
├── actions.md           # Target actions, severity criteria, difficulty
├── architecture.md      # Model architecture, code structure, constraints
├── performance.md       # Baselines, targets, current best
├── techniques.md        # Tried/untried techniques, effectiveness
├── data-constraints.md  # Dataset size, class imbalance, known issues
└── loss-functions.md    # Loss function guide and combinations
```

Agents auto-inject relevant sections using `@knowledge/file.md#section` references. Use `##` and `###` headings to define extractable sections.

### 3. Customize agent prompts

Edit the agent prompts in `anna/agents/` to match your task:
- `researcher.py` — What kind of techniques to search for
- `engineer.py` — Your codebase structure, valid argparse choices, constraints
- `judge.py` — What metrics to analyze, what insights to generate

### 4. Bootstrap insights (optional)

Edit `anna/reflective_memory.py` → `BOOTSTRAP_INSIGHTS` with known insights about your project.

### 5. Run

```bash
# Initialize (imports existing experiment results)
python -m anna init

# Start the 3-agent loop
python -m anna run --max-iterations 50

# Monitor
python -m anna status
python -m anna report
python -m anna gpus

# Approval gate (for risky experiments)
python -m anna pending                          # List pending approvals
python -m anna approve <checkpoint_id>          # Approve
python -m anna reject <checkpoint_id> --reason "too risky"  # Reject

# Resume after human review checkpoint
python -m anna resume
```

## How It Works

Each iteration follows a 7-step pipeline:

1. **STEP 1 — Research** (every N iterations): Researcher agent searches papers for new ML techniques relevant to the current bottleneck. Knowledge Base provides model/data context via `@knowledge/architecture.md#모델 요약`.

2. **STEP 2 — Propose**: Engineer agent proposes 2 experiments. Uses Knowledge Base for code structure (`@knowledge/architecture.md#코드 구조`), constraints, and hard action resolution guides. Selects parent experiment via MCTS-inspired UCB1 scoring.

3. **STEP 3 — Implement & Launch**: For each proposed experiment:
   - **Risk Assessment** — checks if the experiment needs human approval (backbone changes, low confidence, repeated technique failures)
   - If risky: creates a checkpoint and skips (user approves/rejects via CLI)
   - If safe: creates isolated sandbox, implements code changes via Claude Code CLI
   - **Validation** — syntax check, safety scan, CLI interface preservation
   - **Auto bug-fix** — if validation fails, attempts one automated fix
   - Launches training on available GPU

4. **STEP 4 — Wait**: Monitors GPU processes and training progress (polls every 60s).

5. **STEP 5 — Collect**: Parses `test_metrics.json` from each completed experiment. Extracts per-action MAE scores and computes averages.

6. **STEP 6 — Judge**: Judge agent analyzes all results. Receives:
   - Top 15 experiments by performance
   - Latest iteration results with parent-vs-child deltas
   - **Post-Action Feedback** — technique type success rates, parent lineage limits, stagnation detection, action-specific regressions
   - Accumulated research insights
   - Knowledge Base analysis guide (`@knowledge/data-constraints.md#분석 가이드`)

   Outputs: analysis, new insights, and next direction (exploit vs. explore).

7. **STEP 7 — Checkpoint**: Saves state. Pauses every N iterations for human review.

### Knowledge Base System

Agents load domain knowledge at prompt-build time via `@knowledge/` references:

```
@knowledge/actions.md#이마_주름       → extracts just the 이마_주름 section
@knowledge/architecture.md           → loads full file
@knowledge/loss-functions.md#유망한 Loss Functions → specific section
```

Section extraction is heading-level aware — a `###` section includes content until the next heading of same or higher level.

### Post-Action Feedback Loop

Before the Judge runs, the orchestrator evaluates recent experiment performance:

| Analysis | Trigger | Effect |
|----------|---------|--------|
| Technique failure streak | Same `code_change_type` regresses 3x in a row | Judge warned to pivot |
| Parent lineage limit | 3+ children of same parent all regressed | Suggest new parent |
| Stagnation detection | Best MAE unchanged for 8+ experiments | Force EXPLORE strategy |
| Action-specific regression | An action worsens in 4/5 recent experiments | Targeted intervention |

### Human Approval Gate

Before implementing a risky experiment, the system creates a checkpoint and waits:

| Risk Condition | Example |
|---------------|---------|
| Backbone modification | Changes to `backbone.py` (pretrained weights) |
| Low confidence | Proposal confidence < 0.3 |
| Repeated failure | Same technique type failed 2+ recent times |

Approve/reject via CLI:
```bash
python -m anna pending                    # See what's waiting
python -m anna approve abc123             # Green-light the experiment
python -m anna reject abc123 --reason "..." # Block with feedback
```

## Project Structure

```
anna/                           # Python package (ai-researcher/anna/)
├── __init__.py
├── __main__.py
├── cli.py                      # CLI: run/status/approve/reject/pending
├── config.py                   # Paths, GPU settings, budget limits
├── orchestrator.py             # Main 3-agent loop + feedback + approval gate
├── knowledge_base.py           # @knowledge/ reference resolver
├── experiment_sandbox.py       # Code isolation & validation
├── experiment_runner.py        # Training subprocess management
├── experiment_db.py            # SQLite: experiments + checkpoints tables
├── experiment_tree.py          # MCTS-inspired direction tracking (UCB1)
├── gpu_manager.py              # GPU detection & reservation
├── reflective_memory.py        # Cross-experiment insight accumulation
└── agents/
    ├── base_agent.py           # Claude Code CLI wrapper
    ├── researcher.py           # Paper/technique search agent
    ├── engineer.py             # Code modification agent
    └── judge.py                # Result analysis agent

anna_v2/knowledge/              # Domain knowledge files
├── actions.md                  # 6 target actions, severity criteria
├── architecture.md             # FaRL ViT-B/16, code structure, constraints
├── performance.md              # Baselines, targets, current best
├── techniques.md               # Tried/untried techniques
├── data-constraints.md         # Dataset size, class imbalance
└── loss-functions.md           # Loss function guide
```

## Safety Features

- **Filesystem restriction**: All operations confined to `ALLOWED_BASE_PATH`
- **Sandbox validation**: Syntax check, dangerous pattern scan (no `os.system`, `subprocess`, `eval`, etc.)
- **Read-only files**: Specified files (e.g., metric utils) cannot be modified
- **CLI interface preservation**: Training script must always accept required arguments
- **Human Approval Gate**: Risky experiments (backbone changes, low confidence, repeated failures) require explicit CLI approval before execution
- **Auto bug-fix**: One retry on validation failure before giving up
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
- Knowledge Base & Approval Gate: Inspired by [meditherapy-platform](https://github.com/meditherapy-platform) WorkflowContext

## License

MIT
