"""Step modules for one ANNA v2 iteration.

Each module owns one phase:
    research.py    — RESEARCHER (conditional, every N iterations)
    propose.py     — ENGINEER (Propose)
    implement.py   — ENGINEER (Implement + Validate + Launch)
    wait.py        — Wait + AI-driven early stop
    collect.py     — Collect results + integrity gate + tree update
    judge.py       — JUDGE + retrospectives + insight re-examination
    checkpoint.py  — Persist loop state + reflective memory

Step bodies are verbatim cuts from the old ``Orchestrator.run()`` /
``_wait_for_completion`` / ``_run_judge`` methods. Step functions take an
``IterationContext`` as the first argument; nothing else has changed.
"""
