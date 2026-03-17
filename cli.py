"""
CLI entrypoint for ANNA v2.

Usage:
    python -m anna_v2 init
    python -m anna_v2 run [--max-iterations N]
    python -m anna_v2 resume
    python -m anna_v2 status
    python -m anna_v2 report
    python -m anna_v2 gpus
"""

import argparse

from .config import SystemConfig
from .orchestrator import Orchestrator


def cmd_init(config: SystemConfig):
    orch = Orchestrator(config)
    orch.initialize()


def cmd_run(config: SystemConfig, max_iterations: int):
    orch = Orchestrator(config)
    if orch.db.get_total_count() == 0:
        orch.initialize()
    orch.run(max_iterations=max_iterations)


def cmd_resume(config: SystemConfig):
    orch = Orchestrator(config)
    orch.resume()


def cmd_status(config: SystemConfig):
    orch = Orchestrator(config)
    orch.show_status()


def cmd_report(config: SystemConfig):
    orch = Orchestrator(config)
    orch.show_report()


def cmd_gpus(config: SystemConfig):
    from .gpu_manager import GPUManager
    gpu_manager = GPUManager()
    print(gpu_manager.format_status())

    available = gpu_manager.get_available_gpus(
        min_free_memory_mb=config.min_free_memory_mb,
        max_utilization_pct=config.max_gpu_utilization_pct,
    )
    print(f"\nAvailable for experiments (>={config.min_free_memory_mb}MB free, "
          f"<={config.max_gpu_utilization_pct}% util): {available or 'None'}")


def main():
    parser = argparse.ArgumentParser(
        prog="anna",
        description="ANNA v2 - Automated Neural Network Architect (3-Agent Research System)"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize: import existing results")

    run_p = subparsers.add_parser("run", help="Start 3-agent research loop")
    run_p.add_argument("--max-iterations", type=int, default=50)
    run_p.add_argument("--budget-hours", type=float, default=None)

    subparsers.add_parser("resume", help="Resume from checkpoint")
    subparsers.add_parser("status", help="Show current status")
    subparsers.add_parser("report", help="Generate summary report")
    subparsers.add_parser("gpus", help="Show GPU status")

    args = parser.parse_args()
    config = SystemConfig()

    if args.command == "init":
        cmd_init(config)
    elif args.command == "run":
        if args.budget_hours:
            config.budget_gpu_hours = args.budget_hours
        cmd_run(config, args.max_iterations)
    elif args.command == "resume":
        cmd_resume(config)
    elif args.command == "status":
        cmd_status(config)
    elif args.command == "report":
        cmd_report(config)
    elif args.command == "gpus":
        cmd_gpus(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
