"""
GPU availability detection and allocation for shared server.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GPUStatus:
    id: int
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_pct: int


class GPUManager:
    def __init__(self, max_gpus: int = 8):
        self.max_gpus = max_gpus
        self._reservations: Dict[int, str] = {}  # gpu_id -> experiment_id

    def get_gpu_status(self) -> List[GPUStatus]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                gpus.append(GPUStatus(
                    id=int(parts[0]),
                    memory_total_mb=int(parts[1]),
                    memory_used_mb=int(parts[2]),
                    memory_free_mb=int(parts[3]),
                    utilization_pct=int(parts[4]),
                ))
            return gpus
        except Exception:
            return []

    def get_available_gpus(self, min_free_memory_mb: int = 20000,
                           max_utilization_pct: int = 10) -> List[int]:
        gpus = self.get_gpu_status()
        available = []
        for gpu in gpus:
            if gpu.id in self._reservations:
                continue
            if gpu.memory_free_mb >= min_free_memory_mb and \
               gpu.utilization_pct <= max_utilization_pct:
                available.append(gpu.id)
        return available

    def reserve_gpu(self, gpu_id: int, experiment_id: str) -> bool:
        if gpu_id in self._reservations:
            return False
        self._reservations[gpu_id] = experiment_id
        return True

    def release_gpu(self, gpu_id: int):
        self._reservations.pop(gpu_id, None)

    def release_by_experiment(self, experiment_id: str):
        to_remove = [
            gid for gid, eid in self._reservations.items()
            if eid == experiment_id
        ]
        for gid in to_remove:
            del self._reservations[gid]

    def wait_for_gpu(self, min_free_memory_mb: int = 20000,
                     max_utilization_pct: int = 10,
                     timeout_seconds: int = 1800,
                     poll_interval: int = 60) -> Optional[int]:
        elapsed = 0
        while elapsed < timeout_seconds:
            available = self.get_available_gpus(min_free_memory_mb, max_utilization_pct)
            if available:
                return available[0]
            time.sleep(poll_interval)
            elapsed += poll_interval
        return None

    def format_status(self) -> str:
        gpus = self.get_gpu_status()
        if not gpus:
            return "Failed to query GPU status"

        lines = [f"{'GPU':>4} {'Used':>8} {'Free':>8} {'Total':>8} {'Util':>6} {'Reserved':>10}"]
        lines.append("-" * 55)
        for gpu in gpus:
            reserved = self._reservations.get(gpu.id, "-")
            lines.append(
                f"{gpu.id:>4} {gpu.memory_used_mb:>7}M {gpu.memory_free_mb:>7}M "
                f"{gpu.memory_total_mb:>7}M {gpu.utilization_pct:>5}% {reserved:>10}"
            )
        return "\n".join(lines)
