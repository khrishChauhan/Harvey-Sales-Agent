"""Hardware detection and local model recommendation for Harvey Local AI."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger("sellbird.hardware")


@dataclass
class HardwareProfile:
    """Hardware profile of the local machine."""
    total_ram_gb: float
    available_ram_gb: float
    cpu_model: str
    cpu_cores: int
    has_nvidia_gpu: bool
    gpu_model: str
    total_vram_mb: int
    free_vram_mb: int
    recommended_model: str
    recommended_context_length: int
    offload_mode: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_ram() -> tuple[float, float]:
    """Detect total and available RAM in GB.

    Uses Windows Win32 GlobalMemoryStatusEx, falling back to os/sysconf on POSIX.
    """
    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = round(stat.ullTotalPhys / (1024**3), 2)
                avail = round(stat.ullAvailPhys / (1024**3), 2)
                return total, avail
        except Exception as e:
            logger.debug(f"Win32 memory query failed: {e}")

    # Fallback
    try:
        total = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2)
        avail = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / (1024**3), 2)
        return total, avail
    except Exception:
        return 16.0, 8.0  # safe conservative fallback


def detect_cpu() -> tuple[str, int]:
    """Detect CPU model and logical core count."""
    cores = os.cpu_count() or 4
    model = platform.processor() or platform.machine() or "Generic CPU"
    return model, cores


def detect_nvidia_gpu() -> tuple[bool, str, int, int]:
    """Detect NVIDIA GPU using nvidia-smi.

    Returns (has_gpu, gpu_model, total_vram_mb, free_vram_mb).
    Never raises exceptions if nvidia-smi is unavailable or fails.
    """
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False, "None", 0, 0

    try:
        result = subprocess.run(
            [nvsmi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        output = result.stdout.strip()
        if not output:
            return False, "None", 0, 0

        first_line = output.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) >= 3:
            name = parts[0]
            total_vram = int(parts[1])
            free_vram = int(parts[2])
            return True, name, total_vram, free_vram

    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")

    return False, "None", 0, 0


def recommend_model(
    total_ram_gb: float,
    avail_ram_gb: float,
    has_gpu: bool,
    free_vram_mb: int,
) -> tuple[str, int, str, str]:
    """Provide realistic model recommendation based on detected resources.

    Returns (recommended_model, recommended_context_length, offload_mode, rationale).
    Does NOT assume full VRAM fit.
    """
    # Target model requested: qwen3:4b
    target = "qwen3:4b"
    # Qwen3:4B (Q4_K_M) weights require ~2.6 GB.
    # A 2048 token KV cache + CUDA runtime overhead requires ~0.8 GB.
    # Total VRAM footprint is approximately 3.4 GB.

    if has_gpu and free_vram_mb >= 3600:
        return (
            target,
            2048,
            "100% GPU Offload",
            f"GPU has {free_vram_mb} MB free VRAM. {target} (Q4_K_M ~2.6 GB + 2048 ctx KV ~0.8 GB) "
            "fits in GPU memory with low paging overhead."
        )
    elif has_gpu and free_vram_mb >= 2500:
        return (
            target,
            2048,
            "Partial GPU Offload (Hybrid GPU + CPU)",
            f"GPU has {free_vram_mb} MB free VRAM. {target} will offload ~20 layers to GPU VRAM and "
            f"the remaining layers to {avail_ram_gb:.1f} GB available system RAM."
        )
    elif has_gpu and free_vram_mb >= 1500:
        return (
            "qwen2.5:1.5b",
            2048,
            "100% GPU Offload (Lightweight Alternative)",
            f"GPU free VRAM ({free_vram_mb} MB) is constrained for a 4B model. "
            "qwen2.5:1.5b (~1.2 GB VRAM) is recommended for fast inference without swapping."
        )
    else:
        return (
            target,
            2048,
            "CPU-Only Inference",
            f"No dedicated GPU VRAM available. Running on CPU using {avail_ram_gb:.1f} GB available RAM."
        )


def detect_hardware() -> HardwareProfile:
    """Run full hardware scan and compute recommendation."""
    total_ram, avail_ram = detect_ram()
    cpu_model, cpu_cores = detect_cpu()
    has_gpu, gpu_model, total_vram, free_vram = detect_nvidia_gpu()

    rec_model, rec_ctx, offload_mode, rationale = recommend_model(
        total_ram, avail_ram, has_gpu, free_vram
    )

    return HardwareProfile(
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        has_nvidia_gpu=has_gpu,
        gpu_model=gpu_model,
        total_vram_mb=total_vram,
        free_vram_mb=free_vram,
        recommended_model=rec_model,
        recommended_context_length=rec_ctx,
        offload_mode=offload_mode,
        rationale=rationale,
    )
