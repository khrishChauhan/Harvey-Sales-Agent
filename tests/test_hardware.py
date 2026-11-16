"""Tests for hardware detection and local model recommendation."""

import pytest
from sellbird.hardware import (
    HardwareProfile,
    detect_cpu,
    detect_hardware,
    detect_ram,
    recommend_model,
)


def test_detect_ram():
    total_gb, avail_gb = detect_ram()
    assert total_gb > 0
    assert avail_gb > 0
    assert total_gb >= avail_gb


def test_detect_cpu():
    model, cores = detect_cpu()
    assert isinstance(model, str)
    assert len(model) > 0
    assert cores > 0


def test_detect_hardware_profile():
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb > 0
    assert profile.cpu_cores > 0
    assert profile.recommended_model in ("qwen3:4b", "qwen2.5:1.5b")
    dct = profile.to_dict()
    assert "total_ram_gb" in dct
    assert "available_ram_gb" in dct
    assert "has_nvidia_gpu" in dct
    assert "recommended_model" in dct


def test_recommend_model_ample_vram():
    # 4 GB VRAM with 3900 MB free -> fits 100% in GPU
    model, ctx, mode, rationale = recommend_model(
        total_ram_gb=16.0,
        avail_ram_gb=10.0,
        has_gpu=True,
        free_vram_mb=3900,
    )
    assert model == "qwen3:4b"
    assert ctx == 2048
    assert "100% GPU" in mode


def test_recommend_model_hybrid_offload():
    # 2800 MB free -> Hybrid GPU + CPU offload
    model, ctx, mode, rationale = recommend_model(
        total_ram_gb=16.0,
        avail_ram_gb=10.0,
        has_gpu=True,
        free_vram_mb=2800,
    )
    assert model == "qwen3:4b"
    assert "Partial GPU Offload" in mode
    assert "Hybrid" in mode


def test_recommend_model_constrained_vram():
    # 1800 MB free -> Lightweight alternative recommended
    model, ctx, mode, rationale = recommend_model(
        total_ram_gb=8.0,
        avail_ram_gb=4.0,
        has_gpu=True,
        free_vram_mb=1800,
    )
    assert model == "qwen2.5:1.5b"


def test_recommend_model_cpu_only():
    # No GPU detected
    model, ctx, mode, rationale = recommend_model(
        total_ram_gb=16.0,
        avail_ram_gb=8.0,
        has_gpu=False,
        free_vram_mb=0,
    )
    assert model == "qwen3:4b"
    assert "CPU-Only" in mode
