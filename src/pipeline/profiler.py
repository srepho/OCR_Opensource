"""GPU memory and timing profiler for OCR inference."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.adapters.base import InferenceProfile


@dataclass
class ModelProfile:
    """Aggregate profile for a model across all pages."""

    model_name: str
    total_pages: int = 0
    total_wall_time: float = 0.0
    page_times: list[float] = field(default_factory=list)
    peak_gpu_memory_mb: float = 0.0
    total_tokens: int = 0

    @property
    def avg_time_per_page(self) -> float:
        if not self.page_times:
            return 0.0
        return sum(self.page_times) / len(self.page_times)

    @property
    def median_time_per_page(self) -> float:
        if not self.page_times:
            return 0.0
        import statistics
        return statistics.median(self.page_times)

    @property
    def pages_per_minute(self) -> float:
        avg = self.avg_time_per_page
        if avg == 0:
            return 0.0
        return 60.0 / avg

    def add_page_profile(self, profile: InferenceProfile) -> None:
        self.total_pages += 1
        self.total_wall_time += profile.wall_time_seconds
        self.page_times.append(profile.wall_time_seconds)
        self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, profile.gpu_memory_peak_mb)
        self.total_tokens += profile.tokens_generated

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_pages": self.total_pages,
            "total_wall_time_seconds": round(self.total_wall_time, 2),
            "avg_time_per_page": round(self.avg_time_per_page, 3),
            "median_time_per_page": round(self.median_time_per_page, 3),
            "pages_per_minute": round(self.pages_per_minute, 1),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 1),
            "total_tokens": self.total_tokens,
        }


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def get_gpu_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def reset_gpu_peak_memory() -> None:
    """Reset GPU peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


@contextmanager
def gpu_memory_tracker():
    """Context manager that tracks GPU memory during a block.

    Usage:
        with gpu_memory_tracker() as tracker:
            # do inference
        print(f"Peak: {tracker.peak_mb} MB")
    """
    tracker = _GPUTracker()
    reset_gpu_peak_memory()
    tracker.start_mb = get_gpu_memory_mb()
    yield tracker
    tracker.peak_mb = get_gpu_peak_memory_mb()
    tracker.end_mb = get_gpu_memory_mb()


class _GPUTracker:
    def __init__(self):
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.peak_mb = 0.0

    @property
    def delta_mb(self) -> float:
        return self.end_mb - self.start_mb


@contextmanager
def timer():
    """Context manager for wall-clock timing.

    Usage:
        with timer() as t:
            # do work
        print(f"Took {t.elapsed:.2f}s")
    """
    t = _Timer()
    t.start = time.perf_counter()
    yield t
    t.end = time.perf_counter()
    t.elapsed = t.end - t.start


class _Timer:
    def __init__(self):
        self.start = 0.0
        self.end = 0.0
        self.elapsed = 0.0
