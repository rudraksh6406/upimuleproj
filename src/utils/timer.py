"""Latency profiling and timer utilities."""
import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """Simple timer for latency profiling in milliseconds."""
    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return self.elapsed_ms

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


@contextmanager
def timeit_context(name: str = "operation") -> Generator[dict, None, None]:
    """Context manager for timing a block of code."""
    res = {"name": name, "elapsed_ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield res
    finally:
        t1 = time.perf_counter()
        res["elapsed_ms"] = (t1 - t0) * 1000.0
