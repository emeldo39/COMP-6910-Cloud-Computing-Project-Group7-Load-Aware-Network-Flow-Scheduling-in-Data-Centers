"""
LAFS — Workload package
========================
COMP 6910 — Group 7

Exports:
    Flow                    — network flow dataclass (5-tuple + metadata)
    MICE_THRESHOLD_BYTES    — < 100 KB
    ELEPHANT_THRESHOLD_BYTES— >= 1 MB
"""

from src.workload.flow import Flow, MICE_THRESHOLD_BYTES, ELEPHANT_THRESHOLD_BYTES

__all__ = ["Flow", "MICE_THRESHOLD_BYTES", "ELEPHANT_THRESHOLD_BYTES"]
