"""
LAFS — Workload package
========================
COMP-6910 — Group 7

Exports
-------
Flow                        — network flow dataclass (5-tuple + metadata)
MICE_THRESHOLD_BYTES        — < 100 KB (mice classification)
ELEPHANT_THRESHOLD_BYTES    — >= 1 MB  (elephant in Flow class)

FacebookWebSearchGenerator  — empirical CDF from Benson et al. IMC 2010
FacebookWebSearchConfig     — configuration dataclass

AllReduceGenerator          — synthetic ring/PS AllReduce for ML training
AllReduceConfig             — configuration dataclass

MicroserviceRPCGenerator    — Google-style RPC chain / fan-out workloads
MicroserviceConfig          — configuration dataclass
ServiceGraph                — service call DAG (chain, fan_out, mixed)

WorkloadRunner              — unified multi-generator orchestrator
WorkloadConfig              — top-level configuration
WorkloadStats               — summary statistics (FCT inputs, Jain's index)
"""

from src.workload.flow import Flow, MICE_THRESHOLD_BYTES, ELEPHANT_THRESHOLD_BYTES
from src.workload.facebook_websearch import (
    FacebookWebSearchGenerator,
    FacebookWebSearchConfig,
)
from src.workload.allreduce import AllReduceGenerator, AllReduceConfig
from src.workload.microservice import (
    MicroserviceRPCGenerator,
    MicroserviceConfig,
    ServiceGraph,
)
from src.workload.runner import WorkloadRunner, WorkloadConfig, WorkloadStats

__all__ = [
    "Flow",
    "MICE_THRESHOLD_BYTES",
    "ELEPHANT_THRESHOLD_BYTES",
    "FacebookWebSearchGenerator",
    "FacebookWebSearchConfig",
    "AllReduceGenerator",
    "AllReduceConfig",
    "MicroserviceRPCGenerator",
    "MicroserviceConfig",
    "ServiceGraph",
    "WorkloadRunner",
    "WorkloadConfig",
    "WorkloadStats",
]
