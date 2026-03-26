"""
LAFS -- Optimizer package
=========================
COMP 6910 -- Group 7

Exports
-------
MILPConfig        -- Solver configuration (backend, time limit, gap, weights)
MILPResult        -- Result dataclass: assignments, max_util, solve_time, status
LAFSMILPSolver    -- Core MILP formulation (PuLP/CBC or Gurobi backend)
LAFSScheduler     -- Drop-in BaseScheduler subclass wrapping the MILP solver
"""

from src.optimizer.milp_solver import MILPConfig, MILPResult, LAFSMILPSolver
from src.optimizer.lafs_scheduler import LAFSScheduler

__all__ = [
    "MILPConfig",
    "MILPResult",
    "LAFSMILPSolver",
    "LAFSScheduler",
]
