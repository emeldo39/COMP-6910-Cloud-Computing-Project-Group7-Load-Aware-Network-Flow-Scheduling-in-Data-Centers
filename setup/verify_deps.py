#!/usr/bin/env python3
"""
LAFS Project — Dependency Verification Script
COMP 6910 — Group 7

Checks all required Python packages, system tools, and services
needed to run the LAFS experiments.

Usage:
    python setup/verify_deps.py
    python setup/verify_deps.py --verbose
    python setup/verify_deps.py --fix     (attempts to install missing packages)
"""

import sys
import os
import subprocess
import importlib
import argparse
import shutil
from typing import Tuple, List

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"

results: List[Tuple[str, bool, str]] = []   # (name, passed, note)


def check(name: str, passed: bool, note: str = "") -> bool:
    symbol = OK if passed else FAIL
    note_str = f"  [{note}]" if note else ""
    print(f"  {symbol} {name}{note_str}")
    results.append((name, passed, note))
    return passed


def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}── {title} {'─' * (55 - len(title))}{RESET}")


# =============================================================================
# 1. Python version
# =============================================================================
section("Python Runtime")

py_ver = sys.version_info
ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
ver_ok = (py_ver.major == 3 and 8 <= py_ver.minor <= 11)
check(f"Python {ver_str}", ver_ok,
      "OK" if ver_ok else "Need 3.8–3.11 (Ryu compatibility)")

if py_ver.minor >= 12:
    print(f"  {WARN} Python 3.12+ may break Ryu 4.34 — use 3.10 or 3.11 if possible")


# =============================================================================
# 2. Core Python packages
# =============================================================================
section("Core Python Packages")

CORE_PACKAGES = [
    ("numpy",        "1.26", "Numerical computation"),
    ("pandas",       "2.2",  "Data analysis & time-series"),
    ("networkx",     "3.3",  "Fat-tree/Clos graph topology"),
    ("matplotlib",   "3.9",  "Plotting & visualisation"),
    ("seaborn",      "0.13", "Statistical plots"),
    ("scipy",        "1.13", "LP solver / statistics"),
    ("statsmodels",  "0.14", "ARIMA & EWMA prediction"),
    ("sklearn",      "1.5",  "scikit-learn — preprocessing"),
    ("tqdm",         "4.66", "Progress bars"),
    ("click",        "8.1",  "CLI interface"),
    ("yaml",         "6.0",  "Config file parsing (pyyaml)"),
    ("psutil",       "6.0",  "CPU/memory monitoring"),
]

for pkg_name, min_ver, purpose in CORE_PACKAGES:
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "unknown")
        check(f"{pkg_name} ({purpose})", True, f"v{ver}")
    except ImportError:
        check(f"{pkg_name} ({purpose})", False, f"MISSING — pip install {pkg_name}")


# =============================================================================
# 3. SDN / Controller packages
# =============================================================================
section("SDN / Controller Packages")

try:
    import ryu
    ryu_ver = getattr(ryu, "__version__", "unknown")
    check("ryu (SDN controller)", True, f"v{ryu_ver}")

    # Check OF 1.3 protocol support
    from ryu.ofproto import ofproto_v1_3
    check("ryu.ofproto.ofproto_v1_3 (OpenFlow 1.3)", True)

    # Check REST API app
    from ryu.app import ofctl_rest
    check("ryu.app.ofctl_rest (REST API)", True)

    # Check topology discovery
    from ryu.topology import api as topo_api
    check("ryu.topology (topology discovery)", True)

except ImportError as e:
    check(f"ryu", False, f"MISSING — {e}")

try:
    import mininet
    mn_ver = getattr(mininet, "VERSION", "unknown")
    check("mininet (network emulator)", True, f"v{mn_ver}")

    from mininet.topo import Topo
    check("mininet.topo (topology API)", True)

    from mininet.net import Mininet
    check("mininet.net (Mininet runtime)", True)

    from mininet.node import OVSSwitch, RemoteController
    check("mininet.node (OVS + RemoteController)", True)

    from mininet.link import TCLink
    check("mininet.link.TCLink (bandwidth limits)", True)

except ImportError as e:
    check("mininet", False, f"Not installed system-wide — run install_mininet.sh ({e})")


# =============================================================================
# 4. Optimization / MILP solvers
# =============================================================================
section("Optimization / MILP Solvers")

# Gurobi
try:
    import gurobipy as gp
    gp_ver = getattr(gp, "gurobi", None)
    ver_str = str(gp_ver.version() if gp_ver else "unknown")
    check("gurobipy (Gurobi Python API)", True, f"v{ver_str}")

    # Test license by creating a small model
    try:
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addVar(name="x")
        m.setObjective(x, gp.GRB.MINIMIZE)
        m.addConstr(x >= 1)
        m.optimize()
        check("Gurobi license & solver", True, "Model solved successfully")
    except gp.GurobiError as e:
        check("Gurobi license & solver", False, f"License error: {e}")

except ImportError:
    check("gurobipy (Gurobi Python API)", False,
          "MISSING — see setup/gurobi_setup.md for license")

# PuLP (open-source fallback)
try:
    import pulp
    check("pulp (CBC fallback solver)", True, f"v{pulp.__version__}")
    # Quick solve test
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    prob += x
    prob += x >= 1
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    check("PuLP/CBC solver test", pulp.LpStatus[status] == "Optimal")
except ImportError:
    check("pulp (CBC fallback)", False, "pip install pulp")

# scipy linprog (lightweight LP)
try:
    from scipy.optimize import linprog
    res = linprog(c=[1], A_ub=[[-1]], b_ub=[-1], bounds=[(0, None)])
    check("scipy.optimize.linprog (LP relaxation)", res.success)
except Exception as e:
    check("scipy.optimize.linprog", False, str(e))


# =============================================================================
# 5. System tools (CLI)
# =============================================================================
section("System Tools (CLI)")

TOOLS = [
    ("mn",        "Mininet CLI"),
    ("ryu-manager","Ryu controller launcher"),
    ("ovs-vsctl", "Open vSwitch control"),
    ("ovs-ofctl", "OpenFlow rule management"),
    ("iperf3",    "Network throughput testing"),
    ("tcpdump",   "Packet capture"),
    ("ip",        "iproute2 network configuration"),
]

for tool, desc in TOOLS:
    found = shutil.which(tool) is not None
    path  = shutil.which(tool) or "not found"
    check(f"{tool} ({desc})", found, path if found else "MISSING")

# OVS OpenFlow 1.3 protocol support
try:
    result = subprocess.run(
        ["ovs-ofctl", "--version"],
        capture_output=True, text=True, timeout=5
    )
    ovs_ok = result.returncode == 0
    ver_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    check("OVS OpenFlow 1.3 support", ovs_ok, ver_line)
except (FileNotFoundError, subprocess.TimeoutExpired):
    check("OVS OpenFlow 1.3 support", False, "ovs-ofctl not available")


# =============================================================================
# 6. OVS daemon status
# =============================================================================
section("Services")

try:
    result = subprocess.run(
        ["systemctl", "is-active", "openvswitch-switch"],
        capture_output=True, text=True, timeout=5
    )
    active = result.stdout.strip() == "active"
    check("openvswitch-switch (OVS daemon)", active,
          "running" if active else "STOPPED — run: sudo systemctl start openvswitch-switch")
except FileNotFoundError:
    check("openvswitch-switch", False, "systemctl not found (not Linux?)")


# =============================================================================
# 7. Testing framework
# =============================================================================
section("Testing Framework")

try:
    import pytest
    check("pytest", True, f"v{pytest.__version__}")
except ImportError:
    check("pytest", False, "pip install pytest")

try:
    import pytest_timeout
    check("pytest-timeout", True)
except ImportError:
    check("pytest-timeout", False, "pip install pytest-timeout")


# =============================================================================
# Summary
# =============================================================================
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"\n{'━'*60}")
print(f"{BOLD}SUMMARY:{RESET}  {GREEN}{passed} passed{RESET}  /  {RED}{failed} failed{RESET}  /  {total} total")
print(f"{'━'*60}")

if failed > 0:
    print(f"\n{RED}Failed checks:{RESET}")
    for name, ok, note in results:
        if not ok:
            print(f"  {FAIL} {name}  →  {note}")
    print(f"\n{YELLOW}Run with --fix to attempt auto-repair of Python packages.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}{BOLD}All checks passed! Ready to run LAFS experiments.{RESET}")
    sys.exit(0)


# =============================================================================
# CLI argument parsing (must be after functions defined above)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAFS dependency verifier")
    parser.add_argument("--verbose", action="store_true", help="Show extra detail")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to install missing Python packages")
    args = parser.parse_args()

    if args.fix:
        import subprocess
        missing_pkgs = [
            note.split("pip install ")[-1]
            for _, ok, note in results
            if not ok and "pip install" in note
        ]
        if missing_pkgs:
            print(f"\n{YELLOW}Attempting to install: {', '.join(missing_pkgs)}{RESET}")
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_pkgs)
        else:
            print(f"\n{GREEN}No pip-installable packages to fix.{RESET}")
