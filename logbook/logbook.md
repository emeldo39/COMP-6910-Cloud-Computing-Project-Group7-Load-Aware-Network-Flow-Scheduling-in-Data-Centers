# LAFS Project Logbook
**COMP 6910 — Group 7**

| Member | Student ID |
|---|---|
| Victor Chisom Muojeke | 202480408 |
| Chiemerie Cletus Obijaiku | 202492457 |
| Olaleye Adeniyi AKINULI | 202488212 |

---

## Entry Format
Each entry records: date, author, tasks completed, decisions made, blockers.

---

## Week 1–2 (Feb 10): Literature Review & Setup

### [2026-02-03] Team
- Reviewed CONGA, Hedera, pFabric, B4/SWAN papers.
- Identified novelty gap: no system combines proactive prediction + multi-objective FCT+fairness.
- Agreed on LAFS architecture: centralized controller + distributed forwarding.
- Selected Mininet (k=8 Fat-tree) + Ryu + Gurobi as implementation stack.
- **Decision:** Use EWMA+ARIMA over LSTM — LSTM latency exceeds 100ms budget.

### [2026-03-19] Team — Phase 1: Development Environment Setup

**Completed:**
- Initialised GitHub repository: `emeldo39/COMP-6910-Cloud-Computing-Project-Group7-Load-Aware-Network-Flow-Scheduling-in-Data-Centers`
- Created full project directory structure (`src/`, `tests/`, `setup/`, `experiments/`, `data/`, `results/`, `logbook/`, `report/`)
- `setup/install_mininet.sh` — automated Mininet 2.3.0 + OVS + OpenFlow 1.3 install for Ubuntu 20.04/22.04, with kernel performance tuning for 128-host topology
- `setup/setup_env.sh` — Python virtual environment creation script
- `setup/verify_deps.py` — checks 30+ dependencies (packages, system tools, OVS daemon, Gurobi license)
- `setup/gurobi_setup.md` — step-by-step Gurobi academic license activation guide
- Test suite: `tests/test_mininet.py`, `tests/test_ryu.py`, `tests/test_gurobi.py`, `tests/test_integration.py`
- `requirements.txt` (Ubuntu/Python 3.10–3.11) and `requirements-windows.txt` (Windows/Python 3.13)

**Issue encountered & resolved:**
- `ryu==4.34` fails on Python 3.13 with `AttributeError: get_script_args` — caused by setuptools API removal.
- **Fix:** Created separate `requirements-windows.txt` using `numpy>=2.0` and excluding Ryu/Mininet (Linux-only). All other packages installed successfully on Windows Python 3.13.

**Decisions:**
- Windows machine = code development, MILP testing, analysis. Ubuntu VM = Mininet experiments.
- MILP solver fallback chain: Gurobi → PuLP/CBC → scipy linprog, controlled via `LAFS_SOLVER` env var.
- Ubuntu must stay on **Python 3.10 or 3.11** for Ryu 4.34 compatibility.

**Open items:**
- Gurobi academic license key not yet activated — pending registration at gurobi.com with university email.
- Ubuntu VM not yet provisioned — Mininet tests blocked until then.

---

## Week 3–4 (Feb 24): Baseline Implementation

*(To be filled)*

---

## Week 5–6 (Mar 10): Prediction Module

*(To be filled)*

---

## Week 7–8 (Mar 21): MILP Optimizer

*(To be filled)*

---

## Week 9–10 (Apr 4): Full Experiments

*(To be filled)*

---

## Week 11 (Apr 10): Report & Packaging

*(To be filled)*
