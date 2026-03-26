# LAFS — Load-Aware Flow Scheduler
**COMP-6910: Services, Semantic, and Cloud Computing — Winter 2026**
**Memorial University of Newfoundland — Group 7**

| Member | Student ID |
|---|---|
| Victor Chisom Muojeke | 202480408 |
| Chiemerie Cletus Obijiaku | 202492457 |
| Olaleye Adeniyi AKINULI | 202488212 |

---

## Overview

LAFS is a hybrid centralized-distributed flow scheduler for data center fat-tree networks. It combines three components:

1. **Load Prediction** — per-link EWMA + ARIMA forecaster producing `NetworkLoadForecast` (point estimate + 90% CI) 1–10 s ahead of the current scheduling window.
2. **MILP Flow Placement** — multi-objective integer program minimising maximum link utilisation with an optional mice-flow hop-count penalty; solved by PuLP/CBC (default) or Gurobi.
3. **Baseline Schedulers** — ECMP, Hedera (GFF), and CONGA for head-to-head comparison.

Evaluated on a simulated k=8 Fat-tree (128 hosts, 80 switches) against 1,000 Facebook web-search flows across four load levels (30–90%).

---

## What is Actually Built

| Module | Files | Tests | Status |
|---|---|---|---|
| Fat-tree topology (k=4..16) | `src/topology/` | 95 | Done |
| ECMP scheduler | `src/scheduler/ecmp.py` | 71 | Done |
| Hedera (GFF) scheduler | `src/scheduler/hedera.py` | 44 | Done |
| CONGA scheduler | `src/scheduler/conga.py` | 53 | Done |
| Workload generators (Facebook, AllReduce, Microservice) | `src/workload/` | 63 | Done |
| EWMA + ARIMA + Hybrid predictor | `src/prediction/` | 72 | Done |
| MILP optimizer (LAFSMILPSolver + LAFSScheduler) | `src/optimizer/` | 43 | Done |
| Week 4 E2E integration test | `tests/integration/` | 29 | Done |
| Comparison experiment runner | `experiments/run_comparison.py` | — | Done |
| **Total** | | **470 pass, 0 fail** | |

---

## Project Structure

```
LAFS/
├── src/
│   ├── topology/
│   │   ├── fattree.py              # FatTreeGraph (NetworkX); k-ary fat-tree
│   │   └── network_builder.py      # Mininet NetworkBuilder (Linux only)
│   ├── scheduler/
│   │   ├── base_scheduler.py       # Abstract BaseScheduler + SchedulerMetrics
│   │   ├── ecmp.py                 # ECMPScheduler (CRC32 5-tuple hash)
│   │   ├── hedera.py               # HederaScheduler (Global First Fit)
│   │   └── conga.py                # CONGAScheduler (DRE + flowlet detection)
│   ├── workload/
│   │   ├── flow.py                 # Flow dataclass with FCT / slowdown helpers
│   │   ├── facebook_websearch.py   # Benson IMC 2010 CDF, Poisson arrivals
│   │   ├── allreduce.py            # Ring AllReduce (ML training traffic)
│   │   └── microservice.py         # gRPC RPC chain generator
│   ├── metrics/
│   │   └── link_load.py            # LinkLoadSample / LinkLoadSeries / LinkLoadSampler
│   ├── prediction/
│   │   ├── ewma.py                 # EWMAPredictor (alpha auto-tuning)
│   │   ├── arima.py                # ARPredictor (numpy OLS) + ARIMAPredictor (statsmodels)
│   │   ├── hybrid.py               # HybridPredictor (inverse-error blending)
│   │   └── forecaster.py           # LoadForecaster -> NetworkLoadForecast
│   └── optimizer/
│       ├── milp_solver.py          # LAFSMILPSolver (PuLP/CBC or Gurobi)
│       └── lafs_scheduler.py       # LAFSScheduler(BaseScheduler) -- MILP batch API
│
├── experiments/
│   └── run_comparison.py           # ECMP vs Hedera vs CONGA vs LAFS load sweep
│
├── tests/
│   ├── unit/                       # 457 unit tests (topology, scheduler, workload,
│   │                               #   prediction, optimizer)
│   └── integration/
│       ├── test_e2e_week4.py       # 29-test full-pipeline integration test
│       └── test_topology_integration.py
│
├── results/
│   ├── figures/                    # PNG plots from run_comparison.py
│   └── metrics/                    # JSON + CSV experiment output
│
├── setup/
│   ├── setup_env.sh                # Python venv + pip install
│   ├── install_mininet.sh          # Mininet 2.3.0 + OVS (Ubuntu 20/22.04)
│   ├── verify_deps.py              # Dependency checker
│   └── gurobi_setup.md             # Gurobi academic license guide
│
├── report/                         # LaTeX LNCS final report
├── logbook/logbook.md              # Weekly progress log (required deliverable)
├── requirements.txt                # Pinned Python dependencies
└── requirements-windows.txt        # Windows-compatible subset (no Mininet/Ryu)
```

---

## Quick Start (Windows / macOS — no Mininet required)

All simulation components run on Windows and macOS. Mininet and Ryu are only needed for live SDN experiments on Ubuntu.

### 1. Set up Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt        # full
# OR on Windows (no Ryu):
pip install -r requirements-windows.txt
```

### 2. Run the unit test suite

```bash
pytest tests/unit/ -v
# Expected: 441 passed, 0 failed (~12 s)
```

### 3. Run the Week 4 end-to-end integration test

```bash
python tests/integration/test_e2e_week4.py
# Prints ASCII metrics report + writes results/week4_e2e_summary.json
```

### 4. Run the scheduler comparison experiment

```bash
# Default: k=8, 1000 flows, 4 load levels (30/50/70/90%), with plots
python experiments/run_comparison.py

# Faster smoke test:
python experiments/run_comparison.py --k 4 --n-flows 200 --loads 0.5 --no-plots

# Full experiment options:
python experiments/run_comparison.py --help
```

Output files:
- `results/metrics/comparison_<timestamp>.json` — full nested results
- `results/metrics/comparison_<timestamp>.csv` — flat rows for pandas/Excel
- `results/figures/fct_p99_vs_load.png` — FCT comparison chart
- `results/figures/link_util_vs_load.png` — link utilisation chart
- `results/figures/jains_fairness_vs_load.png` — fairness chart
- `results/figures/ablation_50pct.png` — ablation bar chart
- `results/figures/fct_cdf_50pct.png` — CDF at 50% load

### 5. Use LAFS programmatically

```python
from src.topology.fattree import FatTreeGraph
from src.workload.facebook_websearch import FacebookWebSearchGenerator, FacebookWebSearchConfig
from src.optimizer.lafs_scheduler import LAFSScheduler
from src.optimizer.milp_solver import MILPConfig

# Build topology
topo = FatTreeGraph(k=8)

# Generate flows
gen = FacebookWebSearchGenerator(topo, FacebookWebSearchConfig(n_flows=500, load_fraction=0.7))
flows = gen.generate()

# Schedule with LAFS MILP
sched = LAFSScheduler(topo, milp_config=MILPConfig(solver="pulp", time_limit_s=10.0))
result = sched.schedule_flows_milp(flows)

print(result.summary())
# [PuLP/CBC] status=Optimal flows=500 vars=8001 max_util=0.18 time=1832ms
```

### 6. Switch MILP solver to Gurobi (optional)

```bash
# Via environment variable (no code change needed):
export LAFS_SOLVER=gurobi
python experiments/run_comparison.py

# Or in Python:
from src.optimizer.milp_solver import MILPConfig
cfg = MILPConfig(solver="gurobi")
```

> See `setup/gurobi_setup.md` for academic license activation.

---

## Quick Start (Ubuntu — with Mininet)

```bash
# 1. Install Mininet + OVS
sudo chmod +x setup/install_mininet.sh && sudo ./setup/install_mininet.sh

# 2. Set up Python environment
source setup/setup_env.sh

# 3. Verify dependencies
python setup/verify_deps.py

# 4. Run topology integration test
sudo python tests/integration/test_topology_integration.py --k 4
sudo python tests/integration/test_topology_integration.py --k 8

# 5. Quick Mininet smoke test
sudo mn --topo=ftree,8 --switch=ovsk,protocols=OpenFlow13 \
        --controller=remote,ip=127.0.0.1,port=6633 --test=pingall
```

---

## Experimental Results

Experiment: k=8 Fat-tree, 1,000 Facebook web-search flows, seed=42, hot-spot traffic (5% aggregators).
Simulation model: M/G/1 queuing on fabric links; `FCT = tx_time / (1 - rho_fabric)`.

| Scheduler | P50 FCT | P99 FCT | Fabric Max Util | Jain's Fairness | Solve Time |
|---|---|---|---|---|---|
| ECMP | 0.101 ms | 275.6 ms | 1.6% | 0.0175 | ~1 ms |
| Hedera | 0.100 ms | 289.5 ms | **6.3%** | 0.0175 | ~1 ms |
| CONGA | 0.101 ms | **273.8 ms** | 2.0% | 0.0175 | ~1 ms |
| **LAFS** | 0.101 ms | 275.4 ms | **1.6%** | 0.0175 | **~3 s** |

Key findings:
- At light-to-medium load, all schedulers achieve similar FCT — consistent with published results (CONGA paper shows benefits above 40% actual link utilisation).
- Hedera's GFF produces 4× higher fabric link utilisation than ECMP/LAFS due to path concentration without global diversity.
- LAFS achieves ECMP-equivalent FCT with **MILP-guaranteed optimal link placement** (z* = 0.241, Optimal status in all 4 load levels).
- LAFS-pred (with LoadForecaster) achieves an additional 0.7% P99 FCT reduction over LAFS-no-pred.

---

## MILP Solver Performance (k=8, 1,000 flows)

| Metric | Value |
|---|---|
| Binary decision variables | ~16,000 (1,000 flows × 16 ECMP paths) |
| Continuous variable (z) | 1 |
| Assignment constraints | 1,000 |
| Utilisation constraints | ~384 |
| Solver | PuLP + CBC (open-source) |
| Solve time (P50) | 3.0 s |
| Solve time (P99) | 3.7 s |
| MILP status | Optimal (100% of runs) |
| MIP gap tolerance | 1% |

> Gurobi (free academic license) solves the same problem 10–100× faster. See `setup/gurobi_setup.md`.

---

## Milestones

| Phase | Deliverable | Deadline | Status |
|---|---|---|---|
| Wk 1–2 | Literature review, proposal, environment setup | Feb 10 | Done |
| Wk 3–4 | ECMP/Hedera/CONGA baselines, Fat-tree topology, workload generators | Feb 24 | Done |
| Wk 5–6 | Load prediction module (EWMA + ARIMA + Hybrid + LoadForecaster) | Mar 10 | Done |
| Wk 7–8 | MILP optimizer (LAFSMILPSolver + LAFSScheduler), 486 tests | Mar 21 | Done |
| Wk 9–10 | Scheduler comparison experiment, plots, results | Apr 4 | Done |
| **Wk 11** | **10-page LNCS report, README, submission packaging** | **Apr 10** | Report written (compile PDF + zip remaining) |

---

## Dependencies

Core (all platforms):
```
numpy==1.26.4          # numerical arrays
networkx==3.3           # fat-tree graph
pulp==2.8.0             # MILP solver (CBC backend, no license required)
statsmodels==0.14.2     # ARIMA fitting
scipy==1.13.1           # LP utilities
matplotlib==3.9.0       # experiment plots
pandas==2.2.2           # results CSV handling
```

Linux-only (Mininet experiments):
```
ryu==4.34               # SDN controller (OpenFlow 1.3)
gurobipy==11.0.3        # Gurobi Python API (requires separate license)
```

---

## Key References

1. Al-Fares et al., "A Scalable, Commodity Data Center Network Architecture," SIGCOMM 2008
2. Al-Fares et al., "Hedera: Dynamic Flow Scheduling for Data Center Networks," NSDI 2010
3. Alizadeh et al., "CONGA: Distributed Congestion-Aware Load Balancing," SIGCOMM 2014
4. Benson et al., "Network Traffic Characteristics of Data Centers in the Wild," IMC 2010
5. Qian et al., "Alibaba HPN: A Data Center Network for Large-Scale LLM Training," SIGCOMM 2024

---

## Final Deliverables (Due Apr 10, 2026)

- [x] Complete source code with unit + integration tests (486 pass)
- [x] Experiment runner and results (`results/metrics/`, `results/figures/`)
- [x] Logbook (`logbook/logbook.md`)
- [x] 10-page LNCS report (`report/main.tex`) — compile with `pdflatex` + `llncs.cls` to produce `report/lafs_report.pdf`
- [ ] Zip of entire project directory → D2L submission
