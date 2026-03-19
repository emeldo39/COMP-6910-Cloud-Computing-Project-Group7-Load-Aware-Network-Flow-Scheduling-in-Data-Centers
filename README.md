# LAFS — Load-Aware Flow Scheduler
**COMP 6910 Services Computing, Semantic Web and Cloud Computing — Winter 2026**
**Group 7**

| Member | Student ID |
|---|---|
| Victor Chisom Muojeke | 202480408 |
| Chiemerie Cletus Obijaiku | 202492457 |
| Olaleye Adeniyi AKINULI | 202488212 |

---

## Overview

LAFS is a hybrid centralized-distributed flow scheduler for data center networks that combines:
- **Proactive load prediction** (EWMA + ARIMA) to anticipate congestion 1–10s ahead
- **Multi-objective MILP optimization** minimizing average FCT (mice), P99 FCT (elephants), and max-min fairness violations across tenants
- **Distributed OpenFlow/P4 forwarding** via pre-computed k-shortest paths on edge switches

Evaluated on a Mininet k=8 Fat-tree (128 hosts) against baselines: ECMP, CONGA, Hedera, Offline-Optimal MILP.

---

## Project Structure

```
LAFS/
├── setup/
│   ├── install_mininet.sh      # Full Mininet 2.3.0 + OVS install (Ubuntu 20/22.04)
│   ├── setup_env.sh            # Python venv creation and pip install
│   ├── verify_deps.py          # Checks all dependencies and services
│   └── gurobi_setup.md         # Step-by-step Gurobi academic license guide
│
├── src/
│   ├── topology/               # Fat-tree (k=8) and 2-tier Clos topology builders
│   ├── controller/             # Ryu-based LAFS controller + flow manager
│   ├── prediction/             # EWMA + ARIMA traffic load predictor
│   ├── optimizer/              # MILP solver (Gurobi / PuLP fallback)
│   ├── baselines/              # ECMP, CONGA, Hedera, Optimal-MILP
│   ├── workload/               # Trace loaders + synthetic workload generators
│   └── metrics/                # FCT, goodput, Jain fairness, link utilization
│
├── experiments/
│   ├── run_experiments.py      # Main experiment runner (vary load, tenants, windows)
│   ├── ablation_study.py       # LAFS vs. no-prediction vs. no-fairness
│   └── scalability_test.py     # Controller CPU/memory under increasing flow count
│
├── tests/
│   ├── test_mininet.py         # Mininet connectivity + Fat-tree construction
│   ├── test_ryu.py             # Ryu controller startup + OpenFlow 1.3 handshake
│   ├── test_gurobi.py          # Gurobi license + MILP solve test
│   └── test_integration.py    # End-to-end: topology → controller → flow → metrics
│
├── data/
│   ├── raw/                    # Raw Facebook/Google trace downloads
│   ├── processed/              # Cleaned, normalised trace CSVs
│   └── traces/                 # Synthetic AllReduce + RPC workloads
│
├── results/
│   ├── figures/                # PNG/PDF plots (FCT CDFs, fairness charts)
│   └── metrics/                # CSV/JSON metric dumps per experiment
│
├── logbook/
│   └── logbook.md              # Weekly progress log (required deliverable)
│
├── report/                     # LaTeX source for 10-page LNCS final report
│
├── requirements.txt            # Python dependencies (exact versions)
└── .gitignore
```

---

## Quick Start

### Prerequisites
- Ubuntu 20.04 or 22.04 LTS
- Python 3.10 or 3.11
- At least 8 GB RAM (128-host Fat-tree is memory-intensive)
- Gurobi academic license (see `setup/gurobi_setup.md`) — or PuLP as fallback

### 1. Install Mininet + OVS

```bash
sudo chmod +x setup/install_mininet.sh
sudo ./setup/install_mininet.sh
```

### 2. Set up Python environment

```bash
# From project root
source setup/setup_env.sh
# OR
bash setup/setup_env.sh && source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Activate Gurobi license

```bash
grbgetkey <YOUR_KEY>
# Then see setup/gurobi_setup.md for full instructions
```

### 4. Verify all dependencies

```bash
python setup/verify_deps.py
```

### 5. Run verification tests

```bash
# Individual checks
python tests/test_mininet.py
python tests/test_ryu.py
python tests/test_gurobi.py

# Full integration test (requires sudo for Mininet)
sudo python tests/test_integration.py

# Or via pytest
pytest tests/ -v --timeout=120
```

### 6. Quick Mininet smoke test

```bash
# k=8 Fat-tree, OpenFlow 1.3, remote controller
sudo mn --topo=ftree,8 \
        --switch=ovsk,protocols=OpenFlow13 \
        --controller=remote,ip=127.0.0.1,port=6633 \
        --test=pingall
```

---

## Development Workflow

### Running a baseline experiment (ECMP)

```bash
# Terminal 1: Start Ryu controller with ECMP app
ryu-manager src/baselines/ecmp.py --ofp-tcp-listen-port 6633

# Terminal 2: Launch Mininet
sudo python src/topology/fat_tree.py --k 8

# Terminal 3: Run workload and collect metrics
python experiments/run_experiments.py --scheduler ecmp --load 0.6 --tenants 8
```

### Running LAFS

```bash
# Terminal 1: Start LAFS controller
ryu-manager src/controller/lafs_controller.py --ofp-tcp-listen-port 6633

# Terminal 2: Launch Mininet
sudo python src/topology/fat_tree.py --k 8

# Terminal 3: Run experiments
python experiments/run_experiments.py --scheduler lafs --load 0.6 --tenants 8
```

---

## Milestones & Timeline

| Phase | Tasks | Deadline | Status |
|---|---|---|---|
| Wk 1–2 | Literature review, proposal, Mininet setup | Feb 10 | ✅ Done |
| Wk 3–4 | ECMP/CONGA baselines, Fat-tree, traces | Feb 24 | ✅ Done |
| Wk 5–6 | Prediction module, controller integration | Mar 10 | ✅ Done |
| **Wk 7–8** | **MILP optimizer, flow placement, debug** | **Mar 21** | 🔄 Active |
| Wk 9–10 | Full experiments, ablation, presentation | Apr 4 | ⏳ |
| Wk 11 | LNCS report, code docs, logbook | Apr 10 | ⏳ |

---

## Key References

1. Alizadeh et al., "CONGA: Distributed Congestion-Aware Load Balancing," SIGCOMM 2014
2. Qian et al., "Alibaba HPN: A Data Center Network for LLM Training," SIGCOMM 2024
3. Al-Fares et al., "A Scalable, Commodity Data Center Network Architecture," SIGCOMM 2008

---

## Final Deliverables (Due Apr 10, 2026)

- [ ] 10-page LNCS-format report (`report/`)
- [ ] Complete source code with README
- [ ] Raw + processed experimental data (`data/`)
- [ ] Results figures and metrics (`results/`)
- [ ] Logbook (`logbook/logbook.md`)
- [ ] Zip of entire project directory → D2L submission
