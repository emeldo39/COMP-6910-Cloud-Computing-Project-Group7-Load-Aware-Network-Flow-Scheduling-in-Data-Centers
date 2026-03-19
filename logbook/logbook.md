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

### [2026-03-19] Team — Phase 2: Fat-Tree Topology Implementation & Verification

**Completed:**
- `src/topology/fattree.py` — dual-class design:
  - `FatTreeGraph`: pure NetworkX graph (Windows-safe, no Mininet dependency)
    - Builds k=4 (16 hosts, 20 switches) in 0.7 ms; k=8 (128 hosts, 80 switches) in 1.9 ms
    - All-pairs ECMP path computation: 8,128 pairs, ~235,000 paths, ~4.5 s for k=8
    - IP/MAC lookup dicts, port numbering, link capacity metadata
  - `FatTreeTopo`: Mininet `Topo` subclass with stub fallback for Windows
    - Delegates path computation to `FatTreeGraph`
    - `_link_opts()` converts Gbps → Mbps for TCLink
- `src/topology/network_builder.py` — high-level Mininet manager
  - `NetworkConfig` / `ControllerConfig` dataclasses
  - Context manager (`managed()`), pingall, ping_pair, iperf_pair, collect_link_stats
  - Forces OpenFlow 1.3 on all OVS bridges after start
- `src/topology/TOPOLOGY.md` — full architecture reference with ASCII diagrams, port numbering tables, IP/MAC schemes, usage examples, troubleshooting guide
- `tests/unit/test_topology.py` — **95 tests, all pass on Windows**
  - Naming helpers (15), k=4 graph (13), k=8 graph (15), paths (17), edges (12), topology properties (10), Mininet stub (8)
- `tests/integration/test_topology_integration.py` — Linux/root/Mininet integration tests (6 test classes)

**Topology verification results (Windows, k=4 and k=8):**

| Metric | k=4 | k=8 | Expected |
|---|---|---|---|
| Hosts | 16 | 128 | k³/4 ✓ |
| Switches | 20 | 80 | 5k²/4 ✓ |
| Links | 48 | 384 | — ✓ |
| Within-pod ECMP paths | 2 | 4 | k/2 ✓ |
| Cross-pod ECMP paths | 4 | 16 | (k/2)² ✓ |
| Graph diameter | 6 | 6 | 6 hops ✓ |
| Unique IPs | 16 | 128 | k³/4 ✓ |
| Unique MACs | 16 | 128 | k³/4 ✓ |
| Fully connected | Yes | Yes | ✓ |

**Bugs found and fixed:**

1. **test_gurobi.py — LP infeasibility**: `cap = [random.uniform(0.8, 1.2)]` with seed=42 produced infeasible LP (capacity < total demand). Fixed: `cap = [sum(load)/n_paths + 0.5 ...]`. Also moved `import os` to top and guarded `m.ObjVal` read with `if m.Status == GRB.OPTIMAL`.
2. **test_ryu.py — 21 FAIL on Windows**: `ModuleNotFoundError: ryu` caused 21 FAILs instead of SKIPs. Fixed: added `_RYU_AVAILABLE` flag and `_RYU_SKIP = unittest.skip(...)` decorator on all 4 test classes.
3. **test_mininet.py — collection ERROR**: `NameError: Topo not defined` on Windows at import time. Fixed: added stub `Topo` class when `_MININET_AVAILABLE = False`; replaced `sys.exit(0)` with pytest.mark.skip.

**Final test suite result (Windows Python 3.13):**
```
105 passed, 34 skipped, 0 failed  (3.86 s)
```
- 95 topology unit tests: all pass
- 10 Gurobi/MILP tests: all pass
- 34 Linux-only tests (Ryu + Mininet): correctly skipped with informative messages

**Committed:** `5663a45` — "Fix test suite Windows compatibility"

**Open items (Ubuntu VM required):**
- Run `sudo python tests/integration/test_topology_integration.py --k 4` (k=4 Mininet test)
- Run `sudo python tests/integration/test_topology_integration.py --k 8` (full 128-host test)
- Activate Gurobi academic license (pending university email registration at gurobi.com)

---

## Week 3–4 (Feb 24): Baseline Implementation

### [2026-03-19] Team — Phase 2a: Scheduler Infrastructure & ECMP Baseline

**Completed:**
- `src/workload/flow.py` — `Flow` dataclass (5-tuple + metadata):
  - Fields: `flow_id`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `size_bytes`, `arrival_time`, `deadline`
  - Properties: `five_tuple`, `is_mice` (<100 KB), `is_elephant` (≥1 MB), `fct`, `ideal_fct`, `slowdown`, `meets_deadline`
  - `Flow.create()` classmethod, `Flow.new_id()` UUID helper, `__post_init__` validation
- `src/scheduler/base_scheduler.py` — `BaseScheduler` ABC + `SchedulerMetrics`:
  - Abstract API: `schedule_flow(flow)` → `Optional[List[str]]`; concrete `schedule_flows(flows)` handles metrics + path assignment
  - `SchedulerMetrics`: mice/medium/elephant counts, path_usage histogram, avg/p99 latency, Jain's index
  - `get_candidate_paths()` wraps `node_for_ip()` with `try/except KeyError` for safe IP lookup
- `src/scheduler/ecmp.py` — `ECMPScheduler`:
  - `ecmp_hash()`: CRC32 of 13-byte packed 5-tuple (`struct.pack("!IIHHB")`), returns uint32
  - Path selection: `hash % n_paths`; `_path_cache` keyed on `(src_node, dst_node)` pair
  - `path_balance_ratio()` (min/max load), `ecmp_stats()`, `hash_distribution()` helpers

**Bugs found and fixed:**
1. `FatTreeGraph.node_for_ip()` raises `KeyError` (not returns None) — wrapped all call sites in `try/except KeyError`.
2. Metrics only update via `schedule_flows()` — tests expecting metric updates were corrected to call the batch API.

---

### [2026-03-19] Team — Phase 2b: Hedera & CONGA Baselines

**Completed:**
- `src/scheduler/hedera.py` — `HederaScheduler` + `PathLoadTracker`:
  - Mice flows → internal `ECMPScheduler` (no load tracking)
  - Elephant flows → Global First Fit: `PathLoadTracker.least_loaded_path()` assigns to least-utilised candidate
  - `reschedule_elephants()`: release all elephant flows, sort by size descending, re-place with GFF
  - `PathLoadTracker`: per-path byte accounting, utilisation = `bytes×8 / link_capacity_bits`
- `src/scheduler/conga.py` — `CONGAScheduler` + `CongestionTable` + `FlowletTable`:
  - `CongestionTable`: DRE via EWMA (`alpha`) + exponential time-decay (`decay_rate`); `best_path_idx()` picks min-DRE path
  - `FlowletTable`: flowlet gap detection (500 µs default); continuing flowlets keep same path
  - `inject_congestion()` for test harness control; `evict_expired_flowlets()` for memory hygiene
- `src/scheduler/__init__.py` — exports all six scheduler symbols

**Test results:**
- `tests/unit/test_scheduler.py` — **71 tests**, all pass (Flow dataclass ×22, ECMP hash ×12, BaseScheduler ×8, ECMP unit ×18, integration ×10 — 100 flows)
- `tests/unit/test_hedera.py` — **46 tests**, all pass (PathLoadTracker ×14, HederaScheduler ×20, integration ×12)
- `tests/unit/test_conga.py` — **51 tests**, all pass (CongestionTable ×14, FlowletTable ×10, CONGAScheduler ×18, integration ×12)

**Full suite after Phase 2:**
```
279 passed, 75 skipped, 0 failed  (~10 s)
```

**Committed:** baseline scheduler infrastructure + ECMP + Hedera + CONGA

---

### [2026-03-19] Team — Phase 3: Workload Generation System

**Completed:**
- `src/workload/facebook_websearch.py` — `FacebookWebSearchGenerator`:
  - Empirical CDF from Benson et al. IMC 2010 (6 piecewise-uniform segments)
  - 20% <1 KB · 55% <10 KB · 90% <100 KB (mice) · 98% <10 MB · 100% ≤100 MB
  - Poisson inter-arrival; TCP port pool {80, 443, 8080, 8443}; multi-tenant round-robin host partition
  - `cross_tenant_fraction` (5%) and `aggregator_fraction` (25%) for realistic request routing
- `src/workload/allreduce.py` — `AllReduceGenerator`:
  - Ring AllReduce (NCCL-style): n_workers flows/iteration, shard_bytes = gradient / n_workers, `dst_port=29500`
  - PS mode: 2×(n_workers−1) flows (upload at t, download at t+1 ms)
  - Pipeline parallelism: 2×(stages−1) activation flows per iteration boundary
  - ±10% Gaussian jitter on `iteration_gap_s`; simulation clock advances by gap + estimated AllReduce duration
  - Model presets: `resnet50` 25 MB · `bert_base` 110 MB · `gpt2` 548 MB · `llama_7b` 14 GB
- `src/workload/microservice.py` — `MicroserviceRPCGenerator` + `ServiceGraph`:
  - `ServiceGraph` factory methods: `linear_chain()`, `fan_out()`, `mixed_dag()`
  - Kahn's topological ordering; `node_ready` timing propagation for chain RTT accumulation
  - 2 flows per edge (request tiny + response small-medium); optional DB data-payload flows
  - Rack-aware placement groups services by edge-switch; random mode shuffles and round-robins
- `src/workload/runner.py` — `WorkloadRunner` + `WorkloadConfig` + `WorkloadStats`:
  - `WorkloadConfig`: proportional weight allocation across generators; `mixed` expands to all three
  - Per-generator sub-seed: `seed + hash(wt) % 100_000` for reproducible independence
  - Reservoir sub-sampling preserves time order when merged count exceeds `n_flows`
  - `WorkloadStats`: mice/medium/elephant counts, percentiles (P50/P90/P99), arrival rate, Jain's fairness index over tenant flow counts: $(Σx)^2 / (n·Σx^2)$
- `src/workload/__init__.py` — updated exports (all 13 symbols)

**Workload validation:**
| Generator | Key property | Verified |
|---|---|---|
| Facebook web search | Mice fraction ≥ 85% | ✓ (≈ 90%) |
| Facebook web search | Max size ≤ 100 MB | ✓ |
| AllReduce (ring) | Flows = n_workers × n_iterations | ✓ |
| AllReduce (ring) | Ring adjacency | ✓ |
| AllReduce (PS) | Flows = 2×(n_workers−1)×n_iterations | ✓ |
| Microservice | Topological order respected | ✓ |
| Microservice | Sorted by arrival_time | ✓ |
| Runner (mixed) | All three generators present | ✓ |
| Runner stats | Jain's index ∈ [0, 1] | ✓ |

**Test results:**
- `tests/unit/test_workload.py` — **63 tests**, all pass
  - FacebookWebSearch ×18, AllReduce ×16, MicroserviceRPC ×15, WorkloadRunner ×14

**Full suite after Phase 3:**
```
342 passed, 75 skipped, 0 failed  (~13 s)
```

**Open items:**
- Ubuntu VM: run Mininet integration tests (`--k 4`, `--k 8`)
- Gurobi academic license activation (pending university email at gurobi.com)
- Phase 4 next: EWMA + ARIMA prediction module (`src/prediction/`)

---

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
