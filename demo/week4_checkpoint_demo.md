# LAFS -- Week 4 Demo Script
**COMP 6910 -- Group 7**
**Checkpoint date: week of Feb 24, 2026**

---

## Overview

This document tells you exactly what to show, what to say, and what
answers to give when the professor asks questions.

The demo has five segments (target 10--15 minutes total):

| # | Segment | Duration |
|---|---------|----------|
| 1 | What we built and why | 2 min |
| 2 | Live: full pipeline run | 3 min |
| 3 | Walk through the numbers | 3 min |
| 4 | Code tour (architecture) | 3 min |
| 5 | Q&A answers | prepared |

---

## Pre-Demo Checklist (do before the professor arrives)

```
[ ] Open a terminal in the project root
[ ] Run:  python -m pytest --tb=short -q
    Expected: 371 passed, 75 skipped, 0 failed
[ ] Run:  python tests/integration/test_e2e_week4.py
    Expected: "OVERALL: ALL CHECKS PASSED -- Ready for demo"
[ ] Open VSCode or PyCharm with the project
[ ] Have these files ready to show:
      src/topology/fattree.py
      src/scheduler/ecmp.py
      src/workload/facebook_websearch.py
      tests/integration/test_e2e_week4.py
      results/week4_e2e_summary.json
[ ] Have the logbook open: logbook/logbook.md
```

---

## Segment 1 -- What We Built and Why (2 min)

**Say:**

> "We're building LAFS -- Load-Aware Flow Scheduling -- for data-center
> networks. The core problem is that dumb per-flow ECMP load balancing
> causes head-of-line blocking: a few large elephant flows monopolize
> paths and inflate the completion times of latency-sensitive mice flows.
>
> By week 4 we have three components fully working:
> a k=8 fat-tree topology with 128 hosts and 80 switches,
> an ECMP baseline scheduler using CRC32 5-tuple hashing,
> and a Facebook web-search workload generator based on empirical
> flow-size data from Benson et al. IMC 2010.
>
> Today we'll show the full pipeline running end-to-end and walk through
> the measured FCT distribution."

**Show:** The architecture diagram in the logbook (logbook/logbook.md,
Phase 3 entry) or draw on whiteboard:

```
[WorkloadGenerator] --> 1000 flows --> [ECMPScheduler] --> paths
         |                                   |
  Facebook CDF                        CRC32 5-tuple hash
  (90% mice)                          path = hash % n_paths
         |                                   |
         +-------> [PathFIFOSimulator] <------+
                         |
                   FCT per flow
                         |
                   [MetricsReport]
                   avg/P50/P95/P99
```

---

## Segment 2 -- Live Pipeline Run (3 min)

**Type in the terminal:**

```bash
python tests/integration/test_e2e_week4.py
```

**While it runs, narrate:**

> "The first line builds the k=8 fat-tree -- that takes about 10 ms.
> Next we generate 1000 Facebook-style flows using inverse-transform
> sampling over the empirical CDF. You can see it prints '89% mice'
> which matches the real data center distribution.
> Then ECMP hashes each flow's 5-tuple and assigns it to one of up to
> 16 equal-cost paths -- that's the 616 ms step because we're also
> running the full path computations with NetworkX."

**Point at the output as it appears:**

- "128 hosts, 80 switches -- that matches the k^3/4 and 5k^2/4 formulas
  for a k=8 fat-tree."
- "1000 flows scheduled, 0 failed -- 100% success rate."
- "956 distinct paths used -- ECMP is spreading flows well."
- "P50 FCT = 0.09 ms for mice flows, which is close to ideal
  (a 10 KB packet at 1 Gbps takes 0.08 ms)."
- "ALL CHECKS PASSED -- we're good for demo."

---

## Segment 3 -- Walk Through the Numbers (3 min)

Open `results/week4_e2e_summary.json` and talk through each section.

### Topology numbers
```
k=8: 128 hosts, 80 switches, 384 links
Max ECMP paths: 16 (cross-pod), 4 (within-pod), 1 (same-edge)
```
**Say:** "These are exact -- k^3/4=128, 5k^2/4=80. Any pair of hosts
has between 1 and 16 equal-cost paths."

### Workload numbers
```
Mice (<100 KB):   889 flows  (88.9%)
Medium:            93 flows
Elephant (>10 MB): 18 flows  (1.8%)
Mean size: 1292 KB    -- pulled up by a few large flows
P90 size:  183 KB
P99 size:   66 MB
```
**Say:** "The heavy-tail is real -- the mean is 13x the median because
elephants dominate bytes even when they're only 2% of flows. This is
exactly the distribution that makes ECMP a bad scheduler: those 18
elephant flows will saturate whichever paths ECMP assigns them to."

### Scheduling numbers
```
956 unique paths used out of the 1000 flows
Balance ratio: 0.57   (min_load / max_load across path indices)
Avg scheduling latency: 607 us
P99 scheduling latency: 3.3 ms
```
**Say:** "Balance ratio of 0.57 means the least-used path got 57% of
the traffic of the busiest path. For pure random hashing you'd expect
0.8+ with 1000 flows. The variance comes from flow-size imbalance --
some paths carry one elephant and look very busy in bytes even if they
have few flows. This is the motivation for Hedera (which we've also
implemented) and ultimately for LAFS."

### FCT numbers
```
Mice    P50=0.08 ms   P95=0.73 ms   P99=0.82 ms
Medium  P50=4.57 ms   P95=7.57 ms   P99=7.88 ms
Elephant P50=71 ms    P95=741 ms    P99=771 ms
```
**Say:** "The gap between mice P99 (0.82 ms) and elephant P50 (71 ms)
is the core problem we're solving. In a perfect scheduler, the elephant
flows would not block the mice paths. LAFS's job is to move that
elephant P50 down while keeping mice P99 near its ideal value."

**Key insight for the professor:**
> "Ideal FCT for a 10 KB mice flow at 1 Gbps is 0.08 ms. We're seeing
> P50=0.08 ms and P99=0.82 ms -- slowdown of ~10x at the 99th percentile.
> That tail comes from flow bursting on shared paths. LAFS will address
> this with proactive EWMA+ARIMA prediction to detect congestion before
> it builds up."

---

## Segment 4 -- Code Tour (3 min)

### Show src/topology/fattree.py (30 seconds)
Point at the `FatTreeGraph` class.

> "Pure-Python / NetworkX implementation. No Mininet dependency -- works
> on Windows. Builds in 10 ms, computes all ECMP paths in under 5 seconds.
> We'll run the actual Mininet experiments on our Ubuntu VM once we have it."

### Show src/scheduler/ecmp.py (60 seconds)
Point at `ecmp_hash()` and `schedule_flow()`.

> "13-byte packed 5-tuple, CRC32, mask to uint32. Path index = hash mod
> number of paths. O(1) after the first path-cache lookup per host pair.
> This is the baseline every other scheduler in our paper beats."

### Show src/workload/facebook_websearch.py (30 seconds)
Point at `_FB_WEBSEARCH_CDF` and `_sample_flow_size()`.

> "Six-segment piecewise-uniform CDF from Benson et al. 2010 Table 1.
> Inverse-transform sampling -- uniform random [0,1] maps exactly to
> the right size segment. Poisson inter-arrival gives us a realistic
> open-loop traffic model."

### Show tests/integration/test_e2e_week4.py (60 seconds)
Point at `TestWeek4EndToEnd.setUpClass` and `simulate_fct`.

> "29-test integration suite. setUpClass runs once -- topology, workload,
> scheduling, simulation -- and each test checks one property.
> The FCT simulator is a work-conserving FIFO per path: flows on the same
> path queue behind each other in arrival-time order. Simple but
> analytically sound for a single bottleneck."

---

## Segment 5 -- Anticipated Questions and Answers

### Q: "Is this a real network or a simulation?"

**A:** "The scheduler and workload generator are real production-grade
components -- the code paths that run in the actual LAFS controller.
The FCT numbers here come from our work-conserving path simulator, which
is a standard analytical model for single-bottleneck queuing.

For real network measurements we'll use Mininet (k=8, 128 hosts,
Ryu controller, OpenFlow 1.3) running on our Ubuntu VM. We have those
integration tests ready (tests/integration/test_topology_integration.py)
and the install script (setup/install_mininet.sh). We're waiting on VM
provisioning -- those experiments will run in Weeks 9-10."

---

### Q: "Why CRC32 for ECMP? Why not MD5 or xxHash?"

**A:** "Three reasons. First, speed: CRC32 is hardware-accelerated on
every modern x86 CPU (the CRC32 instruction), making it sub-nanosecond.
Second, distribution quality: for IP/port tuples with random ephemeral
ports, CRC32 gives excellent bit dispersion -- our unit tests verify
chi-square uniformity over 1000 flows. Third, determinism: CRC32 gives
the same hash value on every platform and Python version -- we have a
regression test that pins the exact uint32 output for a known 5-tuple."

---

### Q: "Your P99 elephant FCT is 770 ms -- that's terrible. Is ECMP actually used in practice?"

**A:** "Yes, and this is the point. ECMP is the default in every commodity
data-center switch today precisely because it's stateless and requires no
controller involvement. Our 770 ms P99 happens because 18 elephant flows
(1.8% of flows but 86% of bytes) all hash to a small number of paths,
creating severe head-of-line blocking for the flows behind them.

That 770 ms is our baseline. Hedera (our next baseline, already
implemented) would bring it down by roughly 2-3x via Global First Fit.
LAFS targets a further 40-60% reduction by predicting congestion before
it happens and pre-routing elephants away from congested paths."

---

### Q: "How does Jain's fairness index of 0.997 fit in?"

**A:** "Jain's index here measures fairness of flows across the 4 tenants,
not across paths. 0.997 is near-perfect because the Facebook workload
generator distributes flows among tenants with a round-robin host
partition. The path load imbalance (balance ratio 0.57) is a different
metric -- it measures utilization variance across ECMP paths, not across
tenants. For the full LAFS evaluation we'll report both: per-tenant
fairness (Jain) and path utilization balance (min/max ratio)."

---

### Q: "You have 75 skipped tests -- what are those?"

**A:** "34 are Mininet integration tests -- they require Linux root and a
running Mininet environment. We skip them on Windows with a clear
informative message. The remaining 41 are Ryu controller unit tests --
Ryu 4.34 doesn't install on Python 3.13 (setuptools API change), so we
skip those on Windows too. On our Ubuntu VM with Python 3.11, all 75 will
run and we expect them to pass."

---

### Q: "When will you have results from the actual LAFS algorithm?"

**A:** "Phase 4 (this week and next) is the EWMA+ARIMA prediction module.
Phase 5 (Weeks 7-8) is the MILP optimizer that combines predictions with
multi-objective FCT+fairness. Full experiments in Weeks 9-10 with all
four schedulers (ECMP, Hedera, CONGA, LAFS) across load levels 30-90%
and tenant counts 4-16. The proposal's hypothesis is that LAFS achieves
40-60% lower P99 FCT than ECMP with Jain's index > 0.95."

---

## Key Numbers to Memorize

| Metric | Value | What it means |
|--------|-------|---------------|
| k=8 hosts | 128 | k^3/4 |
| k=8 switches | 80 | 5k^2/4 |
| Max ECMP paths | 16 | (k/2)^2 cross-pod |
| Flows generated | 1,000 | 10 s trace, 50% load |
| Mice fraction | 88.9% | target ~90%, Benson 2010 |
| Scheduling success | 100% | 0 failed |
| Unique paths used | 956 | near 1:1 with flows |
| Balance ratio | 0.57 | 1.0 = perfect |
| Mice P99 FCT | 0.82 ms | ~10x slowdown from ideal |
| Elephant P50 FCT | 71 ms | the problem LAFS solves |
| Test suite | 371 pass / 75 skip | 0 failed |

---

## Emergency Backup Plan

If the live terminal run fails for any reason:

1. Show the pre-saved JSON: `results/week4_e2e_summary.json`
2. Run the unit tests instead: `python -m pytest tests/unit/ -v --tb=short`
3. Walk through the code statically

The unit tests are faster (< 5 s) and more granular, so they actually
make a better demo if time is short.

---

*Generated: 2026-03-19 | LAFS COMP 6910 Group 7*
