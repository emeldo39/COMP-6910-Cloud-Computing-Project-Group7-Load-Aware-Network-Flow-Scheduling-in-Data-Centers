# Fat-Tree Topology — Architecture & Usage Guide

**LAFS Project, COMP-6910 — Group 7**

---

## 1. Architecture Overview

### What is a Fat-Tree?

A k-ary Fat-Tree is a specific instance of a Clos network that provides **full bisection bandwidth** — every host can simultaneously send at line rate without any oversubscription.  It was popularised for data centres by Al-Fares et al. (SIGCOMM 2008).

### Structural Formula

For parameter **k** (number of ports per switch):

| Component | Count | Formula |
|---|---|---|
| Pods | k | k |
| Core switches | (k/2)² | 16 for k=8 |
| Aggregation switches | k × (k/2) | 32 for k=8 |
| Edge switches | k × (k/2) | 32 for k=8 |
| **Total switches** | **5k²/4** | **80 for k=8** |
| **Hosts** | **k³/4** | **128 for k=8** |

---

## 2. ASCII Art Diagram (k=4, simplified)

```
                        CORE LAYER
               ┌──────────────────────────┐
               │  c_0_0  c_0_1  c_1_0  c_1_1  │
               └──┬──┬──────┬──┬──────┬──┬────┘
                  │  │      │  │      │  │
        ──────────────────────────────────────
        POD 0                   POD 1
        ─────────               ─────────
     AGGREGATION               AGGREGATION
     a_0_0  a_0_1             a_1_0  a_1_1
       │  ╲ ╱  │               │  ╲ ╱  │
       │   ╳   │               │   ╳   │
       │  ╱ ╲  │               │  ╱ ╲  │
      EDGE      EDGE           EDGE      EDGE
     e_0_0    e_0_1           e_1_0    e_1_1
      │  │     │  │            │  │     │  │
     h h  h h  h h  h h       h h  h h  h h  h h
    0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7   ...

   (h = host, numbered as h_pod_edge_idx)
```

### k=8 Fat-Tree (LAFS Target — 128 hosts)

```
                           ── CORE LAYER (16 switches) ──
    c_0_0 c_0_1 c_0_2 c_0_3  c_1_0 c_1_1 c_1_2 c_1_3  … c_3_3
       │     │     │     │      │     │     │     │          │
  ─────────────────────────────────────────────────────────────
  POD 0              POD 1              …         POD 7
  ─────              ─────                        ─────
  AGG: a_0_0…a_0_3   AGG: a_1_0…a_1_3           AGG: a_7_0…a_7_3
      │╲│╲│╲│             │╲│╲│╲│                    │╲│╲│╲│
  EDGE: e_0_0…e_0_3   EDGE: e_1_0…e_1_3         EDGE: e_7_0…e_7_3
      │ │ │ │               │ │ │ │                    │ │ │ │
  h_0_0_0…h_0_0_3       h_1_0_0…h_1_0_3          h_7_3_0…h_7_3_3
  h_0_1_0…h_0_1_3       …                         (4 hosts/edge sw)
  h_0_2_0…h_0_2_3
  h_0_3_0…h_0_3_3
  (16 hosts/pod × 8 pods = 128 total)
```

---

## 3. Node Naming Convention

| Node type | Pattern | Example | Count (k=8) |
|---|---|---|---|
| Core switch | `c_<row>_<col>` | `c_0_0`, `c_3_3` | 16 |
| Aggregation | `a_<pod>_<idx>` | `a_0_0`, `a_7_3` | 32 |
| Edge switch | `e_<pod>_<idx>` | `e_0_0`, `e_7_3` | 32 |
| Host | `h_<pod>_<edge>_<idx>` | `h_0_0_0`, `h_7_3_3` | 128 |

**Ranges (k=8, half_k=4):**
- `row`, `col` ∈ [0, 3] for core
- `pod` ∈ [0, 7] for agg / edge
- `idx` ∈ [0, 3] for agg, edge within pod
- `idx` ∈ [0, 3] for hosts within edge switch

---

## 4. IP Addressing

```
Host h_<pod>_<edge>_<idx>  →  10.<pod>.<edge>.<idx + 2>
```

| Host | IP |
|---|---|
| `h_0_0_0` | `10.0.0.2` |
| `h_0_0_1` | `10.0.0.3` |
| `h_0_0_2` | `10.0.0.4` |
| `h_0_0_3` | `10.0.0.5` |
| `h_7_3_3` | `10.7.3.5` |

**.0 and .1 are reserved** (network address and gateway).

Subnet mask: `/8` (all 10.x.x.x hosts are in the same L3 network; routing is handled by the SDN controller, not IP subnetting).

---

## 5. MAC Addressing

```
Host h_<pod>_<edge>_<idx>  →  00:00:0a:<pod_hex>:<edge_hex>:<(idx+2)_hex>
```

| Host | MAC |
|---|---|
| `h_0_0_0` | `00:00:0a:00:00:02` |
| `h_7_3_3` | `00:00:0a:07:03:05` |

The MAC mirrors the IP (0a = 10 in hex), making ARP debugging straightforward.

---

## 6. Port Numbering

### Edge switch `e_pod_i`

| Port | Connected to | Direction |
|---|---|---|
| 1 | `h_pod_i_0` | downlink (host) |
| 2 | `h_pod_i_1` | downlink (host) |
| … | … | … |
| k/2 | `h_pod_i_(k/2-1)` | downlink (host) |
| k/2+1 | `a_pod_0` | uplink (agg) |
| k/2+2 | `a_pod_1` | uplink (agg) |
| … | … | … |
| k | `a_pod_(k/2-1)` | uplink (agg) |

### Aggregation switch `a_pod_i`

| Port | Connected to | Direction |
|---|---|---|
| 1 | `e_pod_0` | downlink (edge) |
| 2 | `e_pod_1` | downlink (edge) |
| … | … | … |
| k/2 | `e_pod_(k/2-1)` | downlink (edge) |
| k/2+1 | `c_i_0` | uplink (core) |
| k/2+2 | `c_i_1` | uplink (core) |
| … | … | … |
| k | `c_i_(k/2-1)` | uplink (core) |

### Core switch `c_i_j`

| Port | Connected to | Direction |
|---|---|---|
| 1 | `a_0_i` (pod 0) | downlink |
| 2 | `a_1_i` (pod 1) | downlink |
| … | … | … |
| k | `a_(k-1)_i` (pod k-1) | downlink |

---

## 7. Path Properties

### Within the same edge switch (0 hops in fabric)
```
h_0_0_0 → e_0_0 → h_0_0_1
(2 hops total, 1 path)
```

### Within the same pod (different edge switches)
```
h_0_0_0 → e_0_0 → a_0_? → e_0_1 → h_0_1_0
(4 hops, k/2 = 4 equal-cost paths for k=8)
```

### Between different pods (cross-pod)
```
h_0_0_0 → e_0_0 → a_0_? → c_?_? → a_1_? → e_1_0 → h_1_0_0
(6 hops, (k/2)² = 16 equal-cost paths for k=8)
```

**Summary:**

| Source | Destination | Path Length | # ECMP Paths (k=8) |
|---|---|---|---|
| Same edge switch | Same edge switch | 2 | 1 |
| Same pod | Different edge | 4 | 4 |
| Different pods | Any | 6 | 16 |

---

## 8. Usage Guide

### 8.1 Pure Graph (Windows / no Mininet)

```python
from src.topology.fattree import FatTreeGraph

# Build k=8 Fat-tree graph
g = FatTreeGraph(k=8)
print(g.summary())

# Count nodes
print(f"Hosts: {g.n_hosts}")          # 128
print(f"Switches: {g.n_switches}")    # 80

# IP lookup
ip = g.get_host_ip("h_0_0_0")        # '10.0.0.2'
node = g.node_for_ip("10.7.3.5")     # 'h_7_3_3'

# All ECMP paths between two hosts
paths = g.get_paths("h_0_0_0", "h_1_0_0")
print(f"{len(paths)} paths, length {len(paths[0])} hops")  # 16, 7

# All-pairs paths (slow for k=8 — cache the result)
all_paths = g.get_all_paths()
print(f"Total (src,dst) pairs: {len(all_paths)}")  # 16256

# Link capacity (Gbps)
cap = g.get_link_capacity("e_0_0", "h_0_0_0")    # 1.0
cap = g.get_link_capacity("a_0_0", "e_0_0")      # 10.0
```

### 8.2 Mininet Topology (Linux + root)

```python
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from src.topology.fattree import FatTreeTopo

topo = FatTreeTopo(k=8, bw_host=1.0, bw_core=10.0)
net  = Mininet(
    topo=topo,
    switch=OVSSwitch,
    controller=RemoteController,
    autoSetMacs=True,
    autoStaticArp=True,
)
net.addController("c0", controller=RemoteController,
                  ip="127.0.0.1", port=6633)
net.start()

# Run commands on hosts
h0 = net.get("h_0_0_0")
h1 = net.get("h_1_0_0")
h0.cmd(f"ping -c 5 {h1.IP()}")

# Path computation (delegates to FatTreeGraph)
paths = topo.get_paths("h_0_0_0", "h_1_0_0")

net.stop()
```

### 8.3 NetworkBuilder (High-Level API, Linux + root)

```python
from src.topology.network_builder import NetworkBuilder, NetworkConfig, ControllerConfig

cfg = NetworkConfig(
    k=8,
    bw_host=1.0,
    bw_core=10.0,
    controller=ControllerConfig(host="127.0.0.1", port=6633, timeout=30),
    autoarp=True,
)

# Context manager — auto start/stop
with NetworkBuilder.managed(cfg) as nb:
    loss = nb.pingall()
    print(f"Packet loss: {loss}%")

    sent, rcvd, rtt = nb.ping_pair("h_0_0_0", "h_1_0_0", count=10)

    result = nb.iperf_pair("h_0_0_0", "h_1_0_0", duration=5)
    print(f"Throughput: {result['throughput_mbps']:.1f} Mbps")

    stats = nb.collect_link_stats()
    flows = nb.dump_all_flows()
```

---

## 9. Running Tests

### Unit tests (Windows / Linux, no root)
```bash
# All unit tests
pytest tests/unit/test_topology.py -v

# Filter by category
pytest tests/unit/test_topology.py -v -k "k8"
pytest tests/unit/test_topology.py -v -k "paths"
pytest tests/unit/test_topology.py -v -k "edges"

# Run directly
python tests/unit/test_topology.py --verbose
```

### Integration tests (Ubuntu, requires root + Mininet)
```bash
# k=4 topology (faster, ~2 min)
sudo python tests/integration/test_topology_integration.py --k 4 --verbose

# k=8 topology (full 128-host test, ~10 min, >= 8 GB RAM)
sudo python tests/integration/test_topology_integration.py --k 8 --verbose

# Skip heavy k=8 performance tests
sudo python tests/integration/test_topology_integration.py --skip-k8

# Via pytest with timeout
sudo pytest tests/integration/test_topology_integration.py -v --timeout=600
```

---

## 10. Troubleshooting

### Problem: `No path between h_X and h_Y`
- **Cause:** Topology graph not fully connected — check build log.
- **Fix:** Verify k is even and >= 2.  Check `nx.is_connected(g.graph)`.

### Problem: `OVS bridge protocols=[]` (empty)
- **Cause:** OpenFlow 1.3 not set after OVS start.
- **Fix:** After `net.start()`, force OF 1.3:
  ```bash
  ovs-vsctl set bridge <sw_name> protocols=OpenFlow13
  ```
  Or use `NetworkBuilder` which does this automatically.

### Problem: `Switch not connected to controller`
- **Cause:** Ryu not started, wrong port, or firewall blocking.
- **Fix:**
  ```bash
  # Check Ryu is running
  ps aux | grep ryu-manager
  # Check port
  ss -tlnp | grep 6633
  # Check OVS controller config
  ovs-vsctl show
  ```

### Problem: `pingAll` shows packet loss
- **Cause:** Controller not installed forwarding rules yet.
- **Fix:** Run `pingAll` twice; the first pass triggers L2 learning and may lose packets.  In production, use proactive flow installation via LAFS controller.

### Problem: `MemoryError` or system hang with k=8
- **Cause:** 128 Mininet namespaces require significant RAM (~6–8 GB).
- **Fix:**
  ```bash
  # Check available RAM
  free -h
  # If insufficient, use k=4 for development, k=8 only for experiments
  ```

### Problem: `TCLink bw parameter` warning
- **Cause:** Mininet expects bw in Mbps, not Gbps.
- **Fix:** Already handled — `FatTreeTopo._link_opts()` converts Gbps → Mbps
  (`bw * 1000`).

---

## 11. References

1. Al-Fares, M., Loukissas, A., & Vahdat, A. (2008). **A scalable, commodity data center network architecture.** ACM SIGCOMM.
2. Leiserson, C. E. (1985). **Fat-trees: Universal networks for hardware-efficient supercomputing.** IEEE Transactions on Computers.
3. Mininet documentation: http://mininet.org
4. Ryu SDN framework: https://ryu-sdn.org
