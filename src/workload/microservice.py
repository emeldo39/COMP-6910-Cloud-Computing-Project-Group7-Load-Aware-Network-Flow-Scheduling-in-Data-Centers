"""
LAFS — Google Microservice RPC Chain Workload Generator
=======================================================
COMP-6910 — Group 7

Generates synthetic traffic traces modelling Google-style microservice
architectures where user-facing services decompose requests into a graph
of internal RPC calls.  The key networking consequence is **tail-latency
cascading**: a P99 delay at any service in a fan-out fan-in graph causes
the aggregating service to wait for the slowest shard, amplifying tail
latency for the end user.

Background — Google microservice patterns
-----------------------------------------
From Gan et al. (2019) "An Open-Source Benchmark Suite for Microservices
and Their Hardware-Software Implications for Cloud & Edge Systems"
(ASPLOS 2019) and Google SRE book:

  * Fan-out / fan-in: frontend calls N leaf services in parallel, waits
    for ALL responses before replying (slowest determines latency).
  * Chain (sequential): A → B → C → D, each hop adds latency.
  * Mixed DAG: a combination of the above (real services).

Traffic characteristics
-----------------------
  * RPC request bodies:   50 B – 5 KB   (tiny, latency-sensitive)
  * RPC response bodies:  1 KB – 100 KB (small-medium, variable)
  * Data payloads:        100 KB – 10 MB (cache fills, query results)
  * Protocol:             TCP (gRPC over HTTP/2)
  * Dst port:             50051 (gRPC), 8080 (HTTP/2), 443 (HTTPS)

Arrival model
-------------
User requests arrive as a Poisson process at rate λ req/s.
Each request fans out to K leaf services, each of which may chain
through 1–3 hops.  Total flows per user request = 2 × fan_out × hops
(request + response per hop).

Service placement
-----------------
Services are placed on topology hosts using two strategies:
  * 'random'   — services placed uniformly at random on hosts.
  * 'rack'     — services in the same microservice cluster prefer the
                 same rack (edge switch), reducing cross-pod traffic.
                 This is the dominant pattern in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow


# ── gRPC / HTTP2 ports ────────────────────────────────────────────────────────
_GRPC_PORT: int = 50051
_HTTP2_PORTS: List[int] = [443, 8080, 50051]

# ── RPC size distributions (bytes) ───────────────────────────────────────────
_RPC_REQUEST_RANGE:  Tuple[int, int] = (50, 5_000)          # tiny
_RPC_RESPONSE_RANGE: Tuple[int, int] = (1_000, 100_000)     # small-medium
_DATA_PAYLOAD_RANGE: Tuple[int, int] = (100_000, 10_000_000) # data xfer


# =============================================================================
# ServiceGraph — describes the microservice call topology
# =============================================================================
@dataclass
class ServiceNode:
    """
    A single microservice.

    Parameters
    ----------
    service_id : str
        Unique identifier (e.g. 'frontend', 'auth', 'search_0').
    service_type : str
        One of: 'frontend', 'middleware', 'leaf', 'db'.
    response_size_range : (int, int)
        (min, max) bytes for the response body.
    """
    service_id: str
    service_type: str = "middleware"
    response_size_range: Tuple[int, int] = _RPC_RESPONSE_RANGE


@dataclass
class ServiceGraph:
    """
    A directed acyclic graph of microservice calls.

    ``edges`` is a list of (caller, callee) pairs; the caller sends a
    request to the callee and waits for a response.

    Provides several factory class methods for common patterns.
    """
    nodes: List[ServiceNode]
    edges: List[Tuple[str, str]]       # (caller_id, callee_id)

    @classmethod
    def linear_chain(cls, depth: int = 4) -> "ServiceGraph":
        """
        A → B → C → D  (depth nodes in a straight line).
        Models a sequential processing pipeline.
        """
        if depth < 2:
            raise ValueError("depth must be >= 2")
        nodes = [
            ServiceNode(f"svc_{i}", "frontend" if i == 0 else
                        "leaf" if i == depth - 1 else "middleware")
            for i in range(depth)
        ]
        edges = [(nodes[i].service_id, nodes[i + 1].service_id)
                 for i in range(depth - 1)]
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def fan_out(cls, fan: int = 8) -> "ServiceGraph":
        """
        Frontend fans out to ``fan`` leaf services in parallel.
        Models search sharding (e.g. web search across 8 index shards).
        """
        if fan < 1:
            raise ValueError("fan must be >= 1")
        frontend = ServiceNode("frontend", "frontend")
        leaves = [ServiceNode(f"leaf_{i}", "leaf") for i in range(fan)]
        nodes = [frontend] + leaves
        edges = [(frontend.service_id, leaf.service_id) for leaf in leaves]
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def mixed_dag(cls, fan: int = 4, depth: int = 2) -> "ServiceGraph":
        """
        Frontend → [middleware_0, middleware_1, …] → [db_0, db_1, …]
        Models a two-tier fan-out (middleware + database tier).
        """
        nodes: List[ServiceNode] = [ServiceNode("frontend", "frontend")]
        edges: List[Tuple[str, str]] = []

        middlewares = [ServiceNode(f"mw_{i}", "middleware") for i in range(fan)]
        nodes.extend(middlewares)
        for mw in middlewares:
            edges.append(("frontend", mw.service_id))

        dbs = [ServiceNode(f"db_{i}", "db",
                           response_size_range=_DATA_PAYLOAD_RANGE)
               for i in range(fan)]
        nodes.extend(dbs)
        for i, mw in enumerate(middlewares):
            edges.append((mw.service_id, dbs[i % len(dbs)].service_id))

        return cls(nodes=nodes, edges=edges)


# =============================================================================
# MicroserviceConfig
# =============================================================================
@dataclass
class MicroserviceConfig:
    """
    Configuration for the microservice RPC workload generator.

    Parameters
    ----------
    n_requests : int
        Number of user-facing requests to simulate.
    arrival_rate : float
        Poisson arrival rate in requests/second.
    graph_type : str
        One of: 'chain', 'fan_out', 'mixed' — selects the service DAG.
    fan_out : int
        Fan-out degree (used by 'fan_out' and 'mixed' graph types).
    chain_depth : int
        Chain length (used by 'chain' graph type).
    placement : str
        'random' or 'rack' — service-to-host placement strategy.
    start_time : float
        Simulation start time (seconds).
    seed : int
        RNG seed.
    include_data_flows : bool
        If True, add a larger data-payload flow for leaf→middleware responses
        (models cache fills, query result sets).
    n_tenants : int
        Number of independent microservice clusters (each gets its own
        service graph and host slice).
    """
    n_requests: int = 500
    arrival_rate: float = 100.0        # requests/second
    graph_type: str = "mixed"          # 'chain', 'fan_out', 'mixed'
    fan_out: int = 4
    chain_depth: int = 3
    placement: str = "rack"            # 'random' or 'rack'
    start_time: float = 0.0
    seed: int = 42
    include_data_flows: bool = True
    n_tenants: int = 1

    def __post_init__(self) -> None:
        if self.graph_type not in ("chain", "fan_out", "mixed"):
            raise ValueError(
                f"graph_type must be 'chain', 'fan_out', or 'mixed', "
                f"got {self.graph_type!r}"
            )
        if self.placement not in ("random", "rack"):
            raise ValueError(f"placement must be 'random' or 'rack'")
        if self.n_requests < 1:
            raise ValueError("n_requests must be >= 1")
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")


# =============================================================================
# MicroserviceRPCGenerator
# =============================================================================
class MicroserviceRPCGenerator:
    """
    Generates flows matching Google-style microservice RPC chain traffic.

    For each user request, the generator traverses the service graph and
    produces two flows per edge:
      1. Request  (caller → callee): tiny (50 B – 5 KB)
      2. Response (callee → caller): small-medium (1 KB – 100 KB)
         or data payload (100 KB – 10 MB) for database-tier nodes.

    Fan-out edges are treated as concurrent (same arrival_time).
    Chain edges are sequential (arrival_time increments by RTT estimate).

    Parameters
    ----------
    topology : FatTreeGraph
        Host pool for service placement.
    config : MicroserviceConfig
        Generation parameters.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> cfg  = MicroserviceConfig(n_requests=100, graph_type='fan_out', fan_out=4)
    >>> gen  = MicroserviceRPCGenerator(topo, cfg)
    >>> flows = gen.generate()
    >>> # fan_out: 1 frontend→leaf + 1 leaf→frontend per leaf = 8 flows/request
    >>> len(flows) // 100
    8
    """

    # Estimated one-way RTT per hop in the fat-tree (propagation + switch).
    _HOP_RTT_S: float = 0.0002   # 200 µs

    def __init__(
        self,
        topology: FatTreeGraph,
        config: Optional[MicroserviceConfig] = None,
    ) -> None:
        self.topology = topology
        self.config = config or MicroserviceConfig()
        self._rng = random.Random(self.config.seed)

        all_hosts = sorted(topology.hosts)
        if len(all_hosts) < 2:
            raise ValueError("Topology must have at least 2 hosts")

        # Build service graphs for each tenant.
        self._graphs: List[ServiceGraph] = []
        for _ in range(config.n_tenants):
            self._graphs.append(self._build_graph())

        # Place services on hosts for each tenant.
        # Each tenant gets a non-overlapping slice of hosts.
        hosts_per_tenant = max(2, len(all_hosts) // config.n_tenants)
        self._placements: List[Dict[str, str]] = []  # service_id → host_ip
        for t_id in range(config.n_tenants):
            host_slice = all_hosts[
                t_id * hosts_per_tenant: (t_id + 1) * hosts_per_tenant
            ]
            if not host_slice:
                host_slice = all_hosts  # fallback
            placement = self._place_services(self._graphs[t_id], host_slice)
            self._placements.append(placement)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> List[Flow]:
        """
        Generate all RPC flows for ``config.n_requests`` user requests.

        Returns
        -------
        List[Flow]
            Flows sorted by arrival_time.
        """
        all_flows: List[Flow] = []
        cfg = self.config
        t = cfg.start_time

        for req_idx in range(cfg.n_requests):
            # Poisson inter-arrival.
            inter = self._rng.expovariate(cfg.arrival_rate)
            t += inter

            # Choose a tenant round-robin.
            tenant_id = req_idx % cfg.n_tenants
            graph = self._graphs[tenant_id]
            placement = self._placements[tenant_id]

            req_flows = self._generate_request_flows(
                graph, placement, req_idx, tenant_id, t
            )
            all_flows.extend(req_flows)

        all_flows.sort(key=lambda f: f.arrival_time)
        return all_flows

    def flows_per_request(self) -> int:
        """
        Expected number of flows generated per user request.
        (2 flows per DAG edge: request + response; + data flows for db nodes.)
        """
        graph = self._graphs[0]
        base = 2 * len(graph.edges)   # request + response per edge
        if self.config.include_data_flows:
            db_edges = sum(
                1 for (_, dst) in graph.edges
                for node in graph.nodes
                if node.service_id == dst and node.service_type == "db"
            )
            base += db_edges          # extra data-payload flow per db edge
        return base

    def service_placement(self, tenant_id: int = 0) -> Dict[str, str]:
        """Return the {service_id: host_ip} placement for a given tenant."""
        return dict(self._placements[tenant_id])

    # ── Request flow generation ───────────────────────────────────────────────

    def _generate_request_flows(
        self,
        graph: ServiceGraph,
        placement: Dict[str, str],
        req_idx: int,
        tenant_id: int,
        base_t: float,
    ) -> List[Flow]:
        """Traverse the graph and emit request + response flows."""
        flows: List[Flow] = []
        cfg = self.config

        # Compute topological order for sequential timing.
        topo_order = self._topological_order(graph)
        # Earliest start time for each node (after all its callers have responded).
        node_ready: Dict[str, float] = {n.service_id: base_t for n in graph.nodes}

        for caller_id in topo_order:
            for (src_id, dst_id) in graph.edges:
                if src_id != caller_id:
                    continue

                t_call = node_ready[caller_id]
                src_ip = placement.get(src_id)
                dst_ip = placement.get(dst_id)
                if not src_ip or not dst_ip or src_ip == dst_ip:
                    # Skip if same host (no network flow needed) or unmapped.
                    continue

                dst_node = next(
                    (n for n in graph.nodes if n.service_id == dst_id), None
                )
                is_db = dst_node and dst_node.service_type == "db"

                req_size = self._rng.randint(*_RPC_REQUEST_RANGE)
                resp_lo, resp_hi = (
                    _DATA_PAYLOAD_RANGE if is_db else _RPC_RESPONSE_RANGE
                )
                resp_size = self._rng.randint(resp_lo, resp_hi)

                dst_port = self._rng.choice(_HTTP2_PORTS)
                src_port = self._rng.randint(32768, 60999)
                fid_base = f"t{tenant_id}_rpc_r{req_idx:05d}_{src_id}_{dst_id}"

                # Request: caller → callee.
                flows.append(Flow(
                    flow_id=fid_base + "_req",
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=6,
                    size_bytes=req_size,
                    arrival_time=t_call,
                ))

                # Estimate RTT for request delivery.
                rtt_s = self._HOP_RTT_S * 2  # 2-way

                # Response: callee → caller (after request processed).
                t_resp = t_call + rtt_s
                flows.append(Flow(
                    flow_id=fid_base + "_resp",
                    src_ip=dst_ip,
                    dst_ip=src_ip,
                    src_port=dst_port,
                    dst_port=src_port,
                    protocol=6,
                    size_bytes=resp_size,
                    arrival_time=t_resp,
                ))

                # Optional data-payload flow for database tier.
                if is_db and cfg.include_data_flows:
                    data_size = self._rng.randint(*_DATA_PAYLOAD_RANGE)
                    flows.append(Flow(
                        flow_id=fid_base + "_data",
                        src_ip=dst_ip,
                        dst_ip=src_ip,
                        src_port=dst_port,
                        dst_port=src_port + 1,
                        protocol=6,
                        size_bytes=data_size,
                        arrival_time=t_resp + rtt_s,
                    ))

                # Update ready time for callee (it can call its own dependencies
                # only after receiving the request).
                node_ready[dst_id] = max(node_ready[dst_id], t_call + rtt_s)

        return flows

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> ServiceGraph:
        cfg = self.config
        if cfg.graph_type == "chain":
            return ServiceGraph.linear_chain(depth=cfg.chain_depth)
        elif cfg.graph_type == "fan_out":
            return ServiceGraph.fan_out(fan=cfg.fan_out)
        else:
            return ServiceGraph.mixed_dag(fan=cfg.fan_out, depth=2)

    def _place_services(
        self, graph: ServiceGraph, hosts: List[str]
    ) -> Dict[str, str]:
        """
        Assign each service to a host IP.

        'rack' placement: services of the same type prefer the same edge
        switch (same pod/rack) — reducing cross-pod traffic for intra-
        service communication. In practice this is realistic: DB replicas
        and caches co-locate with the service that uses them.

        'random' placement: uniform random assignment.
        """
        placement: Dict[str, str] = {}
        cfg = self.config

        if cfg.placement == "rack":
            # Group hosts by their edge switch (pod + edge index from name).
            rack_groups: Dict[str, List[str]] = {}
            for h in hosts:
                # e.g. h_0_0_0 → rack key "0_0"
                parts = h.split("_")
                rack_key = "_".join(parts[1:3]) if len(parts) >= 3 else "0"
                rack_groups.setdefault(rack_key, []).append(h)
            racks = list(rack_groups.values())

            for i, node in enumerate(graph.nodes):
                rack = racks[i % len(racks)]
                host = self._rng.choice(rack)
                placement[node.service_id] = self.topology.get_host_ip(host)
        else:
            shuffled = list(hosts)
            self._rng.shuffle(shuffled)
            for i, node in enumerate(graph.nodes):
                host = shuffled[i % len(shuffled)]
                placement[node.service_id] = self.topology.get_host_ip(host)

        return placement

    @staticmethod
    def _topological_order(graph: ServiceGraph) -> List[str]:
        """
        Kahn's algorithm: return service IDs in topological order
        (callers before callees).
        """
        in_degree: Dict[str, int] = {n.service_id: 0 for n in graph.nodes}
        for (_, dst) in graph.edges:
            in_degree[dst] = in_degree.get(dst, 0) + 1

        queue = sorted(
            [nid for nid, deg in in_degree.items() if deg == 0]
        )
        order: List[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            neighbors = sorted(
                [dst for (src, dst) in graph.edges if src == nid]
            )
            for neighbor in neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort()
        return order
