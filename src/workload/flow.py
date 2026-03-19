"""
LAFS — Flow dataclass
=====================
COMP 6910 — Group 7

A Flow represents a single network transmission identified by its 5-tuple.
It carries all scheduling-relevant metadata and is the primary unit of work
passed between the workload generator, the scheduler, and the metrics system.

Elephant / mice classification thresholds follow standard data-centre
conventions (Benson et al., IMC 2010):
  * Mice   : size < 100 KB  (latency-sensitive, short-lived)
  * Elephant: size ≥ 1 MB   (bandwidth-hungry, long-lived)
  * Medium : everything in between

Protocol numbers (IANA):
  1  = ICMP
  6  = TCP
  17 = UDP
"""

from __future__ import annotations

import ipaddress
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Flow-size thresholds (bytes) ──────────────────────────────────────────────
MICE_THRESHOLD_BYTES: int = 100_000        # < 100 KB  → mice
ELEPHANT_THRESHOLD_BYTES: int = 1_000_000  # ≥ 1 MB    → elephant

# ── Supported IP protocols ────────────────────────────────────────────────────
VALID_PROTOCOLS: frozenset = frozenset({1, 6, 17})   # ICMP, TCP, UDP
_PROTO_NAMES: dict = {1: "ICMP", 6: "TCP", 17: "UDP"}

# ── Ideal link rate assumption for slowdown calculation ───────────────────────
# Hosts connect to edge switches at 1 Gbps in the LAFS Fat-tree topology.
_HOST_LINK_RATE_GBPS: float = 1.0


# =============================================================================
# Flow
# =============================================================================
@dataclass
class Flow:
    """
    A network flow identified by its 5-tuple.

    Mandatory parameters
    --------------------
    flow_id : str
        Unique identifier for this flow.  Use ``Flow.new_id()`` for an
        auto-generated UUID4-based ID.
    src_ip : str
        Source IPv4 address (dotted-decimal, e.g. "10.0.0.2").
    dst_ip : str
        Destination IPv4 address.
    src_port : int
        Source TCP/UDP port in [0, 65535].
    dst_port : int
        Destination TCP/UDP port in [0, 65535].
    protocol : int
        IP protocol number: 1 (ICMP), 6 (TCP), or 17 (UDP).
    size_bytes : int
        Total flow size in bytes (≥ 0).

    Optional parameters
    -------------------
    arrival_time : float
        UNIX timestamp when the flow arrived at the scheduler.
        Defaults to the current wall-clock time.
    deadline : float or None
        Absolute UNIX timestamp by which the flow must complete.
        ``None`` means no deadline.

    Scheduler-set fields (written after path assignment)
    ----------------------------------------------------
    assigned_path : List[str] or None
        Ordered node-name path from the scheduler, e.g.
        ["h_0_0_0", "e_0_0", "a_0_1", "c_1_0", "a_1_0", "e_1_0", "h_1_0_0"].
    schedule_time : float or None
        Wall-clock timestamp when the scheduler made its decision.
    completion_time : float or None
        Wall-clock timestamp when the last byte of the flow was received
        (set by the metrics/monitoring layer after flow completes).

    Derived properties
    ------------------
    five_tuple       : (src_ip, dst_ip, src_port, dst_port, protocol)
    is_mice          : size_bytes < MICE_THRESHOLD_BYTES
    is_elephant      : size_bytes >= ELEPHANT_THRESHOLD_BYTES
    flow_type        : "mice" | "medium" | "elephant"
    fct              : float or None   — completion_time − schedule_time (s)
    ideal_fct        : float           — size_bytes × 8 / link_rate_bps (s)
    slowdown         : float or None   — fct / ideal_fct
    meets_deadline   : bool or None    — True iff completion_time ≤ deadline
    """

    # ── Required fields ───────────────────────────────────────────────────────
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int       # 1=ICMP, 6=TCP, 17=UDP
    size_bytes: int     # total flow size in bytes

    # ── Optional fields with sensible defaults ────────────────────────────────
    arrival_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None

    # ── Scheduler-set fields (not compared in equality checks) ────────────────
    assigned_path: Optional[List[str]] = field(default=None, compare=False, repr=False)
    schedule_time: Optional[float] = field(default=None, compare=False, repr=False)
    completion_time: Optional[float] = field(default=None, compare=False, repr=False)

    # ── Post-init validation ──────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate all fields; raise ValueError with a descriptive message."""
        # flow_id
        if not self.flow_id or not isinstance(self.flow_id, str):
            raise ValueError("flow_id must be a non-empty string")

        # IP addresses
        for field_name, ip in (("src_ip", self.src_ip), ("dst_ip", self.dst_ip)):
            try:
                ipaddress.IPv4Address(ip)
            except (ipaddress.AddressValueError, ValueError) as exc:
                raise ValueError(
                    f"Flow '{self.flow_id}': {field_name}={ip!r} is not a valid IPv4 address"
                ) from exc

        # Ports
        for field_name, port in (("src_port", self.src_port), ("dst_port", self.dst_port)):
            if not isinstance(port, int) or not (0 <= port <= 65535):
                raise ValueError(
                    f"Flow '{self.flow_id}': {field_name}={port} not in [0, 65535]"
                )

        # Protocol
        if self.protocol not in VALID_PROTOCOLS:
            raise ValueError(
                f"Flow '{self.flow_id}': protocol={self.protocol} not in "
                f"{{1 (ICMP), 6 (TCP), 17 (UDP)}}"
            )

        # Size
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError(
                f"Flow '{self.flow_id}': size_bytes={self.size_bytes} must be a non-negative int"
            )

        # Deadline coherence
        if self.deadline is not None and self.deadline < self.arrival_time:
            raise ValueError(
                f"Flow '{self.flow_id}': deadline ({self.deadline:.3f}) is before "
                f"arrival_time ({self.arrival_time:.3f})"
            )

    # ── Class-level helpers ───────────────────────────────────────────────────

    @staticmethod
    def new_id() -> str:
        """Return a short unique flow ID (first 8 hex chars of a UUID4)."""
        return uuid.uuid4().hex[:8]

    @classmethod
    def create(
        cls,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int = 6,
        size_bytes: int = 1024,
        **kwargs,
    ) -> "Flow":
        """
        Convenience factory: auto-generate a flow_id if not provided.

        Example
        -------
        >>> f = Flow.create("10.0.0.2", "10.1.0.2", 12345, 80)
        """
        flow_id = kwargs.pop("flow_id", cls.new_id())
        return cls(
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            size_bytes=size_bytes,
            **kwargs,
        )

    # ── 5-tuple ───────────────────────────────────────────────────────────────

    @property
    def five_tuple(self) -> Tuple[str, str, int, int, int]:
        """
        Canonical 5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol).

        This is the input to the ECMP hash function and is also used as a
        dict key for per-flow state in the controller.
        """
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

    # ── Flow type classification ──────────────────────────────────────────────

    @property
    def is_mice(self) -> bool:
        """True if size_bytes < 100 KB (latency-sensitive short flow)."""
        return self.size_bytes < MICE_THRESHOLD_BYTES

    @property
    def is_elephant(self) -> bool:
        """True if size_bytes >= 1 MB (bandwidth-hungry long flow)."""
        return self.size_bytes >= ELEPHANT_THRESHOLD_BYTES

    @property
    def flow_type(self) -> str:
        """One of 'mice', 'medium', or 'elephant'."""
        if self.is_mice:
            return "mice"
        if self.is_elephant:
            return "elephant"
        return "medium"

    # ── Timing / performance metrics ──────────────────────────────────────────

    @property
    def fct(self) -> Optional[float]:
        """
        Flow completion time in seconds.

        Defined as ``completion_time − schedule_time``.
        Returns ``None`` until both timestamps are set.
        """
        if self.schedule_time is not None and self.completion_time is not None:
            return max(0.0, self.completion_time - self.schedule_time)
        return None

    @property
    def ideal_fct(self) -> float:
        """
        Ideal (minimum possible) FCT in seconds.

        Assumes the entire flow is transmitted at the host-link rate
        (1 Gbps) with no queuing or propagation delay.  This lower bound
        is used to compute normalised slowdown.

        Formula: size_bytes × 8 bits/byte ÷ (1e9 bits/second)
        """
        return (self.size_bytes * 8) / (_HOST_LINK_RATE_GBPS * 1e9)

    @property
    def slowdown(self) -> Optional[float]:
        """
        Normalised slowdown: actual_fct / ideal_fct.

        A slowdown of 1.0 means the flow completed at line rate.
        Values > 1.0 indicate queuing / contention.
        Returns ``None`` until FCT is available or if size_bytes == 0.
        """
        actual = self.fct
        if actual is None or self.size_bytes == 0:
            return None
        ideal = self.ideal_fct
        if ideal == 0.0:
            return None
        return actual / ideal

    @property
    def meets_deadline(self) -> Optional[bool]:
        """
        Whether the flow completed before its deadline.

        Returns:
          True  — completion_time is set and ≤ deadline
          False — completion_time is set and > deadline
          None  — no deadline set, or completion_time not yet recorded
        """
        if self.deadline is None or self.completion_time is None:
            return None
        return self.completion_time <= self.deadline

    @property
    def head_of_line_delay(self) -> Optional[float]:
        """
        Time between arrival and scheduling decision, in seconds.

        Returns ``None`` until schedule_time is set.
        """
        if self.schedule_time is None:
            return None
        return max(0.0, self.schedule_time - self.arrival_time)

    # ── Protocol helpers ──────────────────────────────────────────────────────

    @property
    def protocol_name(self) -> str:
        """Human-readable protocol name: 'TCP', 'UDP', or 'ICMP'."""
        return _PROTO_NAMES.get(self.protocol, f"PROTO_{self.protocol}")

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        size_str = (
            f"{self.size_bytes / 1e6:.2f} MB"
            if self.size_bytes >= 1_000_000
            else f"{self.size_bytes / 1e3:.1f} KB"
        )
        return (
            f"Flow(id={self.flow_id!r}, "
            f"{self.src_ip}:{self.src_port}→{self.dst_ip}:{self.dst_port}, "
            f"{self.protocol_name}, {size_str}, type={self.flow_type})"
        )
