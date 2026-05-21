# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase-vector gossip synchronisation

"""Deterministic phase-vector gossip primitives.

This module defines the canonical JSON wire record for UPDE node phase sharing
and a local ingestion/replay implementation for offline validation. All
messages are digest checked, sequence-watermarked, protocol-versioned, and
phase-step bounded before they can influence a local node; real transports are
kept outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "DistributedSyncConfig",
    "GossipIngestResult",
    "LossyGossipReplayResult",
    "PhaseGossipNode",
    "PhaseSyncMessage",
    "simulate_lossy_phase_gossip",
]

_TWO_PI = 2.0 * math.pi
_PROTOCOL_KIND = "spo.phase_sync"


@dataclass(frozen=True)
class DistributedSyncConfig:
    """Safety envelope for phase-vector gossip synchronisation."""

    node_id: str
    n_oscillators: int
    protocol_version: int = 1
    phase_blend: float = 0.25
    max_phase_step_rad: float = 0.1
    peer_timeout_s: float = 5.0
    local_weight: float = 1.0
    peer_weight: float = 1.0

    def __post_init__(self) -> None:
        _node_id(self.node_id)
        if (
            isinstance(self.n_oscillators, bool)
            or not isinstance(self.n_oscillators, int)
            or self.n_oscillators <= 0
        ):
            raise ValueError("n_oscillators must be positive")
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ValueError("protocol_version must be positive")
        if not math.isfinite(self.phase_blend) or not (0.0 <= self.phase_blend <= 1.0):
            raise ValueError("phase_blend must be finite and in [0, 1]")
        if (
            not math.isfinite(self.max_phase_step_rad)
            or self.max_phase_step_rad <= 0.0
            or self.max_phase_step_rad > math.pi
        ):
            raise ValueError("max_phase_step_rad must be finite and in (0, pi]")
        if not math.isfinite(self.peer_timeout_s) or self.peer_timeout_s <= 0.0:
            raise ValueError("peer_timeout_s must be positive and finite")
        _positive_weight(self.local_weight, "local_weight")
        _positive_weight(self.peer_weight, "peer_weight")


@dataclass(frozen=True)
class PhaseSyncMessage:
    """Canonical wire message carrying one node's current phase vector."""

    node_id: str
    sequence: int
    phases: tuple[float, ...]
    wall_time_s: float
    protocol_version: int = 1
    digest: str = ""

    def __post_init__(self) -> None:
        _node_id(self.node_id)
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be positive")
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ValueError("protocol_version must be positive")
        if (
            not isinstance(self.wall_time_s, (int, float))
            or isinstance(self.wall_time_s, bool)
            or not math.isfinite(self.wall_time_s)
            or self.wall_time_s < 0.0
        ):
            raise ValueError("wall_time_s must be finite and non-negative")
        phases = _phase_tuple(self.phases, "phases")
        object.__setattr__(self, "phases", phases)
        digest = self.digest or _message_digest(
            self.node_id,
            self.sequence,
            self.protocol_version,
            self.wall_time_s,
            phases,
        )
        object.__setattr__(self, "digest", digest)
        if digest != _message_digest(
            self.node_id,
            self.sequence,
            self.protocol_version,
            self.wall_time_s,
            phases,
        ):
            raise ValueError("phase sync message digest mismatch")

    @property
    def n_oscillators(self) -> int:
        """Return the phase-vector width."""
        return len(self.phases)

    @classmethod
    def from_phases(
        cls,
        *,
        node_id: str,
        sequence: int,
        phases: FloatArray,
        wall_time_s: float | None = None,
        protocol_version: int = 1,
    ) -> PhaseSyncMessage:
        """Build a canonical message from a local phase vector."""
        timestamp = time.time() if wall_time_s is None else float(wall_time_s)
        return cls(
            node_id=node_id,
            sequence=int(sequence),
            phases=_phase_tuple(phases, "phases"),
            wall_time_s=timestamp,
            protocol_version=int(protocol_version),
        )

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe record."""
        return {
            "kind": _PROTOCOL_KIND,
            "protocol_version": self.protocol_version,
            "node_id": self.node_id,
            "sequence": self.sequence,
            "wall_time_s": self.wall_time_s,
            "n_oscillators": self.n_oscillators,
            "phases": list(self.phases),
            "digest": self.digest,
        }

    def to_wire(self) -> bytes:
        """Encode the message as canonical UTF-8 JSON bytes."""
        return json.dumps(
            self.to_record(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_wire(cls, payload: bytes | str | Mapping[str, Any]) -> PhaseSyncMessage:
        """Decode and validate a canonical wire message."""
        if isinstance(payload, bytes):
            decoded = json.loads(payload.decode("utf-8"))
        elif isinstance(payload, str):
            decoded = json.loads(payload)
        elif isinstance(payload, Mapping):
            decoded = dict(payload)
        else:
            raise ValueError("payload must be bytes, string, or decoded mapping")
        if not isinstance(decoded, Mapping):
            raise ValueError("payload must decode to a mapping")
        if decoded.get("kind") != _PROTOCOL_KIND:
            raise ValueError("phase sync message kind mismatch")
        phases = _phase_tuple(decoded.get("phases"), "phases")
        declared = decoded.get("n_oscillators")
        if not isinstance(declared, int) or declared != len(phases):
            raise ValueError("n_oscillators must match phases length")
        return cls(
            node_id=_string(decoded.get("node_id"), "node_id"),
            sequence=_int(decoded.get("sequence"), "sequence"),
            phases=phases,
            wall_time_s=_number(decoded.get("wall_time_s"), "wall_time_s"),
            protocol_version=_int(
                decoded.get("protocol_version"),
                "protocol_version",
            ),
            digest=_string(decoded.get("digest"), "digest"),
        )


@dataclass(frozen=True)
class GossipIngestResult:
    """Accepted/rejected result from phase sync message ingestion."""

    accepted: bool
    reason: str
    peer_id: str | None = None
    sequence: int | None = None

    def to_audit_record(self) -> dict[str, Any]:
        """Return an audit-safe ingestion record."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "peer_id": self.peer_id,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class LossyGossipReplayResult:
    """Summary of deterministic lossy phase gossip replay."""

    initial_mean_pairwise_error: float
    final_mean_pairwise_error: float
    final_phases: Mapping[str, FloatArray]
    accepted_messages: int
    rejected_messages: int
    rounds: int

    def to_audit_record(self) -> dict[str, Any]:
        """Return a compact JSON-safe replay summary."""
        return {
            "kind": "lossy_phase_gossip_replay",
            "rounds": self.rounds,
            "initial_mean_pairwise_error": self.initial_mean_pairwise_error,
            "final_mean_pairwise_error": self.final_mean_pairwise_error,
            "accepted_messages": self.accepted_messages,
            "rejected_messages": self.rejected_messages,
            "nodes": sorted(self.final_phases),
        }


class PhaseGossipNode:
    """Sequence-checked phase gossip state machine for one UPDE node."""

    def __init__(self, config: DistributedSyncConfig):
        self.config = config
        self._local_sequence = 0
        self._peer_sequences: dict[str, int] = {}
        self._peers: dict[str, PhaseSyncMessage] = {}

    @property
    def peer_sequences(self) -> Mapping[str, int]:
        """Return accepted sequence watermarks by peer."""
        return dict(self._peer_sequences)

    @property
    def peer_count(self) -> int:
        """Return the number of accepted active peer states currently retained."""
        return len(self._peers)

    def observe_local(
        self,
        phases: FloatArray,
        *,
        wall_time_s: float | None = None,
    ) -> PhaseSyncMessage:
        """Create the next outbound phase-sync message for this node."""
        phase_array = _phase_array(phases, self.config.n_oscillators, "phases")
        self._local_sequence += 1
        return PhaseSyncMessage.from_phases(
            node_id=self.config.node_id,
            sequence=self._local_sequence,
            phases=phase_array,
            wall_time_s=wall_time_s,
            protocol_version=self.config.protocol_version,
        )

    def ingest(self, payload: bytes | str | Mapping[str, Any]) -> GossipIngestResult:
        """Validate and retain a peer phase-sync message."""
        try:
            message = PhaseSyncMessage.from_wire(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return GossipIngestResult(False, str(exc))
        if message.protocol_version != self.config.protocol_version:
            return GossipIngestResult(
                False,
                "protocol_version mismatch",
                message.node_id,
                message.sequence,
            )
        if message.node_id == self.config.node_id:
            return GossipIngestResult(
                False,
                "self message ignored",
                message.node_id,
                message.sequence,
            )
        if message.n_oscillators != self.config.n_oscillators:
            return GossipIngestResult(
                False,
                "n_oscillators mismatch",
                message.node_id,
                message.sequence,
            )
        previous = self._peer_sequences.get(message.node_id, 0)
        if message.sequence <= previous:
            return GossipIngestResult(
                False,
                "stale or duplicate sequence",
                message.node_id,
                message.sequence,
            )
        self._peer_sequences[message.node_id] = message.sequence
        self._peers[message.node_id] = message
        return GossipIngestResult(True, "accepted", message.node_id, message.sequence)

    def synchronise(
        self,
        local_phases: FloatArray,
        *,
        now_s: float | None = None,
    ) -> FloatArray:
        """Return phases nudged toward active peer circular means."""
        local = _phase_array(
            local_phases,
            self.config.n_oscillators,
            "local_phases",
        )
        timestamp = time.time() if now_s is None else float(now_s)
        if not math.isfinite(timestamp):
            raise ValueError("now_s must be finite")
        active = [
            message
            for message in self._peers.values()
            if timestamp - message.wall_time_s <= self.config.peer_timeout_s
        ]
        if not active:
            return local.copy()
        peer_stack = np.asarray([message.phases for message in active], dtype=float)
        target = _weighted_circular_mean(
            local,
            peer_stack,
            local_weight=self.config.local_weight,
            peer_weight=self.config.peer_weight,
        )
        delta = np.angle(np.exp(1j * (target - local)))
        bounded = np.clip(
            self.config.phase_blend * delta,
            -self.config.max_phase_step_rad,
            self.config.max_phase_step_rad,
        )
        return cast(FloatArray, np.mod(local + bounded, _TWO_PI))

    def to_audit_record(self) -> dict[str, Any]:
        """Return current node sync state as a JSON-safe audit record."""
        return {
            "kind": "phase_gossip_node",
            "node_id": self.config.node_id,
            "local_sequence": self._local_sequence,
            "peer_sequences": {
                peer: self._peer_sequences[peer]
                for peer in sorted(self._peer_sequences)
            },
            "peer_count": self.peer_count,
        }


def simulate_lossy_phase_gossip(
    initial_phases: Mapping[str, FloatArray],
    *,
    rounds: int,
    config: DistributedSyncConfig,
    drop_edges: set[tuple[str, str]] | None = None,
) -> LossyGossipReplayResult:
    """Replay deterministic all-to-all gossip with caller-declared dropped edges."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not initial_phases:
        raise ValueError("initial_phases must contain at least one node")
    states = {
        node_id: _phase_array(phases, config.n_oscillators, f"{node_id}.phases")
        for node_id, phases in initial_phases.items()
    }
    for node_id in states:
        _node_id(node_id)
    drops = drop_edges or set()
    nodes = {
        node_id: PhaseGossipNode(replace(config, node_id=node_id)) for node_id in states
    }
    accepted = 0
    rejected = 0
    initial_error = _mean_pairwise_phase_error(states)
    for round_index in range(rounds):
        now_s = float(round_index + 1)
        outbound = {
            node_id: nodes[node_id].observe_local(states[node_id], wall_time_s=now_s)
            for node_id in sorted(nodes)
        }
        for source_id, message in outbound.items():
            wire = message.to_wire()
            for target_id, node in nodes.items():
                if target_id == source_id or (source_id, target_id) in drops:
                    continue
                result = node.ingest(wire)
                if result.accepted:
                    accepted += 1
                else:
                    rejected += 1
        states = {
            node_id: nodes[node_id].synchronise(states[node_id], now_s=now_s)
            for node_id in sorted(nodes)
        }
    return LossyGossipReplayResult(
        initial_mean_pairwise_error=initial_error,
        final_mean_pairwise_error=_mean_pairwise_phase_error(states),
        final_phases=states,
        accepted_messages=accepted,
        rejected_messages=rejected,
        rounds=rounds,
    )


def _weighted_circular_mean(
    local: FloatArray,
    peers: FloatArray,
    *,
    local_weight: float,
    peer_weight: float,
) -> FloatArray:
    z = local_weight * np.exp(1j * local)
    z += peer_weight * np.sum(np.exp(1j * peers), axis=0)
    return cast(FloatArray, np.mod(np.angle(z), _TWO_PI))


def _mean_pairwise_phase_error(states: Mapping[str, FloatArray]) -> float:
    node_ids = sorted(states)
    if len(node_ids) < 2:
        return 0.0
    errors = []
    for left_index, left_id in enumerate(node_ids):
        for right_id in node_ids[left_index + 1 :]:
            diff = np.angle(np.exp(1j * (states[left_id] - states[right_id])))
            errors.append(float(np.mean(np.abs(diff))))
    return float(np.mean(errors))


def _message_digest(
    node_id: str,
    sequence: int,
    protocol_version: int,
    wall_time_s: float,
    phases: tuple[float, ...],
) -> str:
    record = {
        "kind": _PROTOCOL_KIND,
        "node_id": node_id,
        "sequence": sequence,
        "protocol_version": protocol_version,
        "wall_time_s": wall_time_s,
        "phases": list(phases),
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _phase_array(value: Any, expected: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (expected,):
        raise ValueError(f"{label} must have shape ({expected},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return cast(FloatArray, np.mod(array.astype(float, copy=True), _TWO_PI))


def _phase_tuple(value: Any, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return tuple(float(v) for v in np.mod(array, _TWO_PI))


def _positive_weight(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")


def _node_id(value: str) -> str:
    text = _string(value, "node_id")
    if not text.replace("_", "-").replace("-", "").isalnum():
        raise ValueError("node_id must contain only letters, numbers, '-' or '_'")
    return text


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)
