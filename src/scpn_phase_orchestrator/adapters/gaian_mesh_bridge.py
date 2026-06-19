# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Layer 12 Distributed Gaian Mesh Bridge

"""UDP Gaian mesh bridge for exchanging reduced macroscopic order parameters.

``GaianMeshNode`` validates peer addresses and local order-parameter updates,
then uses background UDP loops to broadcast and receive reduced ``R``/``psi``
heartbeats. Mesh drive computation filters stale or malformed peer state and
returns an external drive proposal only. The bridge exchanges no raw phases,
coupling matrices, credentials, or actuation commands.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_tcp_port,
)

__all__ = ["GaianMeshNode", "PeerState"]


@dataclass
class PeerState:
    """State received from a peer node in the mesh."""

    node_id: str
    R: float
    psi: float
    timestamp: float

    def __post_init__(self) -> None:
        self.node_id = require_non_empty_str(self.node_id, field="node_id")
        if any(ord(char) < 32 for char in self.node_id):
            raise ValueError("node_id must not contain control characters")
        self.R = _require_unit_interval(self.R, field="R")
        self.psi = _require_phase(self.psi, field="psi")
        self.timestamp = _require_finite_real(
            self.timestamp,
            field="timestamp",
            positive=False,
        )


def _require_finite_real(
    value: object,
    *,
    field: str,
    positive: bool,
) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    if not positive and result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _require_unit_interval(value: object, *, field: str) -> float:
    result = _require_finite_real(value, field=field, positive=False)
    if result > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _require_phase(value: object, *, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    return result % (2.0 * np.pi)


def _valid_peer_state(peer: PeerState, *, now: float, timeout_s: float) -> bool:
    if not isinstance(peer.node_id, str) or not peer.node_id:
        return False
    if not isinstance(peer.R, Real) or isinstance(peer.R, bool):
        return False
    if not isinstance(peer.psi, Real) or isinstance(peer.psi, bool):
        return False
    if not isinstance(peer.timestamp, Real) or isinstance(peer.timestamp, bool):
        return False
    return (
        isfinite(float(peer.R))
        and 0.0 <= float(peer.R) <= 1.0
        and isfinite(float(peer.psi))
        and isfinite(float(peer.timestamp))
        and (now - float(peer.timestamp)) < timeout_s
    )


def _decode_peer_state_payload(
    payload: object,
    *,
    local_node_id: str,
    timestamp: float,
) -> PeerState | None:
    if not isinstance(payload, dict):
        return None

    peer_id = payload.get("node_id")
    if not isinstance(peer_id, str) or not peer_id or peer_id == local_node_id:
        return None
    if "R" not in payload or "psi" not in payload:
        return None

    return PeerState(
        node_id=peer_id,
        R=payload["R"],
        psi=payload["psi"],
        timestamp=timestamp,
    )


def _validated_peer_addresses(
    peer_addresses: list[tuple[str, int]] | None,
) -> list[tuple[str, int]]:
    if peer_addresses is None:
        return []
    if not isinstance(peer_addresses, list):
        raise ValueError("peer_addresses must be a list of (host, port) tuples")

    validated: list[tuple[str, int]] = []
    for peer in peer_addresses:
        if not isinstance(peer, tuple) or len(peer) != 2:
            raise ValueError("peer_addresses must contain (host, port) tuples")
        host, port = peer
        validated.append(
            (
                require_non_empty_str(host, field="peer_addresses host"),
                require_tcp_port(port, field="peer_addresses port"),
            )
        )
    return validated


class GaianMeshNode:
    """Distributed Gaian Mesh (Layer 12) Coupling Bridge.

    Allows multiple independent instances of scpn-phase-orchestrator
    running on different servers/machines to 'couple' together over UDP.
    They exchange aggregate Order Parameters (R, Psi) acting as a massive,
    decentralized super-oscillator.

    The peers' macroscopic fields are combined into a resultant vector,
    which is then applied to the local integration engine via the external
    driver parameters `zeta` and `psi`.
    """

    def __init__(
        self,
        node_id: str,
        host: str = "127.0.0.1",
        port: int = 12000,
        peer_addresses: list[tuple[str, int]] | None = None,
        mesh_coupling_strength: float = 1.0,
        heartbeat_interval_s: float = 0.05,
        peer_timeout_s: float = 1.0,
    ):
        self.node_id = require_non_empty_str(node_id, field="node_id")
        self.host = require_non_empty_str(host, field="host")
        self.port = require_tcp_port(port, field="port")
        self.peer_addresses = _validated_peer_addresses(peer_addresses)
        self.mesh_coupling_strength = _require_finite_real(
            mesh_coupling_strength,
            field="mesh_coupling_strength",
            positive=False,
        )
        self.heartbeat_interval_s = _require_finite_real(
            heartbeat_interval_s,
            field="heartbeat_interval_s",
            positive=True,
        )
        self.peer_timeout_s = _require_finite_real(
            peer_timeout_s,
            field="peer_timeout_s",
            positive=True,
        )

        self._peers: dict[str, PeerState] = {}
        self._running = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))

        self._local_R = 0.0
        self._local_psi = 0.0

        self._listen_thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the mesh networking threads.

        Raises
        ------
        RuntimeError
            If the mesh networking threads cannot start.
        """
        if self._running:
            return

        self._running = True
        if self._listen_thread is None:
            self._listen_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True,
            )
            self._broadcast_thread = threading.Thread(
                target=self._broadcast_loop,
                daemon=True,
            )

        listen_thread = self._listen_thread
        broadcast_thread = self._broadcast_thread
        if listen_thread is None or broadcast_thread is None:
            raise RuntimeError("mesh threads were not initialised")

        if listen_thread.is_alive() or broadcast_thread.is_alive():
            return

        listen_thread.start()
        broadcast_thread.start()

    def stop(self) -> None:
        """Stop the mesh networking threads."""
        self._running = False
        if self._sock.fileno() != -1:
            with contextlib.suppress(OSError):
                self._sock.close()

        if self._listen_thread is not None and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1.0)
        if self._broadcast_thread is not None and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=1.0)

    def __enter__(self) -> GaianMeshNode:
        """Start networking threads on ``with GaianMeshNode(...) as node:``."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop networking threads and release the UDP socket on context exit."""
        self.stop()

    def update_local_state(self, R: float, psi: float) -> None:
        """Update the local macro state to be broadcasted to peers.

        Parameters
        ----------
        R : float
            Kuramoto order parameter.
        psi : float
            Mean phase in radians.
        """
        self._local_R = _require_unit_interval(R, field="R")
        self._local_psi = _require_phase(psi, field="psi")

    def compute_mesh_drive(self) -> tuple[float, float]:
        """Compute the effective external drive (zeta, psi) from the mesh.

        Returns
        -------
            zeta: The magnitude of the mesh mean field.
            psi_target: The phase angle of the mesh mean field.
        """
        now = time.time()

        # Filter out stale peers
        active_peers = [
            p
            for p in self._peers.values()
            if _valid_peer_state(p, now=now, timeout_s=self.peer_timeout_s)
        ]

        if not active_peers:
            return 0.0, 0.0

        # Combine peer order parameters into a resultant complex vector
        z_mesh = 0j
        for p in active_peers:
            z_mesh += p.R * np.exp(1j * p.psi)

        z_mesh /= len(active_peers)

        # Multiply by coupling strength
        zeta = self.mesh_coupling_strength * float(np.abs(z_mesh))
        psi_target = float(np.angle(z_mesh))
        if psi_target < 0:
            psi_target += 2 * np.pi

        return zeta, psi_target

    def _listen_loop(self) -> None:
        """Background loop to receive UDP heartbeats from peers."""
        self._sock.settimeout(0.5)
        while self._running:
            try:
                data, addr = self._sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                peer_state = _decode_peer_state_payload(
                    msg,
                    local_node_id=self.node_id,
                    timestamp=time.time(),
                )
                if peer_state is not None:
                    self._peers[peer_state.node_id] = peer_state
            except (
                TimeoutError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                continue

    def _broadcast_loop(self) -> None:
        """Background loop to broadcast local state to peers."""
        while self._running:
            if not self.peer_addresses:
                time.sleep(self.heartbeat_interval_s)
                continue

            msg = json.dumps(
                {
                    "node_id": self.node_id,
                    "R": self._local_R,
                    "psi": self._local_psi,
                }
            ).encode("utf-8")

            for addr in self.peer_addresses:
                with contextlib.suppress(OSError):
                    self._sock.sendto(msg, addr)

            time.sleep(self.heartbeat_interval_s)
