# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Layer 12 Distributed Gaian Mesh Bridge

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["GaianMeshNode", "PeerState"]


@dataclass
class PeerState:
    """State received from a peer node in the mesh."""
    node_id: str
    R: float
    psi: float
    timestamp: float


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
        host: str = "0.0.0.0",
        port: int = 12000,
        peer_addresses: list[tuple[str, int]] | None = None,
        mesh_coupling_strength: float = 1.0,
        heartbeat_interval_s: float = 0.05,
        peer_timeout_s: float = 1.0,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peer_addresses = peer_addresses or []
        self.mesh_coupling_strength = mesh_coupling_strength
        self.heartbeat_interval_s = heartbeat_interval_s
        self.peer_timeout_s = peer_timeout_s

        self._peers: dict[str, PeerState] = {}
        self._running = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        
        self._local_R = 0.0
        self._local_psi = 0.0

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)

    def start(self) -> None:
        """Start the mesh networking threads."""
        self._running = True
        self._listen_thread.start()
        self._broadcast_thread.start()

    def stop(self) -> None:
        """Stop the mesh networking threads."""
        self._running = False
        self._sock.close()
        self._listen_thread.join(timeout=1.0)
        self._broadcast_thread.join(timeout=1.0)

    def update_local_state(self, R: float, psi: float) -> None:
        """Update the local macro state to be broadcasted to peers."""
        self._local_R = R
        self._local_psi = psi

    def compute_mesh_drive(self) -> tuple[float, float]:
        """Compute the effective external drive (zeta, psi) from the mesh.

        Returns:
            zeta: The magnitude of the mesh mean field.
            psi_target: The phase angle of the mesh mean field.
        """
        now = time.time()
        
        # Filter out stale peers
        active_peers = [
            p for p in self._peers.values()
            if (now - p.timestamp) < self.peer_timeout_s
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
                
                peer_id = msg.get("node_id")
                if peer_id and peer_id != self.node_id:
                    self._peers[peer_id] = PeerState(
                        node_id=peer_id,
                        R=float(msg.get("R", 0.0)),
                        psi=float(msg.get("psi", 0.0)),
                        timestamp=time.time(),
                    )
            except socket.timeout:
                continue
            except Exception:
                continue

    def _broadcast_loop(self) -> None:
        """Background loop to broadcast local state to peers."""
        while self._running:
            if not self.peer_addresses:
                time.sleep(self.heartbeat_interval_s)
                continue
                
            msg = json.dumps({
                "node_id": self.node_id,
                "R": self._local_R,
                "psi": self._local_psi,
            }).encode("utf-8")
            
            for addr in self.peer_addresses:
                try:
                    self._sock.sendto(msg, addr)
                except Exception:
                    pass
                    
            time.sleep(self.heartbeat_interval_s)
