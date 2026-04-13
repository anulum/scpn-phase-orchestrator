# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Gaian mesh bridge tests

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import (
    GaianMeshNode,
    PeerState,
)


@pytest.fixture()
def node():
    """Create a mesh node on a random port, stop after test."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    n = GaianMeshNode(node_id="test-node", host="127.0.0.1", port=port)
    yield n
    n._running = False
    n._sock.close()


class TestPeerState:
    def test_fields(self) -> None:
        ps = PeerState(node_id="a", R=0.8, psi=1.5, timestamp=100.0)
        assert ps.node_id == "a"
        assert ps.R == 0.8
        assert ps.psi == 1.5


class TestNodeInit:
    def test_default_values(self, node: GaianMeshNode) -> None:
        assert node.node_id == "test-node"
        assert node.mesh_coupling_strength == 1.0
        assert node.peer_timeout_s == 1.0

    def test_custom_params(self) -> None:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        n = GaianMeshNode(
            node_id="custom",
            host="127.0.0.1",
            port=port,
            mesh_coupling_strength=2.5,
            peer_timeout_s=3.0,
        )
        assert n.mesh_coupling_strength == 2.5
        assert n.peer_timeout_s == 3.0
        n._sock.close()


class TestUpdateLocalState:
    def test_updates(self, node: GaianMeshNode) -> None:
        node.update_local_state(R=0.9, psi=1.23)
        assert node._local_R == 0.9
        assert node._local_psi == 1.23


class TestComputeMeshDrive:
    def test_no_peers_returns_zero(self, node: GaianMeshNode) -> None:
        zeta, psi = node.compute_mesh_drive()
        assert zeta == 0.0
        assert psi == 0.0

    def test_single_peer(self, node: GaianMeshNode) -> None:
        node._peers["p1"] = PeerState(
            node_id="p1", R=0.8, psi=0.5, timestamp=time.time()
        )
        zeta, psi = node.compute_mesh_drive()
        assert zeta == pytest.approx(0.8, abs=0.01)
        assert 0.0 <= psi < 2 * np.pi

    def test_two_aligned_peers(self, node: GaianMeshNode) -> None:
        now = time.time()
        node._peers["p1"] = PeerState(node_id="p1", R=1.0, psi=0.0, timestamp=now)
        node._peers["p2"] = PeerState(node_id="p2", R=1.0, psi=0.0, timestamp=now)
        zeta, psi = node.compute_mesh_drive()
        assert zeta == pytest.approx(1.0, abs=0.01)

    def test_opposing_peers_cancel(self, node: GaianMeshNode) -> None:
        now = time.time()
        node._peers["p1"] = PeerState(node_id="p1", R=1.0, psi=0.0, timestamp=now)
        node._peers["p2"] = PeerState(node_id="p2", R=1.0, psi=np.pi, timestamp=now)
        zeta, _ = node.compute_mesh_drive()
        assert zeta < 0.1  # nearly cancel

    def test_stale_peers_excluded(self, node: GaianMeshNode) -> None:
        node._peers["stale"] = PeerState(
            node_id="stale", R=1.0, psi=0.0, timestamp=time.time() - 100.0
        )
        zeta, psi = node.compute_mesh_drive()
        assert zeta == 0.0

    def test_coupling_strength_scales(self, node: GaianMeshNode) -> None:
        node.mesh_coupling_strength = 3.0
        node._peers["p1"] = PeerState(
            node_id="p1", R=0.5, psi=0.0, timestamp=time.time()
        )
        zeta, _ = node.compute_mesh_drive()
        assert zeta == pytest.approx(1.5, abs=0.01)

    def test_psi_non_negative(self, node: GaianMeshNode) -> None:
        node._peers["p1"] = PeerState(
            node_id="p1", R=1.0, psi=5.0, timestamp=time.time()
        )
        _, psi = node.compute_mesh_drive()
        assert psi >= 0.0


class TestStartStop:
    def test_start_stop_cycle(self, node: GaianMeshNode) -> None:
        node.start()
        assert node._running is True
        node.stop()
        assert node._running is False

    def test_double_stop_safe(self, node: GaianMeshNode) -> None:
        node.start()
        node.stop()
        node.stop()  # should not raise


class TestListenLoop:
    def test_receives_peer_data(self) -> None:
        import socket as sock

        s = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        node = GaianMeshNode(node_id="rx", host="127.0.0.1", port=port)
        node.start()

        sender = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
        msg = json.dumps({"node_id": "tx", "R": 0.75, "psi": 1.2})
        sender.sendto(msg.encode(), ("127.0.0.1", port))
        sender.close()

        time.sleep(0.6)
        node.stop()

        assert "tx" in node._peers
        assert pytest.approx(0.75) == node._peers["tx"].R

    def test_ignores_own_id(self) -> None:
        import socket as sock

        s = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        node = GaianMeshNode(node_id="self-node", host="127.0.0.1", port=port)
        node.start()

        sender = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
        msg = json.dumps({"node_id": "self-node", "R": 0.5, "psi": 0.0})
        sender.sendto(msg.encode(), ("127.0.0.1", port))
        sender.close()

        time.sleep(0.6)
        node.stop()

        assert "self-node" not in node._peers
