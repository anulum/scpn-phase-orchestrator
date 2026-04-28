# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Gaian Mesh tests

import time

import numpy as np

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import GaianMeshNode


class TestGaianMesh:
    def test_single_node_drive(self):
        node = GaianMeshNode("node1", port=12001)
        node.start()
        time.sleep(0.1)
        # Without peers, drive should be zero
        zeta, psi = node.compute_mesh_drive()
        assert zeta == 0.0
        assert psi == 0.0
        node.stop()

    def test_two_node_communication(self):
        node1 = GaianMeshNode(
            "node1",
            port=12002,
            peer_addresses=[("127.0.0.1", 12003)],
            heartbeat_interval_s=0.01,
        )
        node2 = GaianMeshNode(
            "node2",
            port=12003,
            peer_addresses=[("127.0.0.1", 12002)],
            heartbeat_interval_s=0.01,
        )

        node1.start()
        node2.start()

        node1.update_local_state(R=0.8, psi=np.pi / 2)
        node2.update_local_state(R=0.6, psi=np.pi)

        # Wait for heartbeats to exchange
        time.sleep(0.15)

        # Node 1 should see Node 2's state
        zeta1, psi1 = node1.compute_mesh_drive()
        # Node 2 state: R=0.6, psi=pi. Resultant vector is just 0.6 * exp(i*pi)
        # Magnitude is 0.6, phase is pi
        assert abs(zeta1 - 0.6) < 1e-5
        assert abs(psi1 - np.pi) < 1e-5

        # Node 2 should see Node 1's state
        zeta2, psi2 = node2.compute_mesh_drive()
        # Node 1 state: R=0.8, psi=pi/2. Resultant vector is 0.8 * exp(i*pi/2)
        assert abs(zeta2 - 0.8) < 1e-5
        assert abs(psi2 - np.pi / 2) < 1e-5

        node1.stop()
        node2.stop()

    def test_three_node_consensus(self):
        node1 = GaianMeshNode(
            "node1",
            port=12004,
            peer_addresses=[("127.0.0.1", 12005), ("127.0.0.1", 12006)],
            heartbeat_interval_s=0.01,
        )
        node2 = GaianMeshNode(
            "node2",
            port=12005,
            peer_addresses=[("127.0.0.1", 12004), ("127.0.0.1", 12006)],
            heartbeat_interval_s=0.01,
        )
        node3 = GaianMeshNode(
            "node3",
            port=12006,
            peer_addresses=[("127.0.0.1", 12004), ("127.0.0.1", 12005)],
            heartbeat_interval_s=0.01,
        )

        node1.start()
        node2.start()
        node3.start()

        # Update all states
        node1.update_local_state(R=0.5, psi=0.0)
        node2.update_local_state(R=0.5, psi=np.pi / 2)
        node3.update_local_state(R=0.5, psi=np.pi)

        time.sleep(0.2)

        # Node 1 sees Node 2 (0.5i) and Node 3 (-0.5)
        # Resultant: (0.5i - 0.5) / 2 = -0.25 + 0.25i
        # Magnitude: sqrt(0.25^2 + 0.25^2) = 0.35355...
        # Phase: 3pi/4
        zeta1, psi1 = node1.compute_mesh_drive()
        assert abs(zeta1 - np.sqrt(0.125)) < 1e-3
        assert abs(psi1 - 3 * np.pi / 4) < 1e-3

        node1.stop()
        node2.stop()
        node3.stop()

    def test_context_manager_starts_and_stops(self):
        """GaianMeshNode used as a context manager must start threads on
        enter and close the socket on exit (T4 resource cleanup)."""
        with GaianMeshNode("ctx_node", port=12020) as node:
            assert node._running is True
            # Socket should be bound and usable while the node is alive.
            time.sleep(0.02)
        # After exit, networking must have been stopped.
        assert node._running is False
        # Second stop() after __exit__ must not raise — idempotent cleanup.
        node.stop()

    def test_context_manager_releases_socket_on_exception(self):
        """An exception in the with-body must still trigger stop() so the
        UDP socket is released."""
        try:
            with GaianMeshNode("ctx_err", port=12021) as node:
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert node._running is False
