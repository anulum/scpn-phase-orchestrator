# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — GaianMesh tests

from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import (
    GaianMeshNode,
    PeerState,
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestGaianMesh:
    def test_single_node_drive(self):
        node = GaianMeshNode("node1", port=_unused_port())
        node.start()
        node.stop()

        zeta, psi = node.compute_mesh_drive()
        assert zeta == 0.0
        assert psi == 0.0

    def test_two_node_communication(self):
        port1 = _unused_port()
        port2 = _unused_port()

        node1 = GaianMeshNode(
            "node1",
            port=port1,
            peer_addresses=[("127.0.0.1", port2)],
            heartbeat_interval_s=0.01,
        )
        node2 = GaianMeshNode(
            "node2",
            port=port2,
            peer_addresses=[("127.0.0.1", port1)],
            heartbeat_interval_s=0.01,
        )

        node1.start()
        node2.start()

        node1.update_local_state(R=0.8, psi=np.pi / 2)
        node2.update_local_state(R=0.6, psi=np.pi)

        time.sleep(0.12)

        zeta1, psi1 = node1.compute_mesh_drive()
        assert abs(zeta1 - 0.6) < 1e-5
        assert abs(psi1 - np.pi) < 1e-5

        zeta2, psi2 = node2.compute_mesh_drive()
        assert abs(zeta2 - 0.8) < 1e-5
        assert abs(psi2 - np.pi / 2) < 1e-5

        node1.stop()
        node2.stop()

    def test_three_node_consensus(self):
        port1 = _unused_port()
        port2 = _unused_port()
        port3 = _unused_port()

        node1 = GaianMeshNode(
            "node1",
            port=port1,
            peer_addresses=[("127.0.0.1", port2), ("127.0.0.1", port3)],
            heartbeat_interval_s=0.01,
        )
        node2 = GaianMeshNode(
            "node2",
            port=port2,
            peer_addresses=[("127.0.0.1", port1), ("127.0.0.1", port3)],
            heartbeat_interval_s=0.01,
        )
        node3 = GaianMeshNode(
            "node3",
            port=port3,
            peer_addresses=[("127.0.0.1", port1), ("127.0.0.1", port2)],
            heartbeat_interval_s=0.01,
        )

        node1.start()
        node2.start()
        node3.start()

        node1.update_local_state(R=0.5, psi=0.0)
        node2.update_local_state(R=0.5, psi=np.pi / 2)
        node3.update_local_state(R=0.5, psi=np.pi)

        time.sleep(0.2)

        zeta1, psi1 = node1.compute_mesh_drive()
        assert abs(zeta1 - np.sqrt(0.125)) < 1e-3
        assert abs(psi1 - 3 * np.pi / 4) < 1e-3

        node1.stop()
        node2.stop()
        node3.stop()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"node_id": ""},
            {"host": ""},
            {"port": 0},
            {"peer_addresses": [("127.0.0.1", 0)]},
            {"peer_addresses": [(127, 12000)]},
            {"peer_addresses": [("127.0.0.1", "bad")]},
            {"mesh_coupling_strength": -0.1},
            {"heartbeat_interval_s": 0.0},
            {"peer_timeout_s": float("nan")},
        ],
    )
    def test_constructor_rejects_malformed_values(self, kwargs: dict[str, object]):
        with pytest.raises(ValueError):
            params: dict[str, object] = {"node_id": "node", "port": _unused_port()}
            params.update(kwargs)
            GaianMeshNode(**params)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "r_value,psi",
        [
            (-0.1, 0.0),
            (1.1, 0.0),
            (np.nan, 0.0),
            (True, 0.0),
            (0.5, np.inf),
            (0.5, True),
        ],
    )
    def test_update_local_state_rejects_invalid_inputs(
        self,
        r_value: object,
        psi: object,
    ) -> None:
        node = GaianMeshNode("local", port=_unused_port())
        with pytest.raises(ValueError):
            node.update_local_state(R=r_value, psi=psi)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("peer_states", "expected_zeta", "expected_psi"),
        [
            ([], 0.0, 0.0),
            (
                [
                    PeerState("stale", 1.0, 0.0, time.time() - 1000.0),
                ],
                0.0,
                0.0,
            ),
            (
                [
                    PeerState("invalid", float("nan"), 0.0, time.time()),
                    PeerState("valid", 0.7, np.pi, time.time()),
                ],
                0.7,
                np.pi,
            ),
            (
                [
                    PeerState("", 1.0, 0.0, time.time()),
                    PeerState("bad_time", 1.0, 0.0, float("nan")),
                    PeerState("good", 0.5, np.pi / 2, time.time()),
                ],
                0.5,
                np.pi / 2,
            ),
        ],
    )
    def test_compute_mesh_drive_filters_stale_or_malformed_peer_states(
        self,
        peer_states: list[PeerState],
        expected_zeta: float,
        expected_psi: float,
    ) -> None:
        node = GaianMeshNode("agg", port=_unused_port())
        now = time.time()
        for peer in peer_states:
            if peer.node_id not in {"stale", "bad_time"}:
                peer.timestamp = now
        node._peers = {f"peer-{index}": peer for index, peer in enumerate(peer_states)}

        zeta, psi = node.compute_mesh_drive()

        assert zeta == pytest.approx(expected_zeta)
        assert psi == pytest.approx(expected_psi)
        node.stop()

    @pytest.mark.parametrize(
        ("invalid_payload",),
        [
            (b"{",),
            (b"not-json",),
            (b"1",),
            (json.dumps({"node_id": 1, "R": "bad", "psi": "bad"}).encode(),),
        ],
    )
    def test_listen_loop_ignores_malformed_packets(
        self,
        invalid_payload: bytes,
    ) -> None:
        rx_port = _unused_port()
        node = GaianMeshNode("rx", port=rx_port)
        node.start()

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(invalid_payload, ("127.0.0.1", rx_port))
        sender.sendto(
            json.dumps({"node_id": "tx", "R": 0.6, "psi": np.pi}).encode(),
            ("127.0.0.1", rx_port),
        )
        sender.close()

        time.sleep(0.2)
        node.stop()

        assert "tx" in node._peers
        assert pytest.approx(0.6) == node._peers["tx"].R
        assert pytest.approx(np.pi) == node._peers["tx"].psi

    def test_start_is_idempotent(self) -> None:
        node = GaianMeshNode("idempotent", port=_unused_port())
        node.start()
        node.start()
        node.stop()
        node.stop()

    def test_context_manager_starts_and_stops(self):
        port = _unused_port()
        with GaianMeshNode("ctx_node", port=port) as node:
            assert node._running is True

        assert node._running is False

    def test_context_manager_releases_socket_on_exception(self):
        port = _unused_port()
        with (
            pytest.raises(RuntimeError, match="simulated failure"),
            GaianMeshNode("ctx_err", port=port),
        ):
            raise RuntimeError("simulated failure")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as release_check:
            release_check.bind(("127.0.0.1", port))
