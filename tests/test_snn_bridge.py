# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SNN bridge tests

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.snn_bridge import (
    TAU_RC,
    TAU_REF,
    SNNControllerBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from tests.typing_contracts import assert_precise_ndarray_hint

_R = TypeVar("_R")


def _call(fn: Callable[..., _R], *args: object, **kwargs: object) -> _R:
    """Call ``fn`` with arguments that are deliberately outside its types."""
    return fn(*args, **kwargs)


def _mapping(value: object) -> Mapping[str, Any]:
    """Return a manifest field as a mapping, asserting that it is one."""
    assert isinstance(value, Mapping)
    return value


def _items(value: object) -> list[object]:
    """Return a manifest field as a list, asserting a list or tuple."""
    assert isinstance(value, (list, tuple))
    return list(value)


def _rows(value: object) -> list[Mapping[str, Any]]:
    """Return a manifest field as mapping rows, asserting every row's shape."""
    return [_mapping(row) for row in _items(value)]


def _text(value: object) -> str:
    """Return a manifest field as text, asserting that it is a string."""
    assert isinstance(value, str)
    return value


def _make_state(r_values: list[float]) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in r_values],
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="nominal",
    )


def test_default_lif_params() -> None:
    bridge = SNNControllerBridge()
    assert bridge.tau_rc == TAU_RC
    assert bridge.tau_ref == TAU_REF


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("n_neurons", {"n_neurons": 0}),
        ("n_neurons", {"n_neurons": True}),
        ("tau_rc", {"tau_rc": 0.0}),
        ("tau_ref", {"tau_ref": float("nan")}),
    ],
)
def test_constructor_rejects_malformed_config(
    field: str,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=field):
        _call(SNNControllerBridge, **kwargs)


def test_upde_to_current() -> None:
    bridge = SNNControllerBridge()
    state = _make_state([0.5, 0.8, 0.3])
    currents = bridge.upde_state_to_input_current(state, i_scale=2.0)
    np.testing.assert_allclose(currents, [1.0, 1.6, 0.6])


def test_upde_to_current_empty() -> None:
    bridge = SNNControllerBridge()
    state = _make_state([])
    currents = bridge.upde_state_to_input_current(state)
    assert len(currents) == 0


def test_upde_to_current_rejects_invalid_scale_and_state_values() -> None:
    bridge = SNNControllerBridge()
    state = _make_state([0.5])
    with pytest.raises(ValueError, match="i_scale"):
        bridge.upde_state_to_input_current(state, i_scale=0.0)

    bad_state = _make_state([float("nan")])
    with pytest.raises(ValueError, match="layer 0 R"):
        bridge.upde_state_to_input_current(bad_state)


@pytest.mark.parametrize("r_value", [-0.01, 1.01])
def test_upde_to_current_rejects_out_of_bounds_order_parameter(
    r_value: float,
) -> None:
    bridge = SNNControllerBridge()
    with pytest.raises(ValueError, match="layer 0 R"):
        bridge.upde_state_to_input_current(_make_state([r_value]))


def test_public_array_contracts_are_parameterised() -> None:
    for hint in [
        get_type_hints(SNNControllerBridge.upde_state_to_input_current)["return"],
        get_type_hints(SNNControllerBridge.spike_rates_to_actions)["rates"],
        get_type_hints(SNNControllerBridge.lif_rate_estimate)["currents"],
        get_type_hints(SNNControllerBridge.lif_rate_estimate)["return"],
    ]:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def test_spike_rates_above_threshold() -> None:
    bridge = SNNControllerBridge()
    rates = np.array([60.0, 40.0, 80.0])
    actions = bridge.spike_rates_to_actions(rates, [0, 1, 2], threshold_hz=50.0)
    assert len(actions) == 2
    assert actions[0].scope == "layer_0"
    assert actions[1].scope == "layer_2"


def test_spike_rates_below_threshold() -> None:
    bridge = SNNControllerBridge()
    rates = np.array([10.0, 20.0])
    actions = bridge.spike_rates_to_actions(rates, [0, 1], threshold_hz=50.0)
    assert len(actions) == 0


def test_spike_rate_action_value() -> None:
    bridge = SNNControllerBridge()
    rates = np.array([100.0])
    actions = bridge.spike_rates_to_actions(rates, [0], threshold_hz=50.0)
    expected_value = 0.05 * (100.0 - 50.0) / 50.0
    assert abs(actions[0].value - expected_value) < 1e-10


def test_spike_rates_reject_malformed_inputs() -> None:
    bridge = SNNControllerBridge()
    with pytest.raises(ValueError, match="rates"):
        bridge.spike_rates_to_actions(np.array([np.nan]), [0], threshold_hz=50.0)
    with pytest.raises(ValueError, match="rates"):
        bridge.spike_rates_to_actions(np.ones((1, 1)), [0], threshold_hz=50.0)
    with pytest.raises(ValueError, match="rates"):
        bridge.spike_rates_to_actions(np.array([-1.0]), [0], threshold_hz=50.0)
    with pytest.raises(ValueError, match="rates"):
        bridge.spike_rates_to_actions(
            np.array([True, False]), [0, 1], threshold_hz=50.0
        )
    with pytest.raises(ValueError, match="rates"):
        bridge.spike_rates_to_actions(
            np.array([1.0, 2.0 + 0.0j], dtype=object),
            [0, 1],
            threshold_hz=50.0,
        )
    with pytest.raises(ValueError, match="layer_assignments"):
        bridge.spike_rates_to_actions(np.array([60.0]), [True], threshold_hz=50.0)
    with pytest.raises(ValueError, match="layer_assignments"):
        _call(
            bridge.spike_rates_to_actions,
            np.array([60.0]),
            (0,),
            threshold_hz=50.0,
        )
    with pytest.raises(ValueError, match="layer_assignments"):
        bridge.spike_rates_to_actions(np.array([60.0, 70.0]), [0], threshold_hz=50.0)
    with pytest.raises(ValueError, match="threshold_hz"):
        bridge.spike_rates_to_actions(np.array([60.0]), [0], threshold_hz=0.0)
    with pytest.raises(ValueError, match="rates"):
        _call(
            bridge.spike_rates_to_actions,
            [object()],
            [0],
            threshold_hz=50.0,
        )


def test_lif_rate_superthreshold() -> None:
    bridge = SNNControllerBridge()
    currents = np.array([1.5, 2.0, 3.0])
    rates = bridge.lif_rate_estimate(currents)
    assert all(r > 0 for r in rates)
    assert rates[0] < rates[1] < rates[2]


def test_lif_rate_subthreshold() -> None:
    bridge = SNNControllerBridge()
    currents = np.array([0.5, 0.9, 1.0])
    rates = bridge.lif_rate_estimate(currents)
    assert all(r == 0.0 for r in rates)


def test_lif_rate_edge() -> None:
    bridge = SNNControllerBridge()
    currents = np.array([1.01])
    rates = bridge.lif_rate_estimate(currents)
    assert rates[0] > 0


def test_lif_rate_rejects_non_finite_currents() -> None:
    bridge = SNNControllerBridge()
    with pytest.raises(ValueError, match="currents"):
        bridge.lif_rate_estimate(np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="currents"):
        bridge.lif_rate_estimate(np.ones((1, 1)))
    with pytest.raises(ValueError, match="currents"):
        bridge.lif_rate_estimate(np.array([True, False]))
    with pytest.raises(ValueError, match="currents"):
        bridge.lif_rate_estimate(np.array([1.0, 2.0 + 0.0j], dtype=object))


def test_custom_lif_params() -> None:
    bridge = SNNControllerBridge(tau_rc=0.05, tau_ref=0.005)
    currents = np.array([2.0])
    rates = bridge.lif_rate_estimate(currents)
    assert rates[0] > 0


def test_numpy_network_builds() -> None:
    bridge = SNNControllerBridge(n_neurons=50)
    model = bridge.build_numpy_network(n_layers=3, seed=42)
    assert model.input_node.shape == (3,)
    assert model.output_node.shape == (3,)
    assert model.ensemble.n_neurons == 50
    assert model.ensemble.encoders.shape == (50, 3)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"n_layers": 0}, "n_layers"),
        ({"n_layers": True}, "n_layers"),
        ({"n_layers": 3, "seed": True}, "seed"),
        ({"n_layers": 3, "synapse": 0.0}, "synapse"),
        ({"n_layers": 3, "synapse": float("nan")}, "synapse"),
    ],
)
def test_numpy_network_rejects_malformed_config(
    kwargs: dict[str, object],
    field: str,
) -> None:
    bridge = SNNControllerBridge(n_neurons=50)
    with pytest.raises(ValueError, match=field):
        _call(bridge.build_numpy_network, **kwargs)


def test_nengo_network_alias() -> None:
    bridge = SNNControllerBridge(n_neurons=50)
    model = bridge.build_nengo_network(n_layers=3, seed=42)
    assert model.ensemble.n_neurons == 50


def test_lava_import_error() -> None:
    bridge = SNNControllerBridge()
    with contextlib.suppress(ImportError):
        bridge.build_lava_process(n_layers=3)


def test_lava_process_rejects_invalid_layer_count() -> None:
    bridge = SNNControllerBridge()
    with pytest.raises(ValueError, match="n_layers"):
        bridge.build_lava_process(n_layers=0)


def test_lava_process_with_mock() -> None:
    from unittest.mock import MagicMock, patch

    mock_lif_mod = MagicMock()
    mock_lif_cls = MagicMock()
    mock_lif_mod.LIF = mock_lif_cls
    modules = {
        "lava": MagicMock(),
        "lava.proc": MagicMock(),
        "lava.proc.lif": MagicMock(),
        "lava.proc.lif.process": mock_lif_mod,
    }
    bridge = SNNControllerBridge(n_neurons=50)
    with patch.dict("sys.modules", modules):
        bridge.build_lava_process(n_layers=3)
    mock_lif_cls.assert_called_once()


def test_neuromorphic_schedule_manifest_is_deterministic_and_safe() -> None:
    state = _make_state([0.25, 0.75])
    state.cross_layer_alignment[0, 1] = 0.4
    bridge = SNNControllerBridge(n_neurons=32)

    manifest = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )
    repeated = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )

    assert manifest == repeated
    assert manifest["manifest_kind"] == "neuromorphic_schedule_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "simulator_parity_passed"
    assert manifest["target_backends"] == ["lava", "pynn"]
    assert manifest["actuation_permitted"] is False
    assert manifest["hardware_write_permitted"] is False
    assert len(_text(manifest["schedule_sha256"])) == 64
    populations = _rows(manifest["populations"])
    assert populations[0]["lava_process"] == "LIF"
    assert populations[0]["pynn_cell"] == "IF_curr_exp"
    assert populations[1]["input_current"] == 1.5
    assert manifest["projections"] == [
        {
            "source": "layer_0",
            "target": "layer_1",
            "weight": 0.4,
            "delay_ms": 1.0,
            "receptor_type": "excitatory",
        }
    ]
    assert _mapping(manifest["simulator_parity"])["max_abs_rate_error_hz"] == 0.0
    assert manifest["operator_commands"] == [
        "review neuromorphic_schedule_manifest.json",
        "run Lava or PyNN simulator parity before hardware handoff",
    ]
    # NIR-structural export is wired into the manifest and hashed.
    assert len(_text(manifest["nir_sha256"])) == 64
    nir = _mapping(manifest["neuromorphic_ir"])
    assert nir["metadata"]["conformance"] == "structural_subset"
    assert nir["metadata"]["format"] == "scpn-neuromorphic-ir-structural"
    assert [node["id"] for node in nir["nodes"]] == ["layer_0", "layer_1"]
    assert nir["nodes"][0]["type"] == "LIF"
    assert nir["nodes"][0]["tau_membrane_ms"] == pytest.approx(bridge.tau_rc * 1000.0)
    assert nir["edges"] == [
        {
            "type": "Linear",
            "source": "layer_0",
            "target": "layer_1",
            "weight": 0.4,
            "delay_ms": 1.0,
        }
    ]


def test_neuromorphic_schedule_manifest_rejects_invalid_state() -> None:
    bridge = SNNControllerBridge()
    state = UPDEState(
        layers=[LayerState(R=0.4, psi=0.0), LayerState(R=0.6, psi=0.0)],
        cross_layer_alignment=np.ones((1, 2)),
        stability_proxy=0.5,
        regime_id="nominal",
    )

    try:
        bridge.build_neuromorphic_schedule_manifest(state)
    except ValueError as exc:
        assert "cross_layer_alignment shape" in str(exc)
    else:
        raise AssertionError("invalid cross-layer shape must be rejected")


def test_neuromorphic_schedule_manifest_rejects_non_real_alignment_aliases() -> None:
    bridge = SNNControllerBridge()
    state = UPDEState(
        layers=[LayerState(R=0.4, psi=0.0), LayerState(R=0.6, psi=0.0)],
        cross_layer_alignment=np.array([[0.0, True], [False, 0.0]], dtype=object),
        stability_proxy=0.5,
        regime_id="nominal",
    )

    with pytest.raises(ValueError, match="cross_layer_alignment"):
        bridge.build_neuromorphic_schedule_manifest(state)


def test_neuromorphic_schedule_manifest_rejects_unbounded_order_parameter() -> None:
    bridge = SNNControllerBridge()
    state = _make_state([0.4, 1.2])

    with pytest.raises(ValueError, match="layer 1 R"):
        bridge.build_neuromorphic_schedule_manifest(state)


def test_neuromorphic_target_readiness_audit_blocks_missing_preconditions() -> None:
    state = _make_state([0.25, 0.75])
    bridge = SNNControllerBridge(n_neurons=32)
    manifest = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )

    record = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="lava",
        hardware_site="lab_lava_cluster",
    )

    assert record["schema"] == "scpn_neuromorphic_target_readiness_v1"
    assert record["status"] == "blocked"
    assert record["target_backend"] == "lava"
    assert record["hardware_site"] == "lab_lava_cluster"
    assert record["manifest_sha256"] == manifest["schedule_sha256"]
    assert record["blocked_reasons"] == [
        "credentials_not_configured",
        "operator_approval_missing",
        "external_simulator_parity_not_verified",
    ]
    assert record["hardware_write_permitted"] is False
    assert record["actuation_permitted"] is False
    assert len(_text(record["readiness_sha256"])) == 64


def test_neuromorphic_target_readiness_audit_is_ready_not_executed_and_stable() -> None:
    state = _make_state([0.25, 0.75])
    bridge = SNNControllerBridge(n_neurons=32)
    manifest = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )

    record = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="pynn",
        hardware_site="brainscales_review_lane",
        credentials_configured=True,
        operator_approved=True,
        external_simulator_parity_verified=True,
    )
    repeated = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="pynn",
        hardware_site="brainscales_review_lane",
        credentials_configured=True,
        operator_approved=True,
        external_simulator_parity_verified=True,
    )

    assert record == repeated
    assert record["status"] == "ready_not_executed"
    assert record["blocked_reasons"] == []
    assert record["credentials_configured"] is True
    assert record["operator_approved"] is True
    assert record["external_simulator_parity_verified"] is True
    assert record["hardware_write_permitted"] is False
    assert record["actuation_permitted"] is False
    assert record["operator_commands"] == [
        "review neuromorphic_schedule_manifest.json",
        "run target simulator parity outside SPO before hardware handoff",
        "submit neuromorphic hardware job only from an approved operator workflow",
    ]


def test_neuromorphic_target_readiness_audit_rejects_invalid_manifest_or_target() -> (
    None
):
    bridge = SNNControllerBridge(n_neurons=32)
    manifest: dict[str, object] = {"manifest_kind": "quantum_compiler_manifest"}

    with pytest.raises(ValueError, match="neuromorphic_schedule_manifest"):
        bridge.audit_hardware_target_readiness(
            manifest,
            target_backend="lava",
            hardware_site="lab",
        )

    state = _make_state([0.25, 0.75])
    schedule = bridge.build_neuromorphic_schedule_manifest(state)
    with pytest.raises(ValueError, match="target_backend"):
        bridge.audit_hardware_target_readiness(
            schedule,
            target_backend="loihi",
            hardware_site="lab",
        )


def test_neuromorphic_target_readiness_rejects_aliases_in_public_inputs() -> None:
    bridge = SNNControllerBridge(n_neurons=8)
    manifest = bridge.build_neuromorphic_schedule_manifest(_make_state([0.5]))

    with pytest.raises(ValueError, match="boolean"):
        _call(
            bridge.audit_hardware_target_readiness,
            manifest,
            target_backend="lava",
            hardware_site="lab",
            credentials_configured=1,
        )

    malformed = dict(manifest)
    malformed["target_backends"] = ["lava", 1]
    with pytest.raises(ValueError, match="list of strings"):
        bridge.audit_hardware_target_readiness(
            malformed,
            target_backend="lava",
            hardware_site="lab",
        )

    with pytest.raises(ValueError, match="mapping"):
        _call(
            bridge.audit_hardware_target_readiness,
            [],
            target_backend="lava",
            hardware_site="lab",
        )

    for target_backend, hardware_site, message in (
        ("", "lab", "non-empty"),
        ("lava", "bad\nsite", "control"),
    ):
        with pytest.raises(ValueError, match=message):
            bridge.audit_hardware_target_readiness(
                manifest,
                target_backend=target_backend,
                hardware_site=hardware_site,
            )


def test_neuromorphic_target_readiness_records_manifest_safety_regressions() -> None:
    bridge = SNNControllerBridge(n_neurons=8)
    manifest = bridge.build_neuromorphic_schedule_manifest(_make_state([0.5]))
    manifest["status"] = "draft"
    manifest["hardware_write_permitted"] = True
    manifest["actuation_permitted"] = True

    record = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="lava",
        hardware_site="lab",
    )

    assert _items(record["blocked_reasons"])[:3] == [
        "simulator_parity_not_passed",
        "hardware_write_permission_must_remain_false",
        "actuation_permission_must_remain_false",
    ]


def test_neuromorphic_schedule_revalidates_mutated_bridge_state() -> None:
    bridge = SNNControllerBridge(n_neurons=8)
    bridge.n_neurons = 0
    with pytest.raises(ValueError, match="n_neurons"):
        bridge.build_neuromorphic_schedule_manifest(_make_state([0.5]))

    bridge.n_neurons = 8
    bridge.tau_rc = 0.0
    with pytest.raises(ValueError, match="tau_rc"):
        bridge.build_neuromorphic_schedule_manifest(_make_state([0.5]))

    bridge.tau_rc = TAU_RC
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=float("nan"))],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.5,
        regime_id="nominal",
    )
    with pytest.raises(ValueError, match="psi"):
        bridge.build_neuromorphic_schedule_manifest(state)

    with pytest.raises(ValueError, match="seed"):
        bridge.build_numpy_network(1, seed=-1)


class TestSNNPipelineWiring:
    """Pipeline: UPDEState → SNN currents → LIF rates → actions."""

    def test_upde_state_to_snn_roundtrip(self) -> None:
        """UPDEState → upde_state_to_input_current → lif_rate_estimate →
        spike_rates_to_actions. Full neuro-feedback loop."""
        state = _make_state([0.3, 0.5, 0.7, 0.9])
        bridge = SNNControllerBridge(n_neurons=100)

        currents = bridge.upde_state_to_input_current(state)
        assert currents.shape == (4,)
        assert np.all(np.isfinite(currents))

        rates = bridge.lif_rate_estimate(currents)
        assert np.all(rates >= 0.0)

        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        assert isinstance(actions, list)


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestSNNControllerBridge:
    def test_upde_to_current(self) -> None:
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        bridge = SNNControllerBridge(n_neurons=100)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=i * 0.1) for i in range(4)],
            cross_layer_alignment=np.eye(4),
            stability_proxy=0.5,
            regime_id="nominal",
        )
        currents = bridge.upde_state_to_input_current(state)
        assert currents.shape == (4,)
        assert np.all(np.isfinite(currents))
        np.testing.assert_allclose(currents, [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(
            bridge.upde_state_to_input_current(state, i_scale=2.5),
            [1.25, 1.25, 1.25, 1.25],
        )

    def test_spike_rates_to_actions(self) -> None:
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        rates = np.array([10.0, 50.0, 100.0, 200.0])
        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        assert [action.scope for action in actions] == ["layer_2", "layer_3"]
        assert [action.knob for action in actions] == ["K", "K"]
        np.testing.assert_allclose([action.value for action in actions], [0.05, 0.15])
        assert [action.ttl_s for action in actions] == [5.0, 5.0]
        assert actions[0].justification == "SNN group 2: 100.0 Hz"
        assert actions[1].justification == "SNN group 3: 200.0 Hz"

    def test_lif_rate_estimate_monotonic(self) -> None:
        """Higher input current → higher firing rate (LIF model property)."""
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        currents = np.array([0.5, 1.0, 1.5, 2.0])
        rates = bridge.lif_rate_estimate(currents)
        assert rates.shape == (4,)
        assert np.all(rates >= 0.0), "Firing rates must be non-negative"
        # Monotonicity: higher current → higher or equal rate
        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i] - 1e-10, (
                f"LIF rate not monotonic: I={currents[i]:.1f}→{rates[i]:.1f}, "
                f"I={currents[i + 1]:.1f}→{rates[i + 1]:.1f}"
            )

    def test_build_numpy_network_returns_valid_object(self) -> None:
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge

        bridge = SNNControllerBridge()
        net = bridge.build_numpy_network(4)
        assert net is not None
        assert net.n_layers == 4
        assert net.synapse == 0.01
        np.testing.assert_array_equal(net.input_node, np.zeros(4))
        np.testing.assert_array_equal(net.output_node, np.zeros(4))
        assert net.ensemble.n_neurons == bridge.n_neurons
        assert net.ensemble.encoders.shape == (bridge.n_neurons, 4)
        assert set(np.unique(net.ensemble.encoders)) == {-1.0, 1.0}
        assert np.all(net.ensemble.alpha > 0.0)

    def test_snn_bridge_pipeline_wiring(self) -> None:
        """End-to-end: UPDEState → SNN currents → rates → actions.
        Verifies the full adapter pipeline, not just individual methods."""
        from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        bridge = SNNControllerBridge(n_neurons=100)
        state = UPDEState(
            layers=[LayerState(R=0.3, psi=i * 0.5) for i in range(4)],
            cross_layer_alignment=np.eye(4),
            stability_proxy=0.3,
            regime_id="degraded",
        )
        currents = bridge.upde_state_to_input_current(state)
        rates = bridge.lif_rate_estimate(currents)
        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        # Pipeline must produce finite currents, non-negative rates, and ≥0 actions
        assert np.all(np.isfinite(currents))
        assert np.all(rates >= 0.0)
        assert isinstance(actions, list)
        np.testing.assert_allclose(currents, [0.3, 0.3, 0.3, 0.3])
        np.testing.assert_array_equal(rates, np.zeros(4))
        assert actions == []
