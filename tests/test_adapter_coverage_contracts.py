# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hermetic adapter coverage contracts

"""Exercise defensive adapter boundaries without live external services."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters import (
    fusion_core_bridge,
    hardware_io,
    modbus_tls,
    neurocore_bridge,
    plasma_control_bridge,
    prometheus,
    redis_store,
    snn_bridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


class _UncoercibleArray:
    """Array-like fixture whose NumPy conversion fails at the first boundary."""

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise TypeError("not array coercible")


def _state(r_values: list[float], *, psi: float = 0.0) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=value, psi=psi) for value in r_values],
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="nominal",
    )


@pytest.mark.parametrize(
    "value",
    [_UncoercibleArray(), np.array([True]), np.array(["not-numeric"], dtype=object)],
)
def test_fusion_vector_rejects_every_non_numeric_conversion_boundary(value: object):
    with pytest.raises(ValueError, match="vector"):
        fusion_core_bridge._finite_vector(value, name="vector")


def test_hardware_buffer_rejects_object_dtype_samples():
    buffer = hardware_io.SampleBuffer(capacity=2, n_channels=1)

    with pytest.raises(ValueError, match="numeric"):
        buffer.push(np.array([["not-numeric"]], dtype=object))


@pytest.mark.parametrize(
    ("module", "blocked_import", "flag_name"),
    [
        (hardware_io, "pymodbus.client", "HAS_MODBUS"),
        (modbus_tls, "pymodbus.client", "HAS_PYMODBUS"),
    ],
)
def test_optional_modbus_import_fallbacks_are_hermetic(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    blocked_import: str,
    flag_name: str,
):
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == blocked_import:
            raise ImportError(f"blocked test import: {name}")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "__import__", guarded_import)
        reloaded = importlib.reload(module)
        assert getattr(reloaded, flag_name) is False

    restored = importlib.reload(module)
    assert isinstance(getattr(restored, flag_name), bool)


class _FakeScalarNeuron:
    def __init__(self) -> None:
        self.reset = False

    def step(self, current: float) -> bool:
        return current > 0.0

    def get_state(self) -> dict[str, object]:
        return {"v": -0.06, "refractory": 0}

    def reset_state(self) -> None:
        self.reset = True


def test_neurocore_seed_noise_and_scalar_backend_are_hermetic(
    monkeypatch: pytest.MonkeyPatch,
):
    noisy = neurocore_bridge.NeurocoreBridge(
        n_layers=1,
        neurons_per_layer=2,
        noise_std=0.01,
        seed=7,
        backend="numpy",
    )
    rates = noisy.step(_state([0.8]), n_substeps=1)
    assert rates.shape == (1,)

    monkeypatch.setattr(neurocore_bridge, "HAS_NEUROCORE", True)
    monkeypatch.setattr(
        neurocore_bridge,
        "StochasticLIFNeuron",
        _FakeScalarNeuron,
        raising=False,
    )
    scalar = neurocore_bridge.NeurocoreBridge(
        n_layers=1,
        neurons_per_layer=2,
        backend="scalar",
    )
    scalar.step(_state([0.8]), n_substeps=1)
    assert scalar.get_neuron_states() == [
        {"v": -0.06, "refractory": 0},
        {"v": -0.06, "refractory": 0},
    ]
    scalar.reset()
    assert all(neuron.reset for neuron in scalar._neurons)


@pytest.mark.parametrize(
    "value",
    [_UncoercibleArray(), np.array(["not-numeric"], dtype=object)],
)
def test_plasma_array_rejects_every_non_numeric_conversion_boundary(value: object):
    with pytest.raises(ValueError, match="array"):
        plasma_control_bridge._finite_array(value, name="array")


def test_plasma_tick_result_requires_mapping():
    bridge = plasma_control_bridge.PlasmaControlBridge()

    with pytest.raises(ValueError, match="dict"):
        bridge.import_snapshot([])  # type: ignore[arg-type]


def test_prometheus_defensive_scalar_and_series_boundaries():
    with pytest.raises(ValueError, match="timeout"):
        prometheus.PrometheusAdapter("http://localhost:9090", timeout=object())  # type: ignore[arg-type]

    adapter = prometheus.PrometheusAdapter("http://localhost:9090")
    with pytest.raises(ValueError, match="query"):
        adapter.fetch_metric(1, 0.0, 1.0, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start"):
        adapter.fetch_metric("up", object(), 1.0, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="series"):
        prometheus._range_values([])
    with pytest.raises(ValueError, match="series"):
        prometheus._instant_value([])


def test_redis_import_and_recursive_number_boundaries(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, "redis", SimpleNamespace())
        reloaded = importlib.reload(redis_store)
        assert reloaded._HAS_REDIS is True

    importlib.reload(redis_store)
    monkeypatch.setattr(redis_store, "_HAS_REDIS", False)
    with pytest.raises(RuntimeError, match="not installed"):
        redis_store.RedisStateStore()

    redis_store._require_finite_json_numbers({"safe": [True]})
    with pytest.raises(ValueError, match="finite"):
        redis_store._require_finite_json_numbers({"bad": [float("inf")]})


def test_snn_private_validators_reject_remaining_malformed_aliases():
    with pytest.raises(ValueError, match="non-negative"):
        snn_bridge._require_nonnegative_int(-1, field="index")
    assert snn_bridge._has_non_real_numeric_alias([object()]) is True
    with pytest.raises(ValueError, match="layer_assignments"):
        snn_bridge._validated_layer_assignments((0, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping"):
        snn_bridge._require_mapping([], field="record")
    with pytest.raises(ValueError, match="non-empty"):
        snn_bridge._require_non_empty_text("", field="name")
    with pytest.raises(ValueError, match="control"):
        snn_bridge._require_non_empty_text("bad\nname", field="name")


def test_snn_readiness_rejects_malformed_boolean_and_backend_list():
    bridge = snn_bridge.SNNControllerBridge(n_neurons=8)
    manifest = bridge.build_neuromorphic_schedule_manifest(_state([0.5]))

    with pytest.raises(ValueError, match="boolean"):
        bridge.audit_hardware_target_readiness(
            manifest,
            target_backend="lava",
            hardware_site="lab",
            credentials_configured=1,  # type: ignore[arg-type]
        )

    malformed = dict(manifest)
    malformed["target_backends"] = ["lava", 1]
    with pytest.raises(ValueError, match="list of strings"):
        bridge.audit_hardware_target_readiness(
            malformed,
            target_backend="lava",
            hardware_site="lab",
        )


def test_snn_readiness_records_manifest_safety_regressions():
    bridge = snn_bridge.SNNControllerBridge(n_neurons=8)
    manifest = bridge.build_neuromorphic_schedule_manifest(_state([0.5]))
    manifest["status"] = "draft"
    manifest["hardware_write_permitted"] = True
    manifest["actuation_permitted"] = True

    record = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="lava",
        hardware_site="lab",
    )

    assert record["blocked_reasons"][:3] == [
        "simulator_parity_not_passed",
        "hardware_write_permission_must_remain_false",
        "actuation_permission_must_remain_false",
    ]


def test_snn_schedule_defensive_state_boundaries(monkeypatch: pytest.MonkeyPatch):
    bridge = snn_bridge.SNNControllerBridge(n_neurons=8)
    bridge.n_neurons = 0
    with pytest.raises(ValueError, match="n_neurons"):
        bridge.build_neuromorphic_schedule_manifest(_state([0.5]))

    bridge.n_neurons = 8
    bridge.tau_rc = 0.0
    with pytest.raises(ValueError, match="tau_rc"):
        bridge.build_neuromorphic_schedule_manifest(_state([0.5]))

    bridge.tau_rc = snn_bridge.TAU_RC
    with pytest.raises(ValueError, match="psi"):
        bridge.build_neuromorphic_schedule_manifest(_state([0.5], psi=float("nan")))

    monkeypatch.setattr(
        snn_bridge,
        "_require_finite_array",
        lambda *_args, **_kwargs: np.array([[float("nan")]]),
    )
    with pytest.raises(ValueError, match="cross_layer_alignment"):
        bridge.build_neuromorphic_schedule_manifest(_state([0.5]))
