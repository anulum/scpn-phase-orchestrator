# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Batch-1 low-coverage tests (first 20 files)

from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import (
    PeerState,
    _require_phase,
    _validated_peer_addresses,
)
from scpn_phase_orchestrator.apps.queuewaves.config import (
    QueueWavesConfig,
    SecurityConfig,
    ServiceDef,
)
from scpn_phase_orchestrator.apps.queuewaves.server import create_app
from scpn_phase_orchestrator.nn import runtime as nn_runtime
from scpn_phase_orchestrator.nn.functional import (
    coupling_laplacian,
    kuramoto_forward_masked,
    kuramoto_step_masked,
    order_parameter,
    plv,
    saf_loss,
    winfree_forward,
)
from scpn_phase_orchestrator.nn.inverse import _build_windows, _symmetrise_K
from scpn_phase_orchestrator.nn.oim import (
    coloring_energy,
    coloring_violations,
    extract_coloring_soft,
    oim_step,
)
from scpn_phase_orchestrator.nn.supervisor import (
    _json_safe_value,
    _layer_scope_index,
    _metadata_dtype,
    _metadata_shape,
    _non_negative_float_sequence,
    _positive_float,
    _positive_int,
    _prefixed_float_metrics,
    _scheduled_scalar,
)
from scpn_phase_orchestrator.nn.ude import CouplingResidual, UDEKuramotoLayer
from scpn_phase_orchestrator.upde.jax_engine import (
    _validate_array,
    _validate_finite_float,
    _validate_method,
    _validate_positive_float,
    _validate_positive_int,
)


@pytest.mark.parametrize(
    "module_name,runtime_name",
    [
        (
            "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback",
            "scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_fallback",
        ),
        (
            "scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback",
            "scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_grpc_fallback",
        ),
        (
            "scpn_phase_orchestrator.grpc_gen.spo_pb2",
            "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2",
        ),
        (
            "scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc",
            "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc",
        ),
    ],
)
def test_grpc_gen_aliases_resolve_to_runtime_modules(
    module_name: str,
    runtime_name: str,
) -> None:
    public_mod = importlib.import_module(module_name)
    runtime_mod = importlib.import_module(runtime_name)
    assert public_mod.__file__ == runtime_mod.__file__


def test_nn_runtime_info_and_accelerator_gate_cpu_allowed() -> None:
    info = nn_runtime.jax_runtime_info()
    if not info.has_jax:
        with pytest.raises(RuntimeError, match="JAX is required"):
            nn_runtime.require_accelerator()
        return
    if info.has_accelerator:
        assert nn_runtime.require_accelerator().startswith(("gpu", "tpu"))
    else:
        assert nn_runtime.require_accelerator(allow_cpu=True).startswith("cpu")
        with pytest.raises(RuntimeError, match="No JAX GPU/TPU accelerator"):
            nn_runtime.require_accelerator()


def test_jax_engine_validation_boundaries() -> None:
    assert _validate_positive_int(3, name="n") == 3
    assert _validate_positive_float(0.1, name="dt") == 0.1
    assert _validate_finite_float(-0.25, name="zeta") == -0.25
    assert _validate_method("rk4") == "rk4"

    with pytest.raises(ValueError, match="positive integer"):
        _validate_positive_int(True, name="n")
    with pytest.raises(ValueError, match="finite positive real"):
        _validate_positive_float(0.0, name="dt")
    with pytest.raises(ValueError, match="finite real"):
        _validate_finite_float(float("nan"), name="psi")
    with pytest.raises(ValueError, match="unsupported method"):
        _validate_method("heun")

    arr = _validate_array([[1.0, 2.0], [3.0, 4.0]], name="knm", shape=(2, 2))
    assert arr.shape == (2, 2)
    with pytest.raises(ValueError, match="shape must be"):
        _validate_array([1.0, 2.0], name="phases", shape=(3,))


def test_gaian_mesh_validation_and_peer_state_contract() -> None:
    assert _validated_peer_addresses([("127.0.0.1", 9001)]) == [("127.0.0.1", 9001)]
    with pytest.raises(ValueError, match="must contain"):
        _validated_peer_addresses([("127.0.0.1",)])
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        PeerState(node_id="node-a", R=1.1, psi=0.5, timestamp=1.0)
    wrapped = _require_phase(8.0 * float(jnp.pi), field="psi")
    assert wrapped == pytest.approx(0.0)


def test_queuewaves_create_app_production_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=[ServiceDef(name="svc", promql="up", layer="micro")],
        security=SecurityConfig(mode="production", api_key_env="MISSING_QW_KEY"),
    )
    monkeypatch.delenv("MISSING_QW_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_QW_KEY is required"):
        create_app(cfg)


def test_nn_functional_masked_and_winfree_paths() -> None:
    phases = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    omegas = jnp.zeros((3,), dtype=jnp.float32)
    K = jnp.ones((3, 3), dtype=jnp.float32) * 0.1
    mask = jnp.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
    )
    stepped = kuramoto_step_masked(phases, omegas, K, mask, 0.01)
    assert stepped.shape == phases.shape
    final, trajectory = kuramoto_forward_masked(phases, omegas, K, mask, 0.01, 4)
    assert final.shape == phases.shape
    assert trajectory.shape == (4, 3)
    wf_final, wf_traj = winfree_forward(phases, omegas, 0.25, 0.01, 4)
    assert wf_final.shape == phases.shape
    assert wf_traj.shape == (4, 3)
    assert 0.0 <= float(order_parameter(final)) <= 1.0
    k_small = jnp.array([[0.0, 0.2], [0.2, 0.0]], dtype=jnp.float32)
    o_small = jnp.array([0.1, 0.4], dtype=jnp.float32)
    assert jnp.isfinite(saf_loss(k_small, o_small, 0.5))


def test_nn_functional_plv_and_laplacian_shapes() -> None:
    trajectory = jnp.array([[0.0, 0.2], [0.1, 0.3], [0.2, 0.4]], dtype=jnp.float32)
    plv_matrix = plv(trajectory)
    assert plv_matrix.shape == (2, 2)
    k = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    lap = coupling_laplacian(k)
    assert lap.shape == (2, 2)
    assert jnp.isfinite(lap).all()


def test_nn_inverse_window_and_symmetry_helpers() -> None:
    observed = jnp.array(
        [
            [0.0, 0.2],
            [0.1, 0.25],
            [0.2, 0.30],
            [0.3, 0.35],
            [0.4, 0.40],
        ],
        dtype=jnp.float32,
    )
    starts, targets = _build_windows(observed, 2)
    assert starts.shape == (2, 2)
    assert targets.shape == (2, 2, 2)
    K = jnp.array([[1.0, 3.0], [0.5, 2.0]], dtype=jnp.float32)
    sym = _symmetrise_K(K)
    assert float(sym[0, 0]) == 0.0
    assert float(sym[1, 1]) == 0.0
    assert float(sym[0, 1]) == pytest.approx(float(sym[1, 0]))


def test_nn_inverse_window_helper_handles_oversized_window() -> None:
    observed = jnp.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32)
    starts, targets = _build_windows(observed, 8)
    assert starts.shape == (0, 2)
    assert targets.shape == (0, 8, 2)


def test_nn_oim_energy_and_violation_paths() -> None:
    phases = jnp.array([0.0, 2.1, 4.2], dtype=jnp.float32)
    adjacency = jnp.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=jnp.float32
    )
    next_phases = oim_step(phases, adjacency, n_colors=3, dt=0.01)
    assert next_phases.shape == phases.shape
    colours = extract_coloring_soft(next_phases, n_colors=3)
    assert colours.shape == (3,)
    violations = coloring_violations(colours, adjacency)
    assert float(violations) >= 0.0
    energy = coloring_energy(next_phases, adjacency, n_colors=3)
    assert jnp.isfinite(energy)


def test_nn_oim_coloring_violations_zero_for_disconnected_graph() -> None:
    colors = jnp.array([0, 0, 0], dtype=jnp.int32)
    adjacency = jnp.zeros((3, 3), dtype=jnp.float32)
    violations = coloring_violations(colors, adjacency)
    assert float(violations) == 0.0


def test_nn_ude_residual_and_layer_shape_contract() -> None:
    key = jax.random.PRNGKey(7)
    residual = CouplingResidual(hidden=4, key=key)
    value = residual(jnp.array(0.25, dtype=jnp.float32))
    assert jnp.isfinite(value)
    layer = UDEKuramotoLayer(n=3, n_steps=2, dt=0.01, key=jax.random.PRNGKey(9))
    phases = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    final = layer(phases)
    assert final.shape == phases.shape


def test_nn_supervisor_validation_helpers_and_json_paths() -> None:
    assert _positive_int(3, "field") == 3
    assert _positive_float(0.5, "field") == 0.5
    assert _scheduled_scalar(0.2, (0.3, 0.4), 7) == 0.4
    assert _layer_scope_index("layer_5") == 5
    assert _non_negative_float_sequence((0.1, 0.2), "weights") == (0.1, 0.2)
    assert _metadata_shape({"shape": [2, 3]}, "shape") == (2, 3)
    assert str(_metadata_dtype({"dtype": "float32"}, "dtype")) == "float32"
    assert _json_safe_value(float("nan")) == {"nonfinite_float": "nan"}
    assert _prefixed_float_metrics("m", {"x": 1.0, "y": 2}) == {"m_x": 1.0, "m_y": 2.0}

    with pytest.raises(ValueError, match="positive integer"):
        _positive_int(0, "field")
    with pytest.raises(ValueError, match="finite positive scalar"):
        _positive_float(False, "field")
    with pytest.raises(ValueError, match="invalid layer scope"):
        _layer_scope_index("bad_scope")
    with pytest.raises(ValueError, match="must be finite"):
        _prefixed_float_metrics("m", {"x": float("inf")})


def test_nn_supervisor_metadata_dtype_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="supported dtype"):
        _metadata_dtype({"dtype": "definitely_not_a_dtype"}, "dtype")
