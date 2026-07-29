# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Knm builder tests

from __future__ import annotations

import dataclasses
import sys
import types
import warnings

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState


class _ArrayProtocolFailure:
    def __array__(self, *_args, **_kwargs):
        raise TypeError("array protocol failed")


def test_symmetric():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    np.testing.assert_allclose(cs.knm, cs.knm.T, atol=1e-14)


def test_zero_diagonal():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=6, base_strength=1.0, decay_alpha=0.1)
    np.testing.assert_allclose(np.diag(cs.knm), 0.0, atol=1e-15)


def test_coupling_decays_with_distance():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    # K(0,1) > K(0,3) > K(0,7)
    assert cs.knm[0, 1] > cs.knm[0, 3]
    assert cs.knm[0, 3] > cs.knm[0, 7]


def test_non_negative():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=10, base_strength=0.5, decay_alpha=0.5)
    assert np.all(cs.knm >= 0.0)


def test_default_template_name():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    assert cs.active_template == "default"


def test_switch_template():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    alt_knm = np.eye(4) * 0.0 + 0.1
    np.fill_diagonal(alt_knm, 0.0)
    templates = {"alt": alt_knm}
    cs2 = builder.switch_template(cs, "alt", templates)
    assert cs2.active_template == "alt"
    np.testing.assert_allclose(cs2.knm, alt_knm)


def test_switch_template_rejects_self_coupling_diagonal():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    template = np.zeros((4, 4))
    template[1, 1] = 0.2

    with pytest.raises(ValueError, match="self-coupling"):
        builder.switch_template(cs, "bad", {"bad": template})


def test_switch_to_missing_template_raises():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    with pytest.raises(KeyError, match="notfound"):
        builder.switch_template(cs, "notfound", {})


def test_alpha_initialized_to_zero():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=5, base_strength=0.3, decay_alpha=0.2)
    np.testing.assert_allclose(cs.alpha, 0.0)


def test_invalid_rust_build_output_falls_back_to_numpy(monkeypatch):
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    class BadRustBuilder:
        def build(self, n_layers: int, _base_strength: float, _decay_alpha: float):
            knm = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
            alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
            return {"n": n_layers, "knm": knm.ravel(), "alpha": alpha.ravel()}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.PyCouplingBuilder = BadRustBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)

    state = CouplingBuilder().build(n_layers=4, base_strength=0.5, decay_alpha=0.3)

    assert state.knm.shape == (4, 4)
    assert np.all(np.isfinite(state.knm))
    np.testing.assert_allclose(np.diag(state.knm), 0.0, atol=1e-15)
    np.testing.assert_allclose(state.knm, state.knm.T, atol=1e-14)


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        (field, payload)
        for field in ("knm", "alpha")
        for payload in (
            np.zeros(16, dtype=bool),
            [0.0, True, *([0.0] * 14)],
            np.full(16, 0.2j, dtype=np.complex128),
            np.full(16, "0.0", dtype=object),
            np.full(16, "bad", dtype=object),
        )
    ],
)
def test_coercive_rust_build_output_falls_back_without_publication(
    monkeypatch,
    field,
    payload,
):
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    valid_knm = np.full((4, 4), 0.25, dtype=np.float64)
    np.fill_diagonal(valid_knm, 0.0)
    outputs = {"knm": valid_knm.ravel(), "alpha": np.zeros(16)}
    outputs[field] = payload

    class CoerciveRustBuilder:
        def build(self, n_layers: int, _base_strength: float, _decay_alpha: float):
            return {"n": n_layers, **outputs}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.PyCouplingBuilder = CoerciveRustBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state = CouplingBuilder().build(4, 0.5, 0.3)

    assert np.any(state.knm > 0.0)
    np.testing.assert_allclose(np.diag(state.knm), 0.0)


def test_real_numeric_object_rust_output_remains_compatible(monkeypatch):
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    knm = np.full((4, 4), np.float32(0.25), dtype=object)
    np.fill_diagonal(knm, np.int64(0))
    alpha = np.zeros((4, 4), dtype=object)

    class ObjectRustBuilder:
        def build(self, n_layers: int, _base_strength: float, _decay_alpha: float):
            return {"n": n_layers, "knm": knm.ravel(), "alpha": alpha.ravel()}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.PyCouplingBuilder = ObjectRustBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)

    state = CouplingBuilder().build(4, 0.5, 0.3)

    assert state.knm.dtype == np.float64
    np.testing.assert_allclose(state.knm, knm.astype(np.float64))


@pytest.mark.parametrize(
    ("knm", "alpha", "match"),
    [
        (np.zeros(4), np.array([0.0, np.nan, 0.0, 0.0]), "alpha"),
        (np.array([0.0, -0.1, -0.1, 0.0]), np.zeros(4), "non-negative"),
        (np.array([0.0, 0.1, 0.2, 0.0]), np.zeros(4), "symmetric"),
        (np.array([0.1, 0.0, 0.0, 0.0]), np.zeros(4), "diagonal"),
    ],
)
def test_coupling_output_physical_contract_branches(knm, alpha, match):
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    with pytest.raises(ValueError, match=match):
        knm_mod._validate_coupling_output(knm, alpha, n_layers=2)


def test_coupling_output_array_protocol_failure_is_public_value_error():
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    with pytest.raises(ValueError, match="requested shape"):
        knm_mod._validate_coupling_output(
            _ArrayProtocolFailure(), np.zeros(4), n_layers=2
        )


@pytest.mark.parametrize("payload", ["{", '{"matrix": null}', '{"matrix": [1]}'])
def test_handshake_structural_failures(tmp_path, payload):
    path = tmp_path / "handshakes.json"
    path.write_text(payload, encoding="utf-8")
    state = CouplingBuilder().build(2, 0.5, 0.3)

    with pytest.raises(ValueError):
        CouplingBuilder().apply_handshakes(state, path)


def test_coupling_state_frozen():
    cs = CouplingState(knm=np.eye(3), alpha=np.zeros((3, 3)), active_template="default")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.active_template = "other"


class TestKnmPipelineWiring:
    """Pipeline: CouplingBuilder → K_nm → engine → R."""

    def test_built_knm_drives_engine(self):
        """CouplingBuilder.build → K_nm → engine → R∈[0,1].
        Proves builder output feeds the simulation core."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        cs = CouplingBuilder().build(n, 0.5, 0.3)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(200):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
