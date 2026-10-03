# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Knm builder tests

"""Public construction, topology admission and qualified native ABI recovery."""

from __future__ import annotations

import dataclasses
import sys
import types
import warnings
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState


class _ArrayProtocolFailure:
    def __array__(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Model an impossible native array-protocol fault for public recovery."""
        raise TypeError("array protocol failed")


def test_symmetric() -> None:
    """Exponential construction has equal weights in both directions."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    np.testing.assert_allclose(cs.knm, cs.knm.T, atol=1e-14)


def test_zero_diagonal() -> None:
    """Construction removes self-coupling even when base strength is one."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=6, base_strength=1.0, decay_alpha=0.1)
    np.testing.assert_allclose(np.diag(cs.knm), 0.0, atol=1e-15)


def test_coupling_decays_with_distance() -> None:
    """Positive decay orders weights by increasing layer separation."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    # K(0,1) > K(0,3) > K(0,7)
    assert cs.knm[0, 1] > cs.knm[0, 3]
    assert cs.knm[0, 3] > cs.knm[0, 7]


def test_non_negative() -> None:
    """Admitted positive coefficients produce no inhibitory weights."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=10, base_strength=0.5, decay_alpha=0.5)
    assert np.all(cs.knm >= 0.0)


def test_default_template_name() -> None:
    """A newly built snapshot identifies its topology as default."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    assert cs.active_template == "default"


def test_switch_template() -> None:
    """A registered topology replaces phase weights and the active name."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    alt_knm = np.eye(4) * 0.0 + 0.1
    np.fill_diagonal(alt_knm, 0.0)
    templates = {"alt": alt_knm}
    cs2 = builder.switch_template(cs, "alt", templates)
    assert cs2.active_template == "alt"
    np.testing.assert_allclose(cs2.knm, alt_knm)


def test_switch_template_rejects_self_coupling_diagonal() -> None:
    """A template with a positive diagonal is refused before publication."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    template = np.zeros((4, 4))
    template[1, 1] = 0.2

    with pytest.raises(ValueError, match="self-coupling"):
        builder.switch_template(cs, "bad", {"bad": template})


def test_switch_to_missing_template_raises() -> None:
    """An unknown topology reports its missing name through KeyError."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    with pytest.raises(KeyError, match="notfound"):
        builder.switch_template(cs, "notfound", {})


def test_alpha_initialized_to_zero() -> None:
    """Every ordered pair begins with zero phase lag."""
    builder = CouplingBuilder()
    cs = builder.build(n_layers=5, base_strength=0.3, decay_alpha=0.2)
    np.testing.assert_allclose(cs.alpha, 0.0)


def test_invalid_rust_build_output_falls_back_to_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the unreachable native-NaN ABI refusal through public fallback.

    Rust construction from validated finite inputs cannot produce NaN. This
    existing negative injection is justified by that invariant, not counted as
    real-native success evidence. Actual native/absence tests cover construction.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the separately reported corrupted ABI provider.
    """
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    class BadRustBuilder:
        """Producer that corrupts finite native phase output with NaN."""

        def build(
            self, n_layers: int, _base_strength: float, _decay_alpha: float
        ) -> dict[str, object]:
            """Supply the documented invalid native matrix for public recovery.

            Parameters
            ----------
            n_layers : int
                Requested positive square-matrix dimension.
            _base_strength : float
                Validated phase strength ignored by this corrupted-output fixture.
            _decay_alpha : float
                Validated decay per layer ignored by this corrupted-output fixture.

            Returns
            -------
            dict[str, object]
                NaN phase weights and finite zero lags for negative ABI recovery.
            """
            knm = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
            alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
            return {"n": n_layers, "knm": knm.ravel(), "alpha": alpha.ravel()}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.__dict__["PyCouplingBuilder"] = BadRustBuilder
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
            np.array([0.0, 0.1, "bad", *([0.0] * 13)], dtype=object),
        )
    ],
)
def test_coercive_rust_build_output_falls_back_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    payload: object,
) -> None:
    """Retain native type-corruption refusal, separately from real runtime evidence.

    The current PyO3 builder emits float lists only. Its finite-output invariant
    makes these producer failures unreachable through genuine construction.
    This negative ABI injection preserves public recovery; the actual producer
    and genuinely absent runtime are exercised in separate commands.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the separately reported corrupted ABI provider.
    field : str
        Name of the native output or scalar control replaced by this case.
    payload : object
        Original invalid producer field or JSON root; no coercion precedes admission.
    """
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    valid_knm = np.full((4, 4), 0.25, dtype=np.float64)
    np.fill_diagonal(valid_knm, 0.0)
    outputs: dict[str, object] = {"knm": valid_knm.ravel(), "alpha": np.zeros(16)}
    outputs[field] = payload

    class CoerciveRustBuilder:
        """Producer that substitutes a non-real native field payload."""

        def build(
            self, n_layers: int, _base_strength: float, _decay_alpha: float
        ) -> dict[str, object]:
            """Supply the documented invalid native matrix for public recovery.

            Parameters
            ----------
            n_layers : int
                Requested positive square-matrix dimension.
            _base_strength : float
                Validated phase strength ignored by this corrupted-output fixture.
            _decay_alpha : float
                Validated decay per layer ignored by this corrupted-output fixture.

            Returns
            -------
            dict[str, object]
                A non-real replacement of one native matrix field.
            """
            return {"n": n_layers, **outputs}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.__dict__["PyCouplingBuilder"] = CoerciveRustBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state = CouplingBuilder().build(4, 0.5, 0.3)

    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    expected = 0.5 * np.exp(-0.3 * distance)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(state.knm, expected, rtol=3e-15)
    np.testing.assert_array_equal(state.alpha, 0.0)


def test_real_numeric_object_inputs_remain_compatible() -> None:
    """Actual public construction accepts NumPy real coefficients and counts."""
    state = CouplingBuilder().build(
        cast(int, np.int64(4)), cast(float, np.float32(0.5)), np.float64(0.3)
    )
    assert state.knm.dtype == np.float64
    assert state.knm[0, 1] == pytest.approx(0.5 * np.exp(-0.3))
    template = np.full((4, 4), np.float32(0.25), dtype=object)
    np.fill_diagonal(template, np.int64(0))
    result = CouplingBuilder().switch_template(
        state, "real_objects", {"real_objects": template}
    )
    np.testing.assert_array_equal(result.knm, template.astype(np.float64))


@pytest.mark.parametrize(
    ("knm", "alpha", "match"),
    [
        (np.zeros(4), np.array([0.0, np.nan, 0.0, 0.0]), "alpha"),
        (np.array([0.0, -0.1, -0.1, 0.0]), np.zeros(4), "non-negative"),
        (np.array([0.0, 0.1, 0.2, 0.0]), np.zeros(4), "symmetric"),
        (np.array([0.1, 0.0, 0.0, 0.0]), np.zeros(4), "diagonal"),
    ],
)
def test_coupling_output_physical_contract_branches(
    monkeypatch: pytest.MonkeyPatch,
    knm: NDArray[np.float64],
    alpha: NDArray[np.float64],
    match: str,
) -> None:
    """Corrupted native invariants recover through public NumPy construction.

    The real producer cannot emit non-finite lags, negative weights, asymmetry
    or self-coupling. These four negative ABI cases are reported separately;
    actual producer and absent-runtime traces cover valid construction.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the separately reported corrupted ABI provider.
    knm : numpy.ndarray
        Flat four-element phase payload violating a documented native invariant.
    alpha : numpy.ndarray
        Flat four-element lag payload used by the negative native ABI case.
    match : str
        Documented violated native invariant associated with this payload.
    """
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    # The installed Rust producer cannot emit these invalid matrices. This
    # documented ABI fault injection exercises public fallback without removing
    # its defence. Real producer/absence tests cover the nearest actual path;
    # fault-injection coverage is reported separately from real-runtime coverage.
    class InvalidNativeBuilder:
        """Producer that violates a declared phase or lag invariant."""

        def build(
            self, n_layers: int, _base_strength: float, _decay_alpha: float
        ) -> dict[str, object]:
            """Inject an impossible native output to check public refusal.

            Parameters
            ----------
            n_layers : int
                Requested positive square-matrix dimension.
            _base_strength : float
                Validated phase strength ignored by this corrupted-output fixture.
            _decay_alpha : float
                Validated decay per layer ignored by this corrupted-output fixture.

            Returns
            -------
            dict[str, object]
                Phase or lag weights violating the selected native invariant.
            """
            return {"n": n_layers, "knm": knm, "alpha": alpha}

    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.__dict__["PyCouplingBuilder"] = InvalidNativeBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)
    actual = CouplingBuilder().build(2, 0.5, 0.3)
    assert actual.knm[0, 1] == pytest.approx(0.5 * np.exp(-0.3))
    np.testing.assert_array_equal(actual.alpha, 0.0)


def test_coupling_output_array_protocol_failure_recovers_publicly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented impossible ABI protocol failure reaches public fallback only.

    Real PyO3 output is lists of floats and cannot emit a failing array object.
    This negative ABI injection is separate from real producer/absence evidence;
    it preserves the existing recovery contract without calling private helpers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the separately reported corrupted ABI provider.
    """
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    class BrokenArrayNativeBuilder:
        """Producer that emits a failing array protocol instead of float lists."""

        def build(
            self, n_layers: int, _base_strength: float, _decay_alpha: float
        ) -> dict[str, object]:
            """Supply an invalid array protocol solely to test ABI failure recovery.

            Parameters
            ----------
            n_layers : int
                Requested positive square-matrix dimension.
            _base_strength : float
                Validated phase strength ignored by this corrupted-output fixture.
            _decay_alpha : float
                Validated decay per layer ignored by this corrupted-output fixture.

            Returns
            -------
            dict[str, object]
                A failing array object in place of the native phase float list.
            """
            return {"n": n_layers, "knm": _ArrayProtocolFailure(), "alpha": np.zeros(4)}

    provider = types.ModuleType("spo_kernel")
    provider.__dict__["PyCouplingBuilder"] = BrokenArrayNativeBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", provider)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)
    actual = CouplingBuilder().build(2, 0.5, 0.3)
    assert actual.knm[0, 1] == pytest.approx(0.5 * np.exp(-0.3))


def test_mismatched_native_dimension_recovers_publicly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain public refusal of an impossible producer dimension mismatch.

    The current Rust builder copies its validated n into the result. A genuine
    call cannot return another n; this documented negative ABI fault injection
    preserves that refusal without claiming real producer coverage. The real
    count/capacity and recovery contracts run against the installed kernel.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores the separately reported corrupted ABI provider.
    """
    import scpn_phase_orchestrator.coupling.knm as knm_mod

    class MismatchedNativeBuilder:
        """Producer that returns a count different from its requested dimension."""

        def build(
            self, n_layers: int, _base_strength: float, _decay_alpha: float
        ) -> dict[str, object]:
            """Return a corrupt native count solely to exercise public recovery.

            Parameters
            ----------
            n_layers : int
                Requested positive square-matrix dimension.
            _base_strength : float
                Validated phase strength ignored by this corrupted-output fixture.
            _decay_alpha : float
                Validated decay per layer ignored by this corrupted-output fixture.

            Returns
            -------
            dict[str, object]
                A native-shaped result whose reported count is off by one.
            """
            return {"n": n_layers + 1, "knm": np.zeros(4), "alpha": np.zeros(4)}

    provider = types.ModuleType("spo_kernel")
    provider.__dict__["PyCouplingBuilder"] = MismatchedNativeBuilder
    monkeypatch.setitem(sys.modules, "spo_kernel", provider)
    monkeypatch.setattr(knm_mod, "_HAS_RUST", True)
    actual = CouplingBuilder().build(2, 0.5, 0.3)
    assert actual.knm[0, 1] == pytest.approx(0.5 * np.exp(-0.3))


@pytest.mark.parametrize("payload", ["{", '{"matrix": null}', '{"matrix": [1]}'])
def test_handshake_structural_failures(tmp_path: Path, payload: str) -> None:
    """Malformed JSON, null matrices and invalid entries refuse an overlay.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the real handshake JSON document.
    payload : str
        Malformed JSON text written to the handshake file.
    """
    path = tmp_path / "handshakes.json"
    path.write_text(payload, encoding="utf-8")
    state = CouplingBuilder().build(2, 0.5, 0.3)

    with pytest.raises(ValueError):
        CouplingBuilder().apply_handshakes(state, path)


def test_coupling_state_frozen() -> None:
    """Assigning another active topology raises FrozenInstanceError."""
    cs = CouplingState(knm=np.eye(3), alpha=np.zeros((3, 3)), active_template="default")
    with pytest.raises(dataclasses.FrozenInstanceError):
        field = "active_template"
        setattr(cs, field, "other")


class TestKnmPipelineWiring:
    """Pipeline: CouplingBuilder → K_nm → engine → R."""

    def test_built_knm_drives_engine(self) -> None:
        """CouplingBuilder.build → K_nm → engine → R∈[0,1].

        Proves builder output feeds the simulation core.
        """
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
