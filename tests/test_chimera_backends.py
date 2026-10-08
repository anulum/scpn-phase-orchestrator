# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for chimera detection

"""Cross-backend parity for :func:`local_order_parameter`.

All backends agree with the Python reference within:

* Rust / Julia / Go — 1e-12 (shared f64).
* Mojo — 1e-9 (subprocess text round-trip).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast, get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from benchmarks.chimera_local_order_reference import scalar_local_order
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_julia as chimera_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_mojo as chimera_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_go import (
    local_order_parameter_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_julia import (
    local_order_parameter_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_mojo import (
    local_order_parameter_mojo,
)
from scpn_phase_orchestrator.monitor import chimera as ch_mod
from scpn_phase_orchestrator.monitor.chimera import (
    AVAILABLE_BACKENDS,
    detect_chimera,
    local_order_parameter,
)
from tests.typing_contracts import assert_precise_ndarray_hint

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * np.pi
LocalOrderBackend = Callable[[FloatArray, FloatArray, int], FloatArray]


def test__chimera_validation_helper_is_directly_linked_to_backend_tests() -> None:
    """Original direct APIs refuse mismatched counts before runtime transport."""
    for backend in (
        local_order_parameter_go,
        local_order_parameter_julia,
        local_order_parameter_mojo,
    ):
        with pytest.raises(ValueError, match="phases length"):
            backend(np.zeros(2), np.zeros(9), 3)


def _problem(seed: int, n: int = 16) -> tuple[FloatArray, FloatArray]:
    """Build deterministic finite phases and directed positive non-self coupling."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > 0.3).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    return phases, knm


def test_backend_array_contracts_are_parameterised() -> None:
    """Keep precise float64 array annotations on direct numerical boundaries."""
    functions = (
        local_order_parameter_go,
        local_order_parameter_julia,
        local_order_parameter_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        for key in ("phases", "knm_flat", "return"):
            assert_precise_ndarray_hint(hints[key])
            assert "float64" in str(hints[key])


class TestDirectBackendBoundaryContracts:
    """Direct optional chimera backends validate before runtime loading."""

    def test_validation_alias_helpers_fail_closed_on_array_protocol_failure(
        self,
    ) -> None:
        """Direct public boundaries refuse real ragged numerical sequences."""
        ragged = cast(FloatArray, [[0.0], [0.0, 0.1]])
        for backend in (
            local_order_parameter_go,
            local_order_parameter_julia,
            local_order_parameter_mojo,
        ):
            with pytest.raises(ValueError):
                backend(ragged, np.zeros(4), 2)

    @pytest.mark.parametrize(
        "backend",
        [
            local_order_parameter_go,
            local_order_parameter_julia,
            local_order_parameter_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("phases", "knm_flat", "n", "match"),
        [
            (np.array([True, False]), np.zeros(4), 2, "phases"),
            (np.array([0.0, np.nan]), np.zeros(4), 2, "phases"),
            (np.array([0.0, 1.0], dtype=np.complex128), np.zeros(4), 2, "real-valued"),
            (
                np.array(["bad", "1.0"], dtype=object),
                np.zeros(4),
                2,
                "finite one-dimensional",
            ),
            (
                np.array(["0.0", "1.0"], dtype=object),
                np.zeros(4),
                2,
                "numeric-string",
            ),
            (
                np.array([0.0 + 0.25j, 1.0], dtype=object),
                np.zeros(4),
                2,
                "real-valued",
            ),
            (np.array([[0.0, 1.0]]), np.zeros(4), 2, "one-dimensional"),
            (np.array([0.0, 1.0]), np.zeros(4), True, "n"),
            (np.array([0.0, 1.0]), np.zeros(4), -1, "n"),
            (np.array([0.0]), np.zeros(4), 2, "phases length"),
            (np.array([0.0, 1.0]), np.array([True, False, False, True]), 2, "knm_flat"),
            (np.array([0.0, 1.0]), np.array([0.0, np.inf, 0.0, 0.0]), 2, "knm_flat"),
            (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0 + 0.25j, 1.0, 0.0], dtype=object),
                2,
                "real-valued",
            ),
            (
                np.array([0.0, 1.0]),
                np.array(["0.0", "1.0", "1.0", "0.0"], dtype=object),
                2,
                "numeric-string",
            ),
            (np.array([0.0, 1.0]), np.zeros((2, 2)), 2, "knm_flat"),
            (np.array([0.0, 1.0]), np.zeros(3), 2, "n\\*n"),
            (np.array([0.0, 1.0]), np.eye(2).ravel(), 2, "diagonal"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: LocalOrderBackend,
        phases: FloatArray,
        knm_flat: FloatArray,
        n: object,
        match: str,
    ) -> None:
        """Reject invalid original measurements before optional runtime loading."""
        with pytest.raises(ValueError, match=match):
            backend(phases, knm_flat, cast(int, n))

    @pytest.mark.parametrize(
        ("local_order", "match"),
        [
            (np.array([0.1, np.nan]), "finite"),
            (np.array([0.1, np.bool_(True)], dtype=object), "boolean"),
            (np.array([0.1, 0.2 + 0.1j], dtype=np.complex128), "real-valued"),
            (np.array([0.1, 0.2 + 0.1j], dtype=object), "real-valued"),
            (np.array(["0.1", "0.2"], dtype=object), "numeric-string"),
            (np.array([0.1, 1.2]), "\\[0, 1\\]"),
            (np.array([0.1]), "shape"),
        ],
    )
    def test_output_validation_rejects_nonphysical_local_order(
        self, monkeypatch: pytest.MonkeyPatch, local_order: FloatArray, match: str
    ) -> None:
        """Refuse deliberately invalid outputs solely as negative controls."""

        def invalid_output(_p: FloatArray, _k: FloatArray, _n: int) -> FloatArray:
            """Supply a deliberately invalid return solely as a negative control."""
            return local_order

        monkeypatch.setattr(ch_mod, "_dispatch", lambda backend=None: invalid_output)
        with pytest.raises(ValueError, match=match):
            local_order_parameter(np.zeros(2), np.array([[0.0, 1.0], [1.0, 0.0]]))

    @pytest.mark.parametrize("bad", [np.array([0.25, 1.25]), np.array([0.25])])
    def test_julia_backend_rejects_nonphysical_output_before_return(
        self, monkeypatch: pytest.MonkeyPatch, bad: FloatArray
    ) -> None:
        """Refuse corrupted Julia output solely as a negative boundary control."""

        class _FakeJulia:
            """Supply invalid output solely to exercise the refusal contract."""

            @staticmethod
            def local_order_parameter(
                phases: FloatArray, knm: FloatArray, n: int
            ) -> FloatArray:
                """Return the deliberately invalid Julia boundary-control vector."""
                return bad

        monkeypatch.setattr(chimera_julia, "_ensure", lambda: _FakeJulia())
        phases, knm = _problem(11, n=2)

        with pytest.raises(ValueError, match="\\[0, 1\\]|length"):
            local_order_parameter_julia(phases, knm.ravel(), 2)

    def test_mojo_backend_rejects_nonfinite_output_before_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refuse corrupted Mojo output solely as a negative boundary control."""

        def _fake_run(payload: str, *, expected_count: int, label: str) -> list[float]:
            """Supply nonfinite transport output solely as a negative control."""
            assert expected_count == 2
            assert label == "CHI"
            return [0.25, np.inf]

        monkeypatch.setattr(chimera_mojo, "_run", _fake_run)
        phases, knm = _problem(12, n=2)

        with pytest.raises(ValueError, match="finite"):
            local_order_parameter_mojo(phases, knm.ravel(), 2)

    @pytest.mark.parametrize(
        ("stdout", "status", "match"),
        [
            ("", 0, "expected 2"),
            ("0.25\n", 0, "expected 2"),
            ("0.25\n\n0.75\n", 0, "expected 2"),
            ("0.25\nnot-a-scalar\n", 0, "non-scalar chimera value"),
            ("", 1, "returned exit 1"),
        ],
    )
    def test_mojo_subprocess_stdout_contract_rejects_malformed_local_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        status: int,
        match: str,
    ) -> None:
        """Refuse deliberately corrupted transport only as a negative control."""
        monkeypatch.setattr(chimera_mojo, "_ensure_exe", lambda: Path("chimera_mojo"))
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=status,
                stdout=stdout,
                stderr="negative transport fault control",
            ),
        )
        with pytest.raises(ValueError, match=match):
            local_order_parameter_mojo(np.zeros(2), np.array([0.0, 1.0, 1.0, 0.0]), 2)


@pytest.mark.native_runtime
class TestRustParity:
    """Require the actual Rust owner and compare its output to the scalar oracle."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        """Fail qualification when the required original native owner is unavailable."""
        assert "rust" in AVAILABLE_BACKENDS, "Required rust owner is unavailable"

    @given(
        n=st.integers(min_value=2, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        """Compare the actual named native owner to the independent scalar equation."""
        phases, knm = _problem(seed, n)
        ref = scalar_local_order(phases, knm)
        got = local_order_parameter(phases, knm, backend="rust")
        np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.native_runtime
class TestJuliaParity:
    """Require the actual Julia owner and compare its output to the scalar oracle."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        """Fail qualification when the required original native owner is unavailable."""
        assert "julia" in AVAILABLE_BACKENDS, "Required julia owner is unavailable"

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        """Compare the actual named native owner to the independent scalar equation."""
        phases, knm = _problem(seed)
        ref = scalar_local_order(phases, knm)
        got = local_order_parameter(phases, knm, backend="julia")
        np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.native_runtime
class TestGoParity:
    """Require the actual Go owner and compare its output to the scalar oracle."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        """Fail qualification when the required original native owner is unavailable."""
        assert "go" in AVAILABLE_BACKENDS, "Required go owner is unavailable"

    @given(
        n=st.integers(min_value=2, max_value=30),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        """Compare the actual named native owner to the independent scalar equation."""
        phases, knm = _problem(seed, n)
        ref = scalar_local_order(phases, knm)
        got = local_order_parameter(phases, knm, backend="go")
        np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.native_runtime
class TestMojoParity:
    """Require the actual Mojo owner and compare its output to the scalar oracle."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        """Fail qualification when the required original native owner is unavailable."""
        assert "mojo" in AVAILABLE_BACKENDS, "Required mojo owner is unavailable"

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        """Compare the actual named native owner to the independent scalar equation."""
        phases, knm = _problem(seed)
        ref = scalar_local_order(phases, knm)
        got = local_order_parameter(phases, knm, backend="mojo")
        np.testing.assert_allclose(got, ref, atol=1e-9)


@pytest.mark.native_runtime
class TestCrossBackendConsistency:
    """Compare every available owner on the same directed graph."""

    def test_all_backends_agree(self) -> None:
        """Compare each actual resolved owner against one independent scalar vector."""
        phases, knm = _problem(2026, n=24)
        ref = scalar_local_order(phases, knm)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            got = local_order_parameter(phases, knm, backend=backend)
            np.testing.assert_allclose(
                got,
                ref,
                atol=tolerances[backend],
                err_msg=f"{backend} diverged from python reference",
            )

    def test_detect_chimera_uses_dispatcher(self) -> None:
        """Classify actual default-owner output through the public dispatcher."""
        phases, knm = _problem(3, n=12)
        state = detect_chimera(phases, knm)
        total = (
            len(state.coherent_indices)
            + len(state.incoherent_indices)
            + int(round(state.chimera_index * 12))
        )
        assert total == 12


def test_go_error_code_propagates_as_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected execution failure cannot become a successful local-order vector."""
    from scpn_phase_orchestrator.experimental.accelerators.monitor import _chimera_go

    class FailedRuntime:
        """Represent a failed native call solely for a negative boundary control."""

        def LocalOrderParameterV2(self, *args: object) -> int:
            """Return a nonzero error code without claiming numerical execution."""
            return 2

    monkeypatch.setattr(_chimera_go, "_load_lib", lambda: FailedRuntime())
    with pytest.raises(ValueError, match="LocalOrderParameterV2 rc=2"):
        local_order_parameter_go(np.zeros(2), np.array([0.0, 1.0, 1.0, 0.0]), 2)
