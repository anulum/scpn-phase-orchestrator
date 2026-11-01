# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for delayed Kuramoto

"""Cross-backend parity for :class:`DelayedEngine.run`.

Every accelerated backend must reproduce the NumPy reference final phases
within 1e-9 (Rust / Julia / Go) or 1e-6 (Mojo, subprocess text round-trip).
Direct backend adapters validate the state arrays, scalars, and step counts
before any runtime loads.

Required native cases execute their explicit real owners. Deliberately malformed
foreign-response injections are bounded Python refusal diagnostics; genuine
native execution and actual absence have separate numerical/profile contracts.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import TypeAlias, TypeVar, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _delay_go as delay_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _delay_julia as delay_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _delay_mojo as delay_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _delay_validation as delay_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_go import (
    delayed_kuramoto_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_julia import (
    delayed_kuramoto_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_mojo import (
    delayed_kuramoto_run_mojo,
)
from scpn_phase_orchestrator.upde import delay as delay_mod
from scpn_phase_orchestrator.upde.delay import AVAILABLE_BACKENDS, DelayedEngine
from tests.test_delay_julia_absence import run_source_profile

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]
DelayProblem: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
DirectDelayBackend: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        float,
        int,
        int,
    ],
    FloatArray,
]


def _direct_backend_args() -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    float,
    float,
    float,
    int,
    int,
]:
    """Return a minimal valid delayed-Kuramoto direct backend problem."""
    return (
        np.zeros(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        2,
        0.0,
        0.0,
        0.05,
        1,
        5,
    )


def _problem(seed: int, n: int = 6) -> DelayProblem:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.uniform(-1.0, 1.0, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(0.0, 0.3, (n, n))
    return phases, omegas, knm, alpha


def _reference(problem: DelayProblem, delay: int, steps: int) -> FloatArray:
    """Run the explicit NumPy owner from a fresh public batch."""
    phases, omegas, knm, alpha = problem
    return DelayedEngine(len(phases), dt=0.05, delay_steps=delay).run(
        phases, omegas, knm, 0.3, 0.7, alpha, n_steps=steps, backend="python"
    )


def _assert_parity(backend: str, seed: int, atol: float, delay: int = 2) -> None:
    """Compare actual explicit native execution with the public NumPy owner."""
    problem = _problem(seed)
    ref = _reference(problem, delay, 40)
    phases, omegas, knm, alpha = problem
    got = DelayedEngine(len(phases), dt=0.05, delay_steps=delay).run(
        phases, omegas, knm, 0.3, 0.7, alpha, n_steps=40, backend=backend
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=0.0)


def test__delay_validation_helper_is_directly_linked_to_backend_tests() -> None:
    """The exported ingress validator owns all converted buffers and scalar controls."""
    phases = np.array([0.4, 0.0, 1.1, 0.0])[::2]
    omega = np.ones(4)[::2]
    coupling = np.array([0.0, 0.5, 0.2, 0.0])
    lag = np.full(4, 0.01)
    result = delay_validation.validate_delay_backend_inputs(
        phases,
        omega,
        coupling,
        lag,
        2,
        0.2,
        0.3,
        0.01,
        3,
        4,
    )
    for converted, original in zip(
        result[:4], (phases, omega, coupling, lag), strict=True
    ):
        assert converted.dtype == np.float64 and converted.flags.c_contiguous
        np.testing.assert_array_equal(converted, original)
        assert not np.shares_memory(converted, original)
    assert result[4:] == (2, 0.2, 0.3, 0.01, 3, 4)


@pytest.mark.native_runtime
class TestRustParity:
    """Compare genuine Rust batches against the explicit NumPy owner."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_matches_python(self, seed: int) -> None:
        """Reproduce the same seeded delayed trajectory in the required runtime."""
        _assert_parity("rust", seed, atol=1e-9)


@pytest.mark.native_runtime
class TestJuliaParity:
    """Compare genuine Julia batches against the explicit NumPy owner."""

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        """Reproduce the same seeded delayed trajectory in the required runtime."""
        _assert_parity("julia", seed, atol=1e-9)


@pytest.mark.native_runtime
class TestGoParity:
    """Compare genuine Go batches against the explicit NumPy owner."""

    @pytest.mark.parametrize("seed", [0, 13])
    def test_matches_python(self, seed: int) -> None:
        """Reproduce the same seeded delayed trajectory in the required runtime."""
        _assert_parity("go", seed, atol=1e-9)


@pytest.mark.native_runtime
class TestMojoParity:
    """Compare genuine Mojo batches against the explicit NumPy owner."""

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        """Reproduce the same seeded delayed trajectory in the required runtime."""
        _assert_parity("mojo", seed, atol=1e-6)


@pytest.mark.native_runtime
class TestCrossBackendConsistency:
    """Require agreement between all installed numerical owners."""

    def test_all_backends_agree(self) -> None:
        """Compare every explicit owner with a shared delayed reference."""
        problem = _problem(2026, n=8)
        ref = _reference(problem, 3, 60)
        phases, omegas, knm, alpha = problem
        tol = {"rust": 1e-9, "julia": 1e-9, "go": 1e-9, "mojo": 1e-6, "python": 0.0}
        for backend in AVAILABLE_BACKENDS:
            got = DelayedEngine(len(phases), dt=0.05, delay_steps=3).run(
                phases, omegas, knm, 0.3, 0.7, alpha, n_steps=60, backend=backend
            )
            np.testing.assert_allclose(got, ref, atol=tol[backend], rtol=0.0)


class TestDirectBackendBoundaryContracts:
    """Exercise public bridge ingress and isolated response refusals."""

    @pytest.mark.parametrize(
        ("bad_output", "match"),
        [
            (np.zeros(3, dtype=np.float64), "shape"),
            (np.array([0.0, np.nan], dtype=np.float64), "finite"),
            (np.array([0.0, 100.0], dtype=np.float64), "phases"),
            (np.array([0.0, 1.0 + 0j]), "real-valued"),
            (np.array(["0.0", "1.0"]), "numeric-string"),
            (np.array(["a", "b"]), "finite phase vector"),
            (np.array([True, False]), "boolean"),
        ],
    )
    def test_output_validator_rejects_invalid_phase_vector(
        self, bad_output: object, match: str
    ) -> None:
        """Reject malformed, aliased or out-of-range public validator output."""
        with pytest.raises(ValueError, match=match):
            delay_validation.validate_delay_backend_output(bad_output, n=2)

    def test_output_validator_returns_an_owned_copy_of_a_valid_phase_vector(
        self,
    ) -> None:
        """A valid backend vector is returned as a contiguous float64 copy."""
        produced = np.array([0.25, 6.0], dtype=np.float32)
        validated = delay_validation.validate_delay_backend_output(produced, n=2)
        assert validated.dtype == np.float64 and validated.flags.c_contiguous
        np.testing.assert_array_equal(validated, produced.astype(np.float64))
        validated[0] = 1.0
        assert produced[0] == np.float32(0.25)

    @pytest.mark.parametrize(
        "backend",
        [
            delayed_kuramoto_run_go,
            delayed_kuramoto_run_julia,
            delayed_kuramoto_run_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"phases": np.array([True, False])}, "phases"),
            ({"phases": np.array(["0.0", "1.0"])}, "numeric-string"),
            ({"omegas": np.array(["0.0", "1.0"])}, "numeric-string"),
            ({"knm_flat": np.array(["0.0"] * 4)}, "numeric-string"),
            ({"alpha_flat": np.array(["0.0"] * 4)}, "numeric-string"),
            ({"phases": np.array([0.0, np.nan])}, "finite"),
            ({"phases": np.array([0.0, 1.0 + 0j])}, "real-valued"),
            ({"phases": np.array(["a", "b"])}, "finite float array"),
            ({"phases": np.zeros((2, 1))}, "one-dimensional"),
            ({"phases": np.zeros(3)}, "phases length"),
            ({"omegas": np.zeros(3)}, "omegas length"),
            ({"knm_flat": np.zeros(3)}, "knm_flat length"),
            ({"alpha_flat": np.zeros(3)}, "alpha_flat length"),
            ({"n": 0}, "n"),
            ({"n": 1.5}, "n"),
            ({"dt": 0.0}, "dt"),
            ({"delay_steps": 0}, "delay_steps"),
            ({"zeta": np.inf}, "zeta"),
            ({"zeta": None}, "zeta"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self, backend: DirectDelayBackend, kwargs: dict[str, object], match: str
    ) -> None:
        """Refuse invalid arguments at each public direct bridge ingress."""
        base: dict[str, object] = {
            "phases": np.zeros(2),
            "omegas": np.zeros(2),
            "knm_flat": np.zeros(4),
            "alpha_flat": np.zeros(4),
            "n": 2,
            "zeta": 0.0,
            "psi": 0.0,
            "dt": 0.05,
            "delay_steps": 1,
            "n_steps": 5,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            _call(
                backend,
                base["phases"],
                base["omegas"],
                base["knm_flat"],
                base["alpha_flat"],
                base["n"],
                base["zeta"],
                base["psi"],
                base["dt"],
                base["delay_steps"],
                base["n_steps"],
            )

    def test_go_rejects_runtime_output_outside_phase_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refuse an injected invalid Go response before returning to the caller."""

        class FakeGoLib:
            """C-shared Go shim writing an invalid phase vector."""

            def invalid_response(self, *_args: object) -> int:
                out_ptr = ctypes.cast(
                    cast("ctypes._Pointer[ctypes.c_double]", _args[-1]),
                    ctypes.POINTER(ctypes.c_double),
                )
                out_ptr[0] = 0.0
                out_ptr[1] = 100.0
                return 0

        monkeypatch.setattr(
            delay_go,
            "_load_lib",
            lambda: SimpleNamespace(DelayedKuramotoRun=FakeGoLib().invalid_response),
        )

        with pytest.raises(ValueError, match="phases"):
            delayed_kuramoto_run_go(*_direct_backend_args())

    def test_go_missing_library_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A public Go request refuses an actual nonexistent configured artifact."""
        monkeypatch.setattr(delay_go, "_LIB", None)
        monkeypatch.setattr(delay_go, "_LIB_PATH", Path("missing-libdelay.so"))
        with pytest.raises(ImportError, match="libdelay.so not found"):
            delayed_kuramoto_run_go(*_direct_backend_args())

    def test_go_rejects_nonzero_runtime_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface an injected foreign refusal status as a public ValueError."""

        class FakeGoLib:
            """C-shared Go shim reporting a runtime failure."""

            def invalid_response(self, *_args: object) -> int:
                return 17

        monkeypatch.setattr(
            delay_go,
            "_load_lib",
            lambda: SimpleNamespace(DelayedKuramotoRun=FakeGoLib().invalid_response),
        )

        with pytest.raises(ValueError, match="rc=17"):
            delayed_kuramoto_run_go(*_direct_backend_args())

    def test_julia_rejects_runtime_output_outside_phase_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refuse an injected Julia response outside the supported phase range."""

        class FakeJuliaModule:
            """Julia shim returning an invalid phase vector."""

            def delayed_kuramoto_run(self, *_args: object) -> FloatArray:
                return np.array([0.0, 100.0], dtype=np.float64)

        monkeypatch.setattr(delay_julia, "_ensure", lambda: FakeJuliaModule())

        with pytest.raises(ValueError, match="phases"):
            delayed_kuramoto_run_julia(*_direct_backend_args())

    def test_julia_rejects_raw_numeric_string_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject textual Julia output before numerical conversion can hide its type."""

        class FakeJuliaModule:
            """Julia shim returning stringified phase values."""

            def delayed_kuramoto_run(self, *_args: object) -> list[str]:
                return ["0.0", "1.0"]

        monkeypatch.setattr(delay_julia, "_ensure", lambda: FakeJuliaModule())

        with pytest.raises(ValueError, match="numeric-string"):
            delayed_kuramoto_run_julia(*_direct_backend_args())

    @pytest.mark.native_runtime
    def test_julia_ensure_returns_cached_module(self) -> None:
        """Repeated real public Julia batches retain the loaded numerical module."""
        main = import_module("juliacall").Main
        evaluate_raw = main.seval
        assert callable(evaluate_raw)
        evaluate: Callable[[str], object] = evaluate_raw
        identity = str(evaluate("objectid(DelayJL)"))
        for _ in range(2):
            result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
                np.array([0.4, 1.1]),
                np.ones(2),
                np.array([[0.0, 0.5], [0.2, 0.0]]),
                0.2,
                0.3,
                np.full((2, 2), 0.01),
                n_steps=4,
                backend="julia",
            )
            np.testing.assert_allclose(
                result,
                [0.45155943106653923, 1.1289796554407023],
                atol=1e-13,
                rtol=0.0,
            )
            assert str(evaluate("objectid(DelayJL)")) == identity

    @pytest.mark.native_runtime
    def test_julia_missing_side_file_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Real Julia refuses a public request for a missing configured file."""
        monkeypatch.setattr(delay_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(delay_julia, "_JULIA_FILE", Path("missing-delay.jl"))
        with pytest.raises(ImportError, match="julia side-file not found"):
            delayed_kuramoto_run_julia(*_direct_backend_args())

    @pytest.mark.native_runtime
    def test_julia_ensure_includes_side_file(self) -> None:
        """The real public Julia result comes from the declared numerical side-file."""
        result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
            np.array([0.4, 1.1]),
            np.ones(2),
            np.array([[0.0, 0.5], [0.2, 0.0]]),
            0.2,
            0.3,
            np.full((2, 2), 0.01),
            n_steps=4,
            backend="julia",
        )
        np.testing.assert_allclose(
            result,
            [0.45155943106653923, 1.1289796554407023],
            atol=1e-13,
            rtol=0.0,
        )
        main = import_module("juliacall").Main
        evaluate_raw = main.seval
        assert callable(evaluate_raw)
        evaluate: Callable[[str], object] = evaluate_raw
        actual_file = str(
            evaluate("String(first(methods(DelayJL.delayed_kuramoto_run)).file)")
        )
        assert Path(actual_file).resolve() == delay_julia._JULIA_FILE.resolve()

    def test_mojo_rejects_runtime_output_outside_phase_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refuse an injected malformed Mojo phase vector at the public boundary."""

        def fake_run_mojo_executable(
            _exe: Path,
            _stdin: str,
            *,
            runner: object,
        ) -> subprocess.CompletedProcess[str]:
            del runner
            return subprocess.CompletedProcess(
                args=["delay_mojo"],
                returncode=0,
                stdout="0.0\n100.0\n",
                stderr="",
            )

        monkeypatch.setattr(delay_mojo, "_ensure_exe", lambda: Path("delay_mojo"))
        monkeypatch.setattr(delay_mojo, "run_mojo_executable", fake_run_mojo_executable)

        with pytest.raises(ValueError, match="phases"):
            delayed_kuramoto_run_mojo(*_direct_backend_args())

    def test_mojo_missing_executable_raises_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The public Mojo bridge refuses a nonexistent configured executable."""
        monkeypatch.setattr(delay_mojo, "_EXE_PATH", Path("missing-delay-mojo"))
        with pytest.raises(ImportError, match="not built"):
            delayed_kuramoto_run_mojo(*_direct_backend_args())

    @pytest.mark.parametrize(
        ("stdout", "returncode", "match"),
        [
            ("", 7, "Mojo delay exit 7"),
            ("0.0\n", 0, "returned 1 lines"),
            ("0.0\nnot-a-float\n", 0, "finite phases"),
        ],
    )
    def test_mojo_rejects_invalid_process_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        returncode: int,
        match: str,
    ) -> None:
        """Refuse injected process errors, truncation and nonnumeric Mojo output."""

        def fake_run_mojo_executable(
            _exe: Path,
            _stdin: str,
            *,
            runner: object,
        ) -> subprocess.CompletedProcess[str]:
            del runner
            return subprocess.CompletedProcess(
                args=["delay_mojo"],
                returncode=returncode,
                stdout=stdout,
                stderr="process failed",
            )

        monkeypatch.setattr(delay_mojo, "_ensure_exe", lambda: Path("delay_mojo"))
        monkeypatch.setattr(delay_mojo, "run_mojo_executable", fake_run_mojo_executable)

        with pytest.raises(ValueError, match=match):
            delayed_kuramoto_run_mojo(*_direct_backend_args())


class TestBackendLoaderDispatch:
    """Qualify real public compiled calls and genuine resource absence."""

    @pytest.mark.native_runtime
    def test_rust_loader_wraps_spo_kernel(self) -> None:
        """Repeated public calls execute the installed kernel for strided inputs."""
        native = import_module("spo_kernel").delayed_kuramoto_run_rust
        events: list[str] = []

        def observe(_frame: FrameType, event: str, argument: object) -> None:
            """Observe real cached native execution through the public entry point."""
            if argument is native and event in {"c_call", "c_return", "c_exception"}:
                events.append(event)

        phases = np.array([0.4, 0.0, 1.1, 0.0])[::2]
        frequencies = np.ones(4)[::2]
        coupling = np.array([[0.0, 0.5], [0.2, 0.0]])
        alpha = np.full((2, 2), 0.01)
        expected = np.array([0.45155943106653923, 1.1289796554407023])
        assert not phases.flags.c_contiguous
        assert sys.getprofile() is None
        sys.setprofile(observe)
        try:
            first = DelayedEngine(2, dt=0.01, delay_steps=3).run(
                phases,
                frequencies,
                coupling,
                0.2,
                0.3,
                alpha,
                n_steps=4,
                backend="rust",
            )
            second = DelayedEngine(2, dt=0.01, delay_steps=3).run(
                phases,
                frequencies,
                coupling,
                0.2,
                0.3,
                alpha,
                n_steps=4,
                backend="rust",
            )
        finally:
            sys.setprofile(None)
        assert events == ["c_call", "c_return", "c_call", "c_return"]
        np.testing.assert_allclose(first, expected, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(second, expected, rtol=0.0, atol=1e-13)
        assert not np.shares_memory(first, second)
        np.testing.assert_array_equal(phases, [0.4, 1.1])

    @pytest.mark.native_runtime
    def test_julia_loader_returns_direct_bridge(self) -> None:
        """Explicit Julia handles readonly strided input through its actual bridge."""
        phases = np.array([0.4, 0.0, 1.1, 0.0])[::2]
        phases.setflags(write=False)
        result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
            phases,
            np.ones(2),
            np.array([[0.0, 0.5], [0.2, 0.0]]),
            0.2,
            0.3,
            np.full((2, 2), 0.01),
            n_steps=4,
            backend="julia",
        )
        np.testing.assert_allclose(
            result,
            [0.45155943106653923, 1.1289796554407023],
            atol=1e-13,
            rtol=0.0,
        )
        assert not np.shares_memory(result, phases)
        np.testing.assert_array_equal(phases, [0.4, 1.1])

    @pytest.mark.native_runtime
    def test_resolve_backends_falls_back_to_python(self, tmp_path: Path) -> None:
        """Actual no-Rust/no-asset admission selects NumPy despite running Julia."""
        run_source_profile(tmp_path, "missing")


class _ArrayRaises:
    """Object whose array conversion always fails (probes the defensive guards)."""

    def __array__(self, *_args: object, **_kwargs: object) -> FloatArray:
        raise ValueError("no array interface")


class TestDispatchFallthrough:
    """Preserve numerical auto-mode fallback after bounded loader failures."""

    def test_dispatch_skips_failing_active_and_dedups(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A loader-failure diagnostic falls back through the real public model."""
        attempts: list[str] = []

        def unavailable(name: str) -> delay_mod._DelayBackend:
            """Refuse every native load; never supply a successful solver."""
            attempts.append(name)
            raise ImportError("injected loader refusal")

        monkeypatch.setattr(delay_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(delay_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(delay_mod, "_load_backend", unavailable)
        result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
            np.array([0.4, 1.1]),
            np.ones(2),
            np.array([[0.0, 0.5], [0.2, 0.0]]),
            0.2,
            0.3,
            np.full((2, 2), 0.01),
            n_steps=4,
        )
        np.testing.assert_allclose(
            result,
            [0.45155943106653923, 1.1289796554407023],
            atol=1e-13,
            rtol=0.0,
        )
        assert attempts == ["rust", "go"]

    def test_dispatch_returns_none_when_only_python(self) -> None:
        """Explicit NumPy selection produces the scalar delayed reference trajectory."""
        result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
            np.array([0.4, 1.1]),
            np.ones(2),
            np.array([[0.0, 0.5], [0.2, 0.0]]),
            0.2,
            0.3,
            np.full((2, 2), 0.01),
            n_steps=4,
            backend="python",
        )
        np.testing.assert_allclose(
            result,
            [0.45155943106653923, 1.1289796554407023],
            atol=1e-13,
            rtol=0.0,
        )

    def test_dispatch_none_when_all_accelerators_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Native loader exhaustion retains a real NumPy result for legacy auto mode."""

        def unavailable(_name: str) -> delay_mod._DelayBackend:
            """Refuse an accelerator without synthesising numerical success."""
            raise ImportError("injected loader refusal")

        monkeypatch.setattr(delay_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(delay_mod, "AVAILABLE_BACKENDS", ["rust", "go"])
        monkeypatch.setattr(delay_mod, "_load_backend", unavailable)
        result = DelayedEngine(2, dt=0.01, delay_steps=3).run(
            np.array([0.4, 1.1]),
            np.ones(2),
            np.array([[0.0, 0.5], [0.2, 0.0]]),
            0.2,
            0.3,
            np.full((2, 2), 0.01),
            n_steps=4,
        )
        np.testing.assert_allclose(
            result,
            [0.45155943106653923, 1.1289796554407023],
            atol=1e-13,
            rtol=0.0,
        )


class TestBackendOutputContract:
    """A backend that violates the phase-vector contract falls back to NumPy."""

    @pytest.mark.parametrize(
        "bad_output",
        [
            np.zeros(3),  # wrong oscillator count
            np.array([0.0, 100.0]),  # outside [0, 2*pi)
            np.array(["a", "b"]),  # not convertible to float
            np.array([0.0, np.nan]),  # non-finite
        ],
    )
    def test_invalid_backend_output_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, bad_output: object
    ) -> None:
        """Reject an injected bad response and retain the actual stationary solution."""

        def bad_backend(*_a: object, **_k: object) -> object:
            return bad_output

        monkeypatch.setattr(delay_mod, "_dispatch", lambda: bad_backend)
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        out = eng.run(np.array([0.4, 1.1]), np.zeros(2), np.zeros((2, 2)), n_steps=5)
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out < TWO_PI))
        np.testing.assert_array_equal(out, [0.4, 1.1])

    def test_numeric_string_backend_output_falls_back_before_publication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retain real phases when an injected response contains numeric text."""

        def bad_backend(*_a: object, **_k: object) -> list[str]:
            return ["0.2", "0.4"]

        monkeypatch.setattr(delay_mod, "_dispatch", lambda: bad_backend)
        phases = np.array([0.7, 1.3])
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        out = eng.run(phases, np.zeros(2), np.zeros((2, 2)), n_steps=1)
        np.testing.assert_array_equal(out, phases)


def test_public_output_validator_refuses_unconvertible_arrays() -> None:
    """Refuse a failing array protocol without leaking its conversion exception."""
    with pytest.raises(ValueError, match="finite phase vector"):
        delay_validation.validate_delay_backend_output(_ArrayRaises(), n=2)


class TestStateArrayDefensiveGuards:
    """Reject invalid public state arrays and scalar controls."""

    def test_step_rejects_unconvertible_object(self) -> None:
        """Map a failing array conversion to the public input error contract."""
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        with pytest.raises(ValueError, match="finite float array"):
            eng.step(cast(FloatArray, _ArrayRaises()), np.zeros(2), np.zeros((2, 2)))

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.zeros(3), "shape"),
            (np.array([0.0, np.nan]), "finite"),
        ],
    )
    def test_step_rejects_invalid_numeric_state_arrays(
        self, phases: FloatArray, match: str
    ) -> None:
        """Reject complex, nonfinite or misshaped phases before stepping."""
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        with pytest.raises(ValueError, match=match):
            eng.step(phases, np.zeros(2), np.zeros((2, 2)))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"zeta": None}, "zeta"),
            ({"psi": np.inf}, "psi"),
        ],
    )
    def test_step_rejects_invalid_scalar_controls(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Reject nonfinite, aliased and complex external-drive controls."""
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        with pytest.raises(ValueError, match=match):
            _call(
                eng.step,
                np.zeros(2),
                np.zeros(2),
                np.zeros((2, 2)),
                zeta=kwargs.get("zeta", 0.0),
                psi=kwargs.get("psi", 0.0),
            )

    def test_boolean_alias_probe_handles_unconvertible(self) -> None:
        """Unconvertible public measurements are refused before any native load."""
        for owner in (
            delayed_kuramoto_run_go,
            delayed_kuramoto_run_julia,
            delayed_kuramoto_run_mojo,
        ):
            with pytest.raises(ValueError, match="phases must be a finite float array"):
                owner(
                    cast(FloatArray, _ArrayRaises()),
                    np.zeros(2),
                    np.zeros(4),
                    np.zeros(4),
                    2,
                    0.0,
                    0.0,
                    0.01,
                    1,
                    1,
                )

    def test_numeric_string_alias_probes_cover_defensive_edges(self) -> None:
        """Public history rejects text aliases and retains actual numeric snapshots."""
        history = delay_mod.DelayBuffer(2, max_delay_steps=2)
        for value in (
            np.array([1.0, "2.0"], dtype=object),
            np.array([1.0, ""], dtype=object),
            np.array([1.0, "not-a-number"], dtype=object),
        ):
            with pytest.raises(ValueError, match="phases"):
                history.push(cast(FloatArray, value))
            assert history.length == 0
        history.push(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(history.get_delayed(1), [1.0, 2.0])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"phases": np.array(["0.0", "1.0"])}, "numeric-string"),
            ({"omegas": np.array(["0.0", "1.0"])}, "numeric-string"),
            (
                {"knm": np.array([["0.0", "0.0"], ["0.0", "0.0"]])},
                "numeric-string",
            ),
            (
                {"alpha": np.array([["0.0", "0.0"], ["0.0", "0.0"]])},
                "numeric-string",
            ),
        ],
    )
    def test_public_arrays_reject_numeric_string_aliases(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Reject textual aliases in every public step state buffer."""
        base: dict[str, object] = {
            "phases": np.zeros(2),
            "omegas": np.zeros(2),
            "knm": np.zeros((2, 2)),
            "alpha": np.zeros((2, 2)),
        }
        base.update(kwargs)
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        with pytest.raises(ValueError, match=match):
            eng.run(
                cast(FloatArray, base["phases"]),
                cast(FloatArray, base["omegas"]),
                cast(FloatArray, base["knm"]),
                alpha=cast(FloatArray, base["alpha"]),
                n_steps=1,
            )
