# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for envelope kernels

"""Cross-backend parity for :func:`extract_envelope` and
:func:`envelope_modulation_depth`.

Tolerances: Rust / Julia / Go 1e-12; Mojo 1e-9. Measured
bit-equivalent (0.0) on Rust/Julia/Go and 3.3e-15 on Mojo.
"""

from __future__ import annotations

import math
import subprocess
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _envelope_julia as envelope_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _envelope_mojo as envelope_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_go import (
    envelope_modulation_depth_go,
    extract_envelope_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_julia import (
    envelope_modulation_depth_julia,
    extract_envelope_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_mojo import (
    envelope_modulation_depth_mojo,
    extract_envelope_mojo,
)
from scpn_phase_orchestrator.upde import (
    _envelope_validation as envelope_validation,
)
from scpn_phase_orchestrator.upde import envelope as e_mod
from scpn_phase_orchestrator.upde.envelope import (
    AVAILABLE_BACKENDS,
    envelope_modulation_depth,
    extract_envelope,
)
from tests.typing_contracts import assert_precise_ndarray_hint

ExtractBackend = Callable[[np.ndarray, int], np.ndarray]
ModBackend = Callable[[np.ndarray], float]
DIRECT_EXTRACT_BACKENDS = (
    extract_envelope_go,
    extract_envelope_julia,
    extract_envelope_mojo,
)
DIRECT_MOD_BACKENDS = (
    envelope_modulation_depth_go,
    envelope_modulation_depth_julia,
    envelope_modulation_depth_mojo,
)


def test__envelope_validation_helper_is_directly_linked_to_backend_tests() -> None:
    """The shared validation helper module is exercised directly by these tests."""
    assert callable(envelope_validation.validate_extract_envelope_input)
    assert callable(envelope_validation.validate_extract_envelope_output)
    assert callable(envelope_validation.validate_envelope_modulation_input)
    assert callable(envelope_validation.validate_envelope_modulation_output)


def _force(backend: str) -> str:
    prev = e_mod.ACTIVE_BACKEND
    e_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    e_mod.ACTIVE_BACKEND = prev


def _ref_env(amps: np.ndarray, window: int) -> np.ndarray:
    """Return the NumPy-reference envelope."""
    prev = _force("python")
    try:
        return extract_envelope(amps, window)
    finally:
        _reset(prev)


def _ref_mod(env: np.ndarray) -> float:
    """Return the NumPy-reference modulation depth."""
    prev = _force("python")
    try:
        return envelope_modulation_depth(env)
    finally:
        _reset(prev)


def _oracle_env(values: list[float], window: int) -> list[float]:
    """Return the trailing-window RMS by plain loops, front-padded with the first."""
    rms = [
        math.sqrt(math.fsum(v * v for v in values[i - window + 1 : i + 1]) / window)
        for i in range(window - 1, len(values))
    ]
    return [rms[0]] * (window - 1) + rms


def _amps(seed: int, n: int = 500) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(1.0, 0.3, n))


class _FakeJuliaEnvelopeModule:
    def __init__(
        self,
        *,
        extract_output: object,
        modulation_output: object,
    ) -> None:
        """Create a minimal Julia envelope module stand-in."""
        self.extract_output = extract_output
        self.modulation_output = modulation_output

    def extract_envelope(self, amps: np.ndarray, window: int) -> object:
        """Return the configured extraction payload."""
        assert amps.flags.c_contiguous
        assert window == 2
        return self.extract_output

    def envelope_modulation_depth(self, env: np.ndarray) -> object:
        """Return the configured modulation-depth payload."""
        assert env.flags.c_contiguous
        return self.modulation_output


class _ArrayConversionFailure:
    def __array__(self, dtype: object | None = None) -> np.ndarray:
        """Raise during NumPy coercion to exercise defensive helper paths."""
        raise TypeError("array conversion failed")


class TestDirectBackendBoundaryContracts:
    """Direct Go/Julia/Mojo adapters validate inputs and outputs at their boundary."""

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    @pytest.mark.parametrize(
        ("amps", "window"),
        [
            (np.ones((1, 4)), 2),
            (np.array([True, False]), 2),
            (np.array([1.0 + 1j, 2.0 + 0j]), 2),
            (np.array([1.0, np.nan, 2.0]), 2),
            (np.array([1.0, np.inf, 2.0]), 2),
            (np.ones(4), True),
            (np.ones(4), 0),
        ],
    )
    def test_extract_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ExtractBackend,
        amps: np.ndarray,
        window: object,
    ) -> None:
        """Invalid extraction inputs are refused before any runtime loads."""
        # Loose alias: the parametrised window is deliberately invalid.
        unchecked: Callable[..., object] = backend
        with pytest.raises((TypeError, ValueError)):
            unchecked(amps, window)

    @pytest.mark.parametrize("backend", DIRECT_MOD_BACKENDS)
    @pytest.mark.parametrize(
        "env",
        [
            np.ones((1, 4)),
            np.array([True, False]),
            np.array([1.0 + 1j, 2.0 + 0j]),
            np.array([1.0, np.nan, 2.0]),
            np.array([1.0, np.inf, 2.0]),
        ],
    )
    def test_mod_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ModBackend,
        env: np.ndarray,
    ) -> None:
        """Invalid modulation inputs are refused before any runtime loads."""
        with pytest.raises((TypeError, ValueError)):
            backend(env)

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    @pytest.mark.parametrize(
        ("amps", "window"),
        [
            (np.array(["1.0", "2.0", "3.0"], dtype=object), 2),
            (np.array(["1.0", "2.0", "3.0"], dtype=np.str_), 2),
            (np.ones(4, dtype=np.float64), "2"),
            (np.ones(4, dtype=np.float64), np.str_("2")),
        ],
    )
    def test_extract_numeric_string_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ExtractBackend,
        amps: np.ndarray,
        window: object,
    ) -> None:
        """Numeric-string extraction inputs are refused before any runtime loads."""
        with pytest.raises(ValueError, match="numeric-string"):
            backend(amps, window)  # type: ignore[arg-type]

    @pytest.mark.parametrize("backend", DIRECT_MOD_BACKENDS)
    @pytest.mark.parametrize(
        "env",
        [
            np.array(["0.1", "0.2"], dtype=object),
            np.array(["0.1", "0.2"], dtype=np.str_),
        ],
    )
    def test_mod_numeric_string_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ModBackend,
        env: np.ndarray,
    ) -> None:
        """Numeric-string modulation inputs are refused before any runtime loads."""
        with pytest.raises(ValueError, match="numeric-string"):
            backend(env)

    def test_extract_output_validator_rejects_numeric_string_aliases(self) -> None:
        """The extraction output validator refuses numeric-string aliases."""
        with pytest.raises(ValueError, match="numeric-string"):
            envelope_validation.validate_extract_envelope_output(
                np.array(["0.5", "0.6"], dtype=object),
                n=2,
            )

    def test_mod_output_validator_rejects_numeric_string_aliases(self) -> None:
        """The modulation output validator refuses numeric-string aliases."""
        with pytest.raises(ValueError, match="numeric-string"):
            envelope_validation.validate_envelope_modulation_output("0.25")

    @pytest.mark.parametrize("value", ["", "   ", "not-a-number"])
    def test_numeric_string_helper_ignores_non_numeric_text(self, value: str) -> None:
        """Non-numeric text is not treated as a numeric-string alias."""
        assert not envelope_validation._contains_numeric_string_alias(value)

    def test_numeric_string_helper_tolerates_array_conversion_failure(self) -> None:
        """A value that cannot become an array is not a numeric-string alias."""
        assert not envelope_validation._contains_numeric_string_alias(
            _ArrayConversionFailure()
        )

    def test_vector_validator_preserves_nonnumeric_string_type_error(self) -> None:
        """A non-numeric string vector keeps its type error."""
        with pytest.raises(TypeError, match="amps must be numeric"):
            envelope_validation.validate_extract_envelope_input(
                np.array(["not-a-number"], dtype=object),
                2,
            )

    def test_private_vector_validator_rejects_disallowed_empty_vector(self) -> None:
        """An empty vector is refused where it is not allowed."""
        with pytest.raises(ValueError, match="must not be empty"):
            envelope_validation._as_finite_vector(
                np.array([], dtype=np.float64),
                name="amps",
                allow_empty=False,
            )

    def test_window_validator_preserves_nonnumeric_type_error(self) -> None:
        """A non-numeric window keeps its type error."""
        with pytest.raises(TypeError, match="window must be an integer"):
            envelope_validation.validate_extract_envelope_input(
                np.ones(4, dtype=np.float64),
                2.5,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "value",
        [
            True,
            0.25 + 0.0j,
            np.array([0.1, 0.2], dtype=np.float64),
            float("nan"),
        ],
    )
    def test_mod_output_validator_preserves_non_string_rejections(
        self,
        value: object,
    ) -> None:
        """Non-string invalid modulation outputs are still refused."""
        with pytest.raises((TypeError, ValueError)):
            envelope_validation.validate_envelope_modulation_output(value)

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    def test_extract_empty_returns_empty_without_optional_runtime(
        self,
        backend: ExtractBackend,
    ) -> None:
        """An empty trace returns empty without loading a runtime."""
        result = backend(np.array([], dtype=np.float64), 3)
        assert result.dtype == np.float64
        assert result.shape == (0,)

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    @pytest.mark.parametrize("window", [4, 8])
    def test_extract_global_rms_without_optional_runtime(
        self,
        backend: ExtractBackend,
        window: int,
    ) -> None:
        """A window covering the trace gives the global RMS without a runtime."""
        amps = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        expected = np.full(4, np.sqrt(np.mean(amps * amps)), dtype=np.float64)
        np.testing.assert_allclose(backend(amps, window), expected, atol=0.0)

    @pytest.mark.parametrize("backend", DIRECT_MOD_BACKENDS)
    def test_mod_empty_and_non_positive_return_zero_without_optional_runtime(
        self,
        backend: ModBackend,
    ) -> None:
        """Empty and non-positive envelopes give zero depth without a runtime."""
        assert backend(np.array([], dtype=np.float64)) == 0.0
        assert backend(np.array([-3.0, -1.0], dtype=np.float64)) == 0.0

    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 3, "RMS", "Mojo envelope RMS returned 0 lines, expected 3"),
            (
                "0.5\n0.6\n",
                3,
                "RMS",
                "Mojo envelope RMS returned 2 lines, expected 3",
            ),
            (
                "0.5\n\n0.6\n",
                2,
                "RMS",
                "Mojo envelope RMS returned 3 lines, expected 2",
            ),
            (
                "0.5\n0.6\n0.7\n",
                1,
                "MOD",
                "Mojo envelope MOD returned 3 lines, expected 1",
            ),
            ("not-a-number\n", 1, "MOD", "finite real values"),
            ("inf\n", 1, "MOD", "finite real values"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        """The Mojo runner refuses malformed executable output."""
        monkeypatch.setattr(envelope_mojo_mod, "_ensure_exe", lambda: "envelope")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            envelope_mojo_mod._run(
                "MOD 1 0.5\n",
                expected_count=expected_count,
                label=label,
            )

    def test_julia_extract_validates_raw_output_before_float_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Julia extraction output is validated before float coercion."""
        fake_module = _FakeJuliaEnvelopeModule(
            extract_output=np.array(["0.5", "0.6", "0.7"], dtype=object),
            modulation_output=0.25,
        )
        monkeypatch.setattr(envelope_julia_mod, "_ensure", lambda: fake_module)

        with pytest.raises(ValueError, match="numeric-string"):
            envelope_julia_mod.extract_envelope_julia(
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                2,
            )

    def test_julia_mod_validates_raw_output_before_float_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Julia modulation output is validated before float coercion."""
        fake_module = _FakeJuliaEnvelopeModule(
            extract_output=np.array([0.5, 0.6, 0.7], dtype=np.float64),
            modulation_output="0.25",
        )
        monkeypatch.setattr(envelope_julia_mod, "_ensure", lambda: fake_module)

        with pytest.raises(ValueError, match="numeric-string"):
            envelope_julia_mod.envelope_modulation_depth_julia(
                np.array([0.5, 0.6, 0.7], dtype=np.float64)
            )


class TestRustParity:
    """Rust agrees with the NumPy law."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=30, max_value=800),
        window=st.integers(min_value=2, max_value=25),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, n: int, window: int, seed: int) -> None:
        """Extraction agrees with the NumPy law on seeded traces."""
        # Hypothesis range keeps ``window ≤ n``. The ``window > n``
        # edge case is ill-defined (can't RMS a window you don't
        # have); Rust returns zeros, the other backends return a
        # whole-trace RMS. Parity tests stay in the physically
        # meaningful regime.
        amps = _amps(seed, n)
        ref = _ref_env(amps, window)
        prev = _force("rust")
        try:
            got = extract_envelope(amps, window)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_modulation(self) -> None:
        """Modulation depth agrees with the NumPy law."""
        env = _amps(7)
        ref = _ref_mod(env)
        prev = _force("rust")
        try:
            got = envelope_modulation_depth(env)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    """Julia agrees with the NumPy law."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_extract_and_mod(self, seed: int) -> None:
        """Extraction and modulation depth agree with the NumPy law."""
        amps = _amps(seed)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
        prev = _force("julia")
        try:
            env = extract_envelope(amps, 10)
            mod = envelope_modulation_depth(amps)
        finally:
            _reset(prev)
        np.testing.assert_allclose(env, ref_env, atol=1e-12)
        assert abs(mod - ref_mod) < 1e-12


class TestGoParity:
    """Go agrees with the NumPy law."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, seed: int) -> None:
        """Extraction agrees with the NumPy law on seeded traces."""
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("go")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestMojoParity:
    """Mojo agrees with the NumPy law."""

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_extract(self, seed: int) -> None:
        """Extraction agrees with the NumPy law on seeded traces."""
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("mojo")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)


class TestCrossBackendConsistency:
    """All available owners agree with each other."""

    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        """Every available owner gives the same envelope within tolerance."""
        amps = _amps(2026)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                env = extract_envelope(amps, 10)
                mod = envelope_modulation_depth(amps)
            finally:
                _reset(prev)
            np.testing.assert_allclose(env, ref_env, atol=tolerances[backend])
            assert abs(mod - ref_mod) <= tolerances[backend]


class TestBackendTypingContracts:
    """Backend adapters annotate float64 arrays precisely."""

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (extract_envelope_go, "go:extract"),
            (extract_envelope_julia, "julia:extract"),
            (extract_envelope_mojo, "mojo:extract"),
        ],
    )
    def test_extract_annotations_use_float64_ndarray(
        self, fn: Callable[..., object], label: str
    ) -> None:
        """Extraction adapters annotate float64 arrays."""
        hints = get_type_hints(fn)
        for name in ("amps", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (envelope_modulation_depth_go, "go:mod"),
            (envelope_modulation_depth_julia, "julia:mod"),
            (envelope_modulation_depth_mojo, "mojo:mod"),
        ],
    )
    def test_mod_annotations_use_float64_ndarray(
        self, fn: Callable[..., object], label: str
    ) -> None:
        """Modulation adapters annotate float64 arrays."""
        hints = get_type_hints(fn)
        text = str(hints["env"])
        assert_precise_ndarray_hint(
            hints["env"],
            context=f"{label}:env",
        )
        assert "numpy.float64" in text, f"{label}:env missing float64 annotation"


class TestBackendLoaderContracts:
    """Loader behaviour of the Julia and Rust envelope owners."""

    def test_julia_loader_returns_cached_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Julia loader returns its cached module."""
        cached = object()
        monkeypatch.setattr(envelope_julia_mod, "_JULIA_MODULE", cached)

        assert envelope_julia_mod._ensure() is cached

    def test_julia_loader_rejects_missing_side_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The Julia loader refuses a missing side file."""
        monkeypatch.setattr(envelope_julia_mod, "_JULIA_MODULE", None)
        monkeypatch.setattr(
            envelope_julia_mod,
            "require_julia_main",
            lambda: types.SimpleNamespace(),
        )
        monkeypatch.setattr(envelope_julia_mod, "_JULIA_FILE", tmp_path / "missing.jl")

        with pytest.raises(ImportError, match="julia side-file not found"):
            envelope_julia_mod._ensure()

    def test_julia_loader_includes_side_file_and_caches_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The Julia loader includes its side file once and caches the module."""
        module = object()
        included: list[str] = []
        julia_file = tmp_path / "envelope.jl"
        julia_file.write_text("module EnvelopeJL\nend\n", encoding="utf-8")
        fake_main = types.SimpleNamespace(
            EnvelopeJL=module,
            include=lambda path: included.append(path),
        )
        monkeypatch.setattr(envelope_julia_mod, "_JULIA_MODULE", None)
        monkeypatch.setattr(
            envelope_julia_mod,
            "require_julia_main",
            lambda: fake_main,
        )
        monkeypatch.setattr(envelope_julia_mod, "_JULIA_FILE", julia_file)

        assert envelope_julia_mod._ensure() is module
        assert included == [str(julia_file)]
        assert envelope_julia_mod._JULIA_MODULE is module

    def test_rust_loader_flattens_inputs_and_returns_float64(self) -> None:
        """The compiled Rust loader ravels 2-D and strided input to float64."""
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")
        loaded = e_mod._load_rust_fns()
        extract = cast("ExtractBackend", loaded["extract"])
        mod = cast("ModBackend", loaded["mod"])
        amps = np.arange(1.0, 13.0).reshape(3, 4)
        strided = np.arange(1.0, 25.0)[::2]
        assert not strided.flags.c_contiguous

        for values in (amps, strided):
            got = extract(values, 3)
            assert got.dtype == np.float64
            assert got.shape == (values.size,)
            np.testing.assert_allclose(
                got, _oracle_env(values.ravel().tolist(), 3), rtol=1e-12, atol=0.0
            )
        depth = mod(np.array([[0.5, 0.75], [1.0, 1.25]]))
        assert isinstance(depth, float)
        assert depth == pytest.approx((1.25 - 0.5) / (1.25 + 0.5), rel=1e-15)


class TestDispatchFallbackChain:
    """Automatic dispatch falls through failing owners."""

    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing active owner hands automatic dispatch to the next one."""
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> dict[str, object]:
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go() -> dict[str, object]:
            calls["go"] += 1
            return {
                "extract": lambda amps, window: np.ascontiguousarray(
                    amps, dtype=np.float64
                ),
                "mod": lambda env: 0.25,
            }

        monkeypatch.setattr(e_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(e_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(e_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})
        monkeypatch.setattr(e_mod, "_BACKEND_CACHE", {})

        owner, fn = e_mod._dispatch("mod")
        assert owner == "go"
        assert callable(fn)
        assert float(fn(np.array([1.0], dtype=np.float64))) == 0.25
        assert calls == {"rust": 1, "go": 1}

    def test_probe_returns_infinite_latency_for_broken_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken owner probes as infinitely slow."""

        def broken_loader() -> dict[str, object]:
            raise RuntimeError("backend unavailable during probe")

        monkeypatch.setitem(e_mod._LOADERS, "go", broken_loader)
        e_mod._BACKEND_CACHE.pop("go", None)

        assert e_mod._extract_probe_seconds("go") == float("inf")

    def test_dispatch_falls_back_to_python_when_active_backend_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An active owner that disappears falls back to the NumPy law."""

        def broken_loader() -> dict[str, object]:
            raise OSError("backend disappeared after discovery")

        monkeypatch.setattr(e_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setitem(e_mod._LOADERS, "go", broken_loader)
        e_mod._BACKEND_CACHE.pop("go", None)

        amps = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        got = extract_envelope(amps, window=2)
        expected = np.array([np.sqrt(2.5), np.sqrt(2.5), np.sqrt(6.5)])

        np.testing.assert_allclose(got, expected)


class TestExplicitOwnerContracts:
    """Named owners never hand the call to another owner."""

    @staticmethod
    def _use_go(monkeypatch: pytest.MonkeyPatch, loaded: dict[str, object]) -> None:
        """Install ``loaded`` as the Go owner, with Go active."""
        monkeypatch.setattr(e_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(e_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(e_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(e_mod._LOADERS, "go", lambda: loaded)

    def test_named_owner_whose_loader_fails_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loader failure is a RuntimeError for a named owner."""

        def broken_loader() -> dict[str, object]:
            """Fail like a missing shared library."""
            raise OSError("libenvelope.so missing")

        monkeypatch.setattr(e_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(e_mod._LOADERS, "go", broken_loader)
        with pytest.raises(RuntimeError, match="'go' unavailable"):
            e_mod.envelope_backend("extract", "go")

    def test_named_owner_without_the_operation_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named owner lacking the slot refuses; ``auto`` moves on to Python."""
        self._use_go(monkeypatch, {"mod": lambda env: 0.5})
        with pytest.raises(RuntimeError, match="'go' lacks extract"):
            extract_envelope(np.array([1.0, 2.0, 3.0]), 2, backend="go")
        assert e_mod.envelope_backend("extract") == "python"

    @pytest.mark.parametrize("operation", ["extract", "mod"])
    def test_failing_kernel_refuses_by_name_and_propagates_in_auto(
        self, monkeypatch: pytest.MonkeyPatch, operation: str
    ) -> None:
        """A named owner that fails raises its own refusal; ``auto`` keeps the error."""

        def crash(*_args: object) -> object:
            """Fail like a crashed native kernel."""
            raise RuntimeError("kernel crashed")

        self._use_go(monkeypatch, {"extract": crash, "mod": crash})
        amps = np.array([1.0, 2.0, 3.0])

        def call(backend: str) -> object:
            """Run the operation under test through the public function."""
            if operation == "extract":
                return extract_envelope(amps, 2, backend=backend)
            return envelope_modulation_depth(amps, backend=backend)

        with pytest.raises(RuntimeError, match="envelope backend 'go' failed"):
            call("go")
        with pytest.raises(RuntimeError, match="kernel crashed"):
            call("auto")
