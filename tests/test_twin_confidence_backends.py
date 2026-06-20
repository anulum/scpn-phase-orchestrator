# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for twin-confidence

"""Per-backend parity tests for the twin-confidence divergence kernel.

Complements ``test_twin_confidence.py`` (which exercises the active backend)
by checking each non-Python backend individually against the NumPy reference,
and by covering the shared backend-validation contract that runs regardless of
which toolchains are installed.

Each backend gates on its toolchain being present:

* Rust — always in a working SPO dev environment (built by maturin). Bit-exact
  parity (Python and Rust share f64 arithmetic on the same hardware).
* Go — needs ``go/libtwin_confidence.so``. Bit-exact parity.
* Julia — needs ``juliacall`` and ``julia/twin_confidence.jl``. Bit-exact parity.
* Mojo — needs ``mojo/twin_confidence_mojo``. The Wasserstein term is bit-exact;
  the Jensen–Shannon term carries a ~1e-9 floor because Mojo's ``std.math.log``
  is an approximation, so its parity budget is ``1e-8``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _twin_confidence_go as go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _twin_confidence_julia as julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _twin_confidence_mojo as mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _twin_confidence_validation as validation,
)
from scpn_phase_orchestrator.monitor import twin_confidence as tc

twin_divergence_go = go_mod.twin_divergence_go
twin_divergence_julia = julia_mod.twin_divergence_julia
twin_divergence_mojo = mojo_mod.twin_divergence_mojo
_load_lib = go_mod._load_lib
_ensure_exe = mojo_mod._ensure_exe

TwinBackend = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int],
    np.ndarray,
]
LN2 = float(np.log(2.0))


def _rust_backend(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int, w: int, nb: int
) -> np.ndarray:
    from spo_kernel import twin_divergence_rust

    return np.asarray(
        twin_divergence_rust(
            np.ascontiguousarray(a),
            np.ascontiguousarray(b),
            np.ascontiguousarray(c),
            np.ascontiguousarray(d),
            n,
            w,
            nb,
        )
    )


def _available(name: str) -> bool:
    try:
        tc._load_backend(name)
    except (ImportError, RuntimeError, OSError, KeyError):
        return False
    return True


def _backend(name: str) -> TwinBackend:
    table: dict[str, TwinBackend] = {
        "rust": _rust_backend,
        "go": twin_divergence_go,
        "julia": twin_divergence_julia,
        "mojo": twin_divergence_mojo,
    }
    return table[name]


_TOLERANCE = {"rust": 1e-12, "go": 1e-12, "julia": 1e-12, "mojo": 1e-8}


# ---------------------------------------------------------------------
# Shared validation contract (always runs)
# ---------------------------------------------------------------------


def test_validation_helper_is_linked_to_backends() -> None:
    assert callable(validation.validate_twin_divergence_backend_inputs)
    assert callable(validation.validate_twin_divergence_backend_output)


def test_validation_inputs_round_trip() -> None:
    out = validation.validate_twin_divergence_backend_inputs(
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7], [0.1, 0.2, 0.3], 2, 3, 12
    )
    model_phases, observed_phases, model_order, observed_order, n, w, nb = out
    assert n == 2
    assert w == 3
    assert nb == 12
    assert model_phases.dtype == np.float64
    assert observed_order.shape == (3,)


def test_validation_rejects_boolean_alias() -> None:
    with pytest.raises(ValueError, match="boolean"):
        validation._validate_vector([True, False], name="model_phases")


def test_validation_rejects_complex() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        validation._validate_vector([1 + 2j, 3], name="model_phases")


def test_validation_rejects_non_numeric_array() -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        validation._validate_vector(["a", "b"], name="model_phases")


def test_validation_rejects_non_one_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validation._validate_vector(np.zeros((2, 2)), name="model_phases")


def test_validation_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validation._validate_vector([0.1, np.nan], name="model_phases")


def test_validation_rejects_order_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validation._validate_order_vector([0.5, 1.2], name="model_order")


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_validation_rejects_non_positive_int(bad: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validation._validate_positive_int(bad, name="n")


def test_validation_inputs_reject_phase_length_mismatch() -> None:
    with pytest.raises(ValueError, match="phase vector lengths"):
        validation.validate_twin_divergence_backend_inputs(
            [0.1, 0.2], [0.3, 0.4], [0.5], [0.6], 3, 1, 8
        )


def test_validation_inputs_reject_order_length_mismatch() -> None:
    with pytest.raises(ValueError, match="order vector lengths"):
        validation.validate_twin_divergence_backend_inputs(
            [0.1, 0.2], [0.3, 0.4], [0.5], [0.6], 2, 2, 8
        )


def test_validation_output_round_trip() -> None:
    out = validation.validate_twin_divergence_backend_output([0.3, 0.4])
    assert out.tolist() == [0.3, 0.4]


def test_validation_output_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"not \(2,\)"):
        validation.validate_twin_divergence_backend_output([0.1, 0.2, 0.3])


def test_validation_output_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validation.validate_twin_divergence_backend_output([np.inf, 0.1])


def test_validation_output_rejects_js_out_of_range() -> None:
    with pytest.raises(ValueError, match="Jensen"):
        validation.validate_twin_divergence_backend_output([1.0, 0.1])


def test_validation_output_rejects_w1_out_of_range() -> None:
    with pytest.raises(ValueError, match="Wasserstein"):
        validation.validate_twin_divergence_backend_output([0.1, 1.5])


# ---------------------------------------------------------------------
# Bridge input validation runs before the toolchain (always runs)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend_fn",
    [twin_divergence_go, twin_divergence_julia, twin_divergence_mojo],
)
def test_bridge_validates_before_toolchain(backend_fn: TwinBackend) -> None:
    with pytest.raises(ValueError, match="order-parameter values"):
        backend_fn(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 9.0]),
            np.array([0.5, 0.6]),
            2,
            2,
            8,
        )


def test_go_lib_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_mod, "_LIB", None)
    monkeypatch.setattr(go_mod, "_LIB_PATH", Path("/nonexistent/libtwin_confidence.so"))
    with pytest.raises(ImportError, match="libtwin_confidence.so not found"):
        go_mod._load_lib()


def test_julia_sidefile_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    juliacall = pytest.importorskip("juliacall")
    if not hasattr(juliacall, "Main"):  # force runtime init, like the dispatcher
        pytest.skip("juliacall.Main unavailable")
    monkeypatch.setattr(julia_mod, "_JULIA_MODULE", None)
    monkeypatch.setattr(
        julia_mod, "_JULIA_FILE", Path("/nonexistent/twin_confidence.jl")
    )
    with pytest.raises(ImportError, match="julia side-file not found"):
        julia_mod._ensure()


def test_mojo_exe_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mojo_mod, "_EXE_PATH", Path("/nonexistent/twin_confidence_mojo")
    )
    with pytest.raises(ImportError, match="not built"):
        mojo_mod._ensure_exe()


def test_go_nonzero_return_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLib:
        def TwinDivergence(self, *args: object) -> int:  # noqa: N802 - C ABI name
            return 7

    monkeypatch.setattr(go_mod, "_LIB", _FakeLib())
    with pytest.raises(ValueError, match="rc=7"):
        go_mod.twin_divergence_go(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5]),
            np.array([0.6]),
            2,
            1,
            8,
        )


def _fake_proc(returncode: int, stdout: str, stderr: str = "") -> object:
    import types

    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_mojo_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mojo_mod, "_ensure_exe", lambda: Path("fake_twin_mojo"))
    monkeypatch.setattr(
        mojo_mod.subprocess, "run", lambda *a, **k: _fake_proc(2, "", "boom")
    )
    with pytest.raises(ValueError, match="exit 2"):
        mojo_mod.twin_divergence_mojo(
            np.array([0.1]), np.array([0.2]), np.array([0.5]), np.array([0.6]), 1, 1, 8
        )


def test_mojo_wrong_line_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mojo_mod, "_ensure_exe", lambda: Path("fake_twin_mojo"))
    monkeypatch.setattr(
        mojo_mod.subprocess, "run", lambda *a, **k: _fake_proc(0, "0.1\n")
    )
    with pytest.raises(ValueError, match="exactly 2 scalar"):
        mojo_mod.twin_divergence_mojo(
            np.array([0.1]), np.array([0.2]), np.array([0.5]), np.array([0.6]), 1, 1, 8
        )


def test_mojo_non_scalar_output_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mojo_mod, "_ensure_exe", lambda: Path("fake_twin_mojo"))
    monkeypatch.setattr(
        mojo_mod.subprocess, "run", lambda *a, **k: _fake_proc(0, "0.1\nNaNsense\n")
    )
    with pytest.raises(ValueError, match="non-scalar"):
        mojo_mod.twin_divergence_mojo(
            np.array([0.1]), np.array([0.2]), np.array([0.5]), np.array([0.6]), 1, 1, 8
        )


# ---------------------------------------------------------------------
# Per-backend parity (gated on toolchain availability)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", ["rust", "go", "julia", "mojo"])
def test_backend_parity_against_reference(backend_name: str) -> None:
    if not _available(backend_name):
        pytest.skip(f"{backend_name} backend toolchain not available")
    fn = _backend(backend_name)
    tol = _TOLERANCE[backend_name]
    rng = np.random.default_rng(2024)
    max_dev = 0.0
    for _ in range(25):
        n = int(rng.integers(5, 220))
        w = int(rng.integers(2, 96))
        nb = int(rng.integers(4, 80))
        a = rng.uniform(-14.0, 14.0, n)
        b = rng.uniform(-14.0, 14.0, n)
        c = rng.uniform(0.0, 1.0, w)
        d = rng.uniform(0.0, 1.0, w)
        reference = tc._python_kernel(a, b, c, d, n, w, nb)
        out = np.asarray(fn(a, b, c, d, n, w, nb))
        max_dev = max(
            max_dev,
            abs(out[0] - reference[0]),
            abs(out[1] - reference[1]),
        )
    assert max_dev < tol, f"{backend_name} parity {max_dev:.2e} exceeds {tol:.0e}"


@pytest.mark.parametrize("backend_name", ["rust", "go", "julia", "mojo"])
def test_backend_identical_streams_are_zero(backend_name: str) -> None:
    if not _available(backend_name):
        pytest.skip(f"{backend_name} backend toolchain not available")
    fn = _backend(backend_name)
    phases = np.array([0.2, 1.4, 2.6, 3.8, 5.0])
    order = np.array([0.3, 0.5, 0.7])
    out = np.asarray(fn(phases, phases.copy(), order, order.copy(), 5, 3, 36))
    assert out[0] == pytest.approx(0.0, abs=_TOLERANCE[backend_name])
    assert out[1] == pytest.approx(0.0, abs=_TOLERANCE[backend_name])


def test_backends_agree_pairwise() -> None:
    present = [name for name in ("rust", "go", "julia", "mojo") if _available(name)]
    if len(present) < 2:
        pytest.skip("need at least two backends for cross-backend agreement")
    _load_lib() if "go" in present else None
    _ensure_exe() if "mojo" in present else None
    rng = np.random.default_rng(55)
    a = rng.uniform(0.0, 2.0 * np.pi, 100)
    b = rng.uniform(0.0, 2.0 * np.pi, 100)
    c = rng.uniform(0.0, 1.0, 40)
    d = rng.uniform(0.0, 1.0, 40)
    outputs = {
        name: np.asarray(_backend(name)(a, b, c, d, 100, 40, 48)) for name in present
    }
    baseline = outputs[present[0]]
    for name, out in outputs.items():
        tol = max(_TOLERANCE[name], _TOLERANCE[present[0]])
        assert out[0] == pytest.approx(baseline[0], abs=tol)
        assert out[1] == pytest.approx(baseline[1], abs=tol)
