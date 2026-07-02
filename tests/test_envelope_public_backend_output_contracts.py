# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - envelope public backend output contracts

"""Public dispatcher contracts for envelope accelerator backend outputs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import envelope as envelope_mod
from scpn_phase_orchestrator.upde.envelope import (
    envelope_modulation_depth,
    extract_envelope,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _install_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extract_output: FloatArray,
    modulation_output: object,
) -> None:
    """Install a deterministic optional envelope backend for public API tests."""

    def extract_backend(_amps: FloatArray, _window: int) -> FloatArray:
        """Return the configured RMS-envelope payload."""
        return np.ascontiguousarray(extract_output, dtype=np.float64)

    def modulation_backend(_env: FloatArray) -> object:
        """Return the configured modulation-depth payload."""
        return modulation_output

    def load_backend() -> dict[str, object]:
        """Return the configured optional backend callables."""
        return {
            "extract": extract_backend,
            "mod": modulation_backend,
        }

    monkeypatch.setattr(envelope_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(envelope_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(envelope_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(envelope_mod, "_LOADERS", {"rust": load_backend})


def test_public_extract_rejects_nonphysical_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject negative optional backend envelope values at the public API."""
    _install_optional_backend(
        monkeypatch,
        extract_output=np.array([0.5, -0.1, 0.7], dtype=np.float64),
        modulation_output=0.25,
    )

    with pytest.raises(ValueError, match="envelope values must be non-negative"):
        extract_envelope(np.array([1.0, 2.0, 3.0], dtype=np.float64), window=2)


def test_public_extract_rejects_wrong_length_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject cardinality drift from optional RMS-envelope backends."""
    _install_optional_backend(
        monkeypatch,
        extract_output=np.array([0.5, 0.7], dtype=np.float64),
        modulation_output=0.25,
    )

    with pytest.raises(ValueError, match="envelope must contain 3 values"):
        extract_envelope(np.array([1.0, 2.0, 3.0], dtype=np.float64), window=2)


def test_public_modulation_rejects_out_of_range_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optional backend modulation depths outside the physical interval."""
    _install_optional_backend(
        monkeypatch,
        extract_output=np.array([0.5, 0.6, 0.7], dtype=np.float64),
        modulation_output=1.5,
    )

    with pytest.raises(ValueError, match="modulation depth must lie in \\[0, 1\\]"):
        envelope_modulation_depth(np.array([0.5, 0.6, 0.7], dtype=np.float64))


def test_public_dispatch_falls_back_to_python_after_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through to the Python floor when optional backend loading fails."""

    def fail_loader() -> dict[str, object]:
        """Raise the optional-backend unavailability signal."""
        raise ImportError("compiled envelope backend unavailable")

    monkeypatch.setattr(envelope_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(envelope_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(envelope_mod, "AVAILABLE_BACKENDS", ["python"])
    monkeypatch.setattr(envelope_mod, "_LOADERS", {"rust": fail_loader})

    result = extract_envelope(np.array([1.0, 2.0, 3.0], dtype=np.float64), window=2)

    np.testing.assert_allclose(
        result,
        np.array([np.sqrt(2.5), np.sqrt(2.5), np.sqrt(6.5)], dtype=np.float64),
    )


def test_public_dispatch_ignores_backend_missing_requested_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the Python floor when a backend lacks the requested callable."""

    def load_backend() -> dict[str, object]:
        """Return a backend table without the modulation-depth callable."""
        return {
            "extract": lambda amps, _window: np.ascontiguousarray(
                amps,
                dtype=np.float64,
            ),
        }

    monkeypatch.setattr(envelope_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(envelope_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(envelope_mod, "AVAILABLE_BACKENDS", ["rust"])
    monkeypatch.setattr(envelope_mod, "_LOADERS", {"rust": load_backend})

    assert envelope_modulation_depth(
        np.array([1.0, 3.0], dtype=np.float64)
    ) == pytest.approx(0.5)


def test_public_extract_two_dimensional_global_rms_shape() -> None:
    """Keep the pure-NumPy 2-D global-RMS edge path shape-stable."""
    amplitudes = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    result = extract_envelope(amplitudes, window=3)

    expected = np.tile(
        np.sqrt(np.mean(amplitudes * amplitudes, axis=0)),
        (amplitudes.shape[0], 1),
    )
    np.testing.assert_allclose(result, expected)


def test_public_modulation_non_positive_python_floor_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the documented zero depth for non-positive Python-floor input."""
    monkeypatch.setattr(envelope_mod, "_dispatch", lambda _fn_name: None)

    assert envelope_modulation_depth(np.array([-3.0, -1.0], dtype=np.float64)) == 0.0
