# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native output fault boundary fixture

"""Check real ImportError recovery using a separately built incomplete extension.

The reviewed disposition requires missing exported classes, not a concealed
module. This build really exports neither stepper, while the absent profile
really lacks the entire extension. Both must match the independent zero-coupling
Euler result; no fixture supplies a successful numerical computation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


def _require_runtime() -> None:
    """Require true absence or the explicitly marked missing-class extension."""
    spec = importlib.util.find_spec("spo_kernel")
    if spec is not None:
        module = importlib.import_module("spo_kernel")
        assert module.__SPO_FAULT_FIXTURE__ == "SPO_NATIVE_OUTPUT_FAULT_FIXTURE_V1"
        assert module.__SPO_FIXTURE_VARIANT__ == "missing-classes"
        assert not hasattr(module, "PySheafUPDEStepper")
        assert not hasattr(module, "PySparseUPDEStepper")
        assert importlib.metadata.version("spo-native-output-fixture") == "0.0.0"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution("spo-kernel")


@pytest.mark.parametrize("count", [1, 3])
def test_public_sheaf_missing_class_matches_absent_numpy_contract(count: int) -> None:
    """Advance real public fallback and compare it to an independent exact oracle."""
    _require_runtime()
    phases = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    omegas = np.array([[0.25, -0.5], [0.75, 1.0], [-1.25, 1.5]])
    before = phases.copy()
    engine = SheafUPDEEngine(3, 2, 0.125)
    result = engine.run(phases, omegas, np.zeros((3, 3, 2, 2)), 0.0, np.zeros(2), count)
    expected = np.remainder(before + 0.125 * count * omegas, 2.0 * np.pi)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(phases, before)
    assert not np.shares_memory(result, phases)
    assert engine.last_dt == 0.125


@pytest.mark.parametrize("count", [1, 3])
def test_public_sparse_missing_class_matches_absent_numpy_contract(count: int) -> None:
    """Check real CSR fallback using the same exact absent-profile Euler oracle."""
    _require_runtime()
    phases = np.array([0.1, 0.2, 0.3])
    omegas = np.array([0.25, -0.5, 0.75])
    before = phases.copy()
    engine = SparseUPDEEngine(3, 0.125)
    result = engine.run(
        phases,
        omegas,
        np.zeros(4, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
        0.0,
        0.0,
        np.empty(0),
        count,
    )
    expected = np.remainder(before + 0.125 * count * omegas, 2.0 * np.pi)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(phases, before)
    assert not np.shares_memory(result, phases)
    assert engine.last_dt == 0.125
