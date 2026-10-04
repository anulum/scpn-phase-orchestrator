# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native output fault boundary fixture

"""Exercise public refusal contracts through an installed defective extension.

The reviewed disposition requires this isolated producer because
the genuine compatible kernel cannot emit these malformed outputs. This fixture
has no numerical solver; it is not used to claim success-path fidelity. G3 admits
coverage only for the exact reviewed guards after the native-plus-absent union.
No production object, availability flag, or imported module is monkeypatched.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from types import ModuleType

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.ei_balance import compute_ei_balance
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


@pytest.fixture
def fault_kernel() -> ModuleType:
    """Require the real compiled fixture and the absence of a genuine distribution."""
    module = importlib.import_module("spo_kernel")
    assert module.__SPO_FAULT_FIXTURE__ == "SPO_NATIVE_OUTPUT_FAULT_FIXTURE_V1"
    assert module.__SPO_FIXTURE_VARIANT__ == "invalid-outputs"
    assert importlib.metadata.version("spo-native-output-fixture") == "0.0.0"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution("spo-kernel")
    return module


@pytest.mark.parametrize("fault", ["ei-ratio-nan", "ei-strength-inf", "ei-block-nan"])
def test_public_ei_refuses_defective_native_metrics(
    fault_kernel: ModuleType, fault: str
) -> None:
    """Refuse actual malformed native metrics without changing caller storage."""
    coupling = np.array([[0.0, 2.0], [1.0, 0.0]])
    before = coupling.copy()
    fault_kernel.fixture_select_fault(fault)
    calls = fault_kernel.fixture_call_count()
    with pytest.raises(
        ValueError, match="^E/I means must remain finite and ratio must not be NaN$"
    ):
        compute_ei_balance(coupling, [0], [1])
    assert fault_kernel.fixture_call_count() == calls + 1
    np.testing.assert_array_equal(coupling, before)


@pytest.mark.parametrize("method", ["step", "run"])
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("sheaf-dimension", "output must be one-dimensional"),
        ("sheaf-cardinality", "returned 5 values, expected 6"),
        ("sheaf-nan", "returned NaN/Inf"),
        ("sheaf-inf", "returned NaN/Inf"),
        ("sheaf-negative", "returned phases outside [0, 2*pi)"),
        ("sheaf-upper-bound", "returned phases outside [0, 2*pi)"),
    ],
)
def test_public_sheaf_refuses_defective_native_output(
    fault_kernel: ModuleType, method: str, fault: str, message: str
) -> None:
    """Validate both public native calls before accepting any state or diagnostic."""
    phases = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    omegas = np.zeros_like(phases)
    maps = np.zeros((3, 3, 2, 2))
    psi = np.zeros(2)
    before = [item.copy() for item in (phases, omegas, maps, psi)]
    engine = SheafUPDEEngine(3, 2, 0.125)
    fault_kernel.fixture_select_fault(fault)
    calls = fault_kernel.fixture_call_count()
    args: tuple[object, ...] = (phases, omegas, maps, 0.0, psi)
    if method == "run":
        args = (*args, 2)
    prefix = f"Rust sheaf {method} "
    if fault == "sheaf-dimension":
        prefix += "output must be one-dimensional"
    else:
        prefix += message
    with pytest.raises(ValueError, match="^" + re.escape(prefix) + "$"):
        getattr(engine, method)(*args)
    assert fault_kernel.fixture_call_count() == calls + 1
    assert engine.last_dt == 0.125
    for original, expected in zip((phases, omegas, maps, psi), before, strict=True):
        np.testing.assert_array_equal(original, expected)


@pytest.mark.parametrize("method", ["step", "run"])
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("sparse-shape", "Sparse output has malformed shape (2,), expected (3,)"),
        ("sparse-bool", "Sparse output must be a real numeric array, got bool"),
        (
            "sparse-complex",
            "Sparse output must be a real numeric array, got complex128",
        ),
        ("sparse-negative", "Sparse output contains phases outside [0, 2*pi)"),
        ("sparse-upper-bound", "Sparse output contains phases outside [0, 2*pi)"),
    ],
)
def test_public_sparse_refuses_defective_native_output(
    fault_kernel: ModuleType, method: str, fault: str, message: str
) -> None:
    """Reject malformed native batch and step outputs without mutating inputs."""
    phases = np.array([0.1, 0.2, 0.3])
    omegas = np.zeros(3)
    row_ptr = np.zeros(4, dtype=np.int64)
    columns = np.empty(0, dtype=np.int64)
    weights = np.empty(0)
    lags = np.empty(0)
    values = (phases, omegas, row_ptr, columns, weights, lags)
    before = [item.copy() for item in values]
    engine = SparseUPDEEngine(3, 0.125)
    fault_kernel.fixture_select_fault(fault)
    calls = fault_kernel.fixture_call_count()
    args: tuple[object, ...] = (
        phases,
        omegas,
        row_ptr,
        columns,
        weights,
        0.0,
        0.0,
        lags,
    )
    if method == "run":
        args = (*args, 2)
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        getattr(engine, method)(*args)
    assert fault_kernel.fixture_call_count() == calls + 1
    assert engine.last_dt == 0.125
    for original, expected in zip(values, before, strict=True):
        np.testing.assert_array_equal(original, expected)
