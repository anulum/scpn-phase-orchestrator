# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public connectome numerical and consumer regressions

"""Exercise original synthetic owners against independent edge and solver laws."""

from __future__ import annotations

import cProfile
import importlib.util
import inspect
import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.connectome_benchmark import run_profile
from benchmarks.connectome_reference import reference_connectome, reference_euler_step
from scpn_phase_orchestrator.coupling import connectome
from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


class _Dataset(Protocol):
    """Original optional-provider instance whose data field admits injected faults."""

    Cmat: object


def installed_matrix(
    owner: str, n_regions: int, seed: int = 42, *, kind: str = "synthetic"
) -> NDArray[np.float64]:
    """Exercise a real public loader in its source-qualified installation.

    Parameters
    ----------
    owner : str
        ``python`` is genuinely kernel-absent; ``rust`` has the original builtin.
    n_regions : int
        Actual region count delivered unchanged to the public loader.
    seed : int
        Original unsigned seed for synthetic generation.
    kind : str
        ``synthetic`` or original optional ``hcp`` dataset ingress.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Actual source-qualified installed consumer result.

    Raises
    ------
    ValueError, TypeError, ImportError
        If the original installed public loader refuses the request.
    """
    raw = os.environ.get("SPO_CONNECTOME_" + owner.upper() + "_PROFILE")
    assert raw is not None, "Qualified installed connectome profile is required"
    record = run_profile(
        Path(raw),
        {"owner": owner, "n_regions": n_regions, "seed": seed, "kind": kind},
    )
    error = record["error"]
    if error is not None:
        failure = cast("dict[str, str]", error)
        if failure["type"] == "ImportError":
            raise ImportError(failure["message"])
        if failure["type"] == "TypeError":
            raise TypeError(failure["message"])
        raise ValueError(failure["message"])
    assert record["native_calls"] == (
        1 if owner == "rust" and kind == "synthetic" else 0
    )
    assert record["dtype"] == "float64" and record["contiguous"] is True
    if kind == "synthetic":
        assert record["independent_copy"] is True
    return np.asarray(record["matrix"], dtype=np.float64)


def inject_native_fault(
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[NDArray[np.float64]], object],
) -> None:
    """Corrupt output after original native execution for public refusal tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Own restoration scope for the producer return-boundary fault.
    transform : Callable
        Explicit data fault applied to an originally generated matrix.

    Notes
    -----
    No availability flag changes. These are fault/compatibility contracts,
    independently separate from successful unmodified backend qualification.
    """
    native: object = import_module("spo_kernel").load_hcp_connectome_rust
    assert inspect.isbuiltin(native)
    original = cast("Callable[[int, int], NDArray[np.float64]]", native)

    def corrupt(n_regions: int, seed: int) -> object:
        """Inject the declared fault after executing the original builtin."""
        return transform(original(n_regions, seed).reshape(n_regions, n_regions))

    monkeypatch.setattr(connectome, "_rust_load_hcp", corrupt)


def inject_hcp_fault(
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[NDArray[np.float64]], object],
) -> None:
    """Inject a data fault after the original dataset reads its actual assets.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restoration scope for the original optional provider's factory field.
    transform : Callable
        Fault applied to the real normalised subject-average structural matrix.

    Notes
    -----
    No fake module or successful replacement dataset is supplied. This fault
    contract does not qualify the algorithm or anatomical provenance.
    """
    module = import_module("neurolib.utils.loadData")
    original = cast("Callable[[str], _Dataset]", module.Dataset)

    def corrupt(name: str) -> object:
        """Read the real HCP files before applying the declared matrix fault."""
        dataset = original(name)
        matrix = np.asarray(dataset.Cmat, dtype=np.float64)
        dataset.Cmat = transform(matrix)
        return dataset

    monkeypatch.setattr(module, "Dataset", corrupt)


def _current_owner() -> str:
    """Observe the original optional native symbol without changing resolution."""
    if importlib.util.find_spec("spo_kernel") is not None:
        native = getattr(import_module("spo_kernel"), "load_hcp_connectome_rust", None)
        if native is not None:
            assert inspect.isbuiltin(native)
            return "rust"
    return "python"


@pytest.mark.parametrize("n_regions", [2, 3, 4, 7, 16, 33])
@pytest.mark.parametrize("seed", [0, 42, 2**64 - 1])
def test_public_generator_matches_independent_owner_equations(
    n_regions: int, seed: int
) -> None:
    """Original default generation preserves every edge, noise and repeated hub."""
    owner = _current_owner()
    actual = load_hcp_connectome(n_regions, seed)
    expected = reference_connectome(n_regions, seed, owner)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(np.diag(actual), np.zeros(n_regions))


def test_small_matrix_retains_repeated_hub_multiplicity() -> None:
    """The two-region historical edge is 0.15 plus sixteen 0.3 hub additions."""
    expected = np.array([[0.0, 4.95], [4.95, 0.0]])
    for seed in (0, 42, 2**64 - 1):
        np.testing.assert_allclose(load_hcp_connectome(2, seed), expected, atol=3e-14)


def test_public_cached_results_are_writable_independent_copies() -> None:
    """Editing a returned matrix leaves later public cache results unchanged."""
    owner = _current_owner()
    profile = cProfile.Profile()
    profile.enable()
    first = load_hcp_connectome(12, 98513277)
    original = first.copy()
    first[0, 1] = -100.0
    second = load_hcp_connectome(12, 98513277)
    profile.disable()
    np.testing.assert_array_equal(second, original)
    assert first.flags.writeable and second.flags.writeable
    assert not np.shares_memory(first, second)
    calls = sum(
        row.callcount
        for row in profile.getstats()
        if isinstance(row.code, str) and "load_hcp_connectome_rust" in row.code
    )
    np.testing.assert_allclose(
        original, reference_connectome(12, 98513277, owner), atol=3e-14
    )
    assert calls == (1 if owner == "rust" else 0)


@pytest.mark.parametrize("n_regions", [2**32, 2**64, np.iinfo(np.intp).max])
def test_public_refuses_unaddressable_dense_storage(n_regions: int) -> None:
    """Invalid dense dimensions are refused before original native allocation."""
    with pytest.raises(ValueError, match="exceeds addressable float64 storage"):
        load_hcp_connectome(n_regions, 42)
    assert np.isfinite(load_hcp_connectome(3, 42)).all()


@pytest.mark.native_runtime
@pytest.mark.parametrize("self_edge", [np.nextafter(0.0, 1.0), 1e-16, 0.25])
def test_public_refuses_corrupted_native_self_edge(
    monkeypatch: pytest.MonkeyPatch, self_edge: float
) -> None:
    """An injected self-edge is refused at the public consumer boundary.

    This fault contract modifies output after original native execution and
    does not supply evidence for successful native generation.
    """
    native: object = import_module("spo_kernel").load_hcp_connectome_rust
    assert inspect.isbuiltin(native)
    original = cast("Callable[[int, int], NDArray[np.float64]]", native)

    def corrupt_output(n_regions: int, seed: int) -> NDArray[np.float64]:
        """Inject the invalid edge after executing the original generator."""
        matrix = original(n_regions, seed)
        matrix[0] = self_edge
        return matrix

    monkeypatch.setattr(connectome, "_rust_load_hcp", corrupt_output)
    with pytest.raises(ValueError, match="diagonal must be zero"):
        load_hcp_connectome(3, 9382517)


@pytest.mark.parametrize("n_regions", [3, 8, 17])
def test_original_public_matrix_drives_independent_euler_transition(
    n_regions: int,
) -> None:
    """Real loader weights affect UPDE phases and coherence by the scalar law."""
    knm = load_hcp_connectome(n_regions, 71)
    phases = np.linspace(0.1, 2.7, n_regions)
    expected = reference_euler_step(phases, knm, 0.001)
    engine = UPDEEngine(n_regions, dt=0.001, method="euler")
    actual = engine.step(phases, np.ones(n_regions), knm, 0.0, 0.0, np.zeros_like(knm))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    uncoupled = (phases + 0.001) % (2.0 * np.pi)
    assert np.max(np.abs(actual - uncoupled)) > 1e-5
    coherence, mean_phase = compute_order_parameter(actual)
    order = np.mean(np.exp(1j * expected))
    assert coherence == pytest.approx(abs(order), abs=3e-14)
    assert mean_phase == pytest.approx(float(np.angle(order)), abs=3e-14)
