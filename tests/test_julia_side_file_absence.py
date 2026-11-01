# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge side-file refusal contracts

"""Every Julia bridge refuses a missing side-file after a real Julia start.

Each bridge loads its ``.jl`` source only after juliacall has started the Julia
runtime, so the missing-file refusal can only run where Julia is installed.
These tests start the real runtime (no stand-in ``Main``), point the bridge at
a path that does not exist and require the explicit ``ImportError`` that the
public dispatchers treat as an unavailable owner. They run in the prepared
Julia CI lane and are deselected elsewhere by the ``native_runtime`` marker.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_ACCELERATORS = "scpn_phase_orchestrator.experimental.accelerators"

# (bridge module, loader function) pairs whose side-file guard the standard
# and native coverage profiles never reached (CI coverage of 347b37e7).
_JULIA_BRIDGES = (
    ("coupling._hodge_julia", "_ensure"),
    ("coupling._spectral_julia", "_ensure"),
    ("monitor._embedding_julia", "_ensure"),
    ("monitor._entropy_prod_julia", "_ensure"),
    ("monitor._itpc_julia", "_ensure"),
    ("monitor._koopman_edmd_julia", "_ensure"),
    ("monitor._lyapunov_julia", "_ensure"),
    ("monitor._npe_julia", "_ensure"),
    ("monitor._pid_julia", "_ensure"),
    ("monitor._poincare_julia", "_ensure"),
    ("monitor._psychedelic_julia", "_ensure"),
    ("monitor._recurrence_julia", "_ensure"),
    ("monitor._te_julia", "_ensure"),
    ("monitor._twin_confidence_julia", "_ensure"),
    ("monitor._winding_julia", "_ensure"),
    ("upde._engine_julia", "_ensure"),
    ("upde._hypergraph_julia", "_ensure"),
    ("upde._inertial_julia", "_ensure"),
    ("upde._market_julia", "_ensure"),
    ("upde._order_params_julia", "_ensure_julia_loaded"),
    ("upde._swarmalator_julia", "_ensure"),
)


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    ("bridge", "loader"),
    _JULIA_BRIDGES,
    ids=[bridge.split(".")[-1] for bridge, _ in _JULIA_BRIDGES],
)
def test_bridge_refuses_missing_side_file_after_real_julia_start(
    bridge: str,
    loader: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``.jl`` source is an explicit ImportError, not a crash."""
    pytest.importorskip(
        "juliacall", reason="the side-file guard runs only after Julia starts"
    )
    module = importlib.import_module(f"{_ACCELERATORS}.{bridge}")
    original = vars(module)["_JULIA_FILE"]
    assert isinstance(original, Path)
    assert original.exists(), f"{bridge} must ship its side-file: {original}"
    missing = tmp_path / original.name

    monkeypatch.setattr(module, "_JULIA_MODULE", None)
    monkeypatch.setattr(module, "_JULIA_FILE", missing)

    with pytest.raises(ImportError, match="julia side-file not found"):
        getattr(module, loader)()
    assert vars(module)["_JULIA_MODULE"] is None
