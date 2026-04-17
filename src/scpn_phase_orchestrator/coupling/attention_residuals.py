# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Phase-3 experiment)

"""Attention-Residuals (AttnRes) inspired coupling modulation.

State-dependent reshaping of the ``K_nm`` coupling matrix based on
the current phase-coherence structure of the network, as proposed in
``docs/internal/research_attention_residuals_2026-04-06.md §3.2``.

**Source.** arXiv:2603.15031 (Moonshot AI / Kimi Team, March 2026)
proposes learned, softmax-weighted aggregation across network depth
for language models. The SCPN analogue treats the ``K_ij`` coupling
weight as a counterpart to a residual connection weight; oscillator
pairs that are currently close in phase receive a multiplicative
boost on their existing K_ij.

**Contract.**

* ``attnres_modulate`` is a pure function — it does not mutate
  ``K_nm``; returns a new ``(N, N)`` float64 array.
* Symmetry ``K_mod[i,j] == K_mod[j,i]`` is enforced by averaging the
  forward and backward softmax direction.
* Zero diagonal is preserved (``K_mod[i,i] == 0``).
* When ``lambda_ == 0`` the result equals ``K_nm`` exactly
  (identity fallback — lets a caller disable the modulation without
  extra branching).
* ``block_size`` restricts attention to a ``±block_size`` window
  around each oscillator, matching the paper's local block attention
  pattern and bounding the per-step cost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["ACTIVE_BACKEND", "AVAILABLE_BACKENDS", "attnres_modulate"]

# Fastest-first fallback chain. Each entry is a ``(name, loader)`` pair;
# the loader either returns a callable with the
# ``(knm_flat, theta, n, block_size, temperature, lambda_) → flat``
# signature, or raises ``ImportError`` / ``RuntimeError`` when its
# toolchain is not available on this host. Python is always the
# terminal fallback so the feature still works without any compiled
# backend. Order follows the global rule
# ``feedback_fallback_chain_ordering.md`` — Rust → Mojo → Julia → Go → Python.

_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust():  # type: ignore[no-untyped-def]
    from spo_kernel import attnres_modulate_rust

    return attnres_modulate_rust


def _load_mojo():  # type: ignore[no-untyped-def]  # pragma: no cover — toolchain-gated
    # Also probe the compiled executable — the module loads fine even
    # when the binary is missing, so the existence check drops us
    # through to the next backend rather than surfacing a runtime
    # error mid-integration.
    from scpn_phase_orchestrator.coupling._attnres_mojo import (
        _ensure_exe,
        attnres_modulate_mojo,
    )

    _ensure_exe()
    return attnres_modulate_mojo


def _load_julia():  # type: ignore[no-untyped-def]  # pragma: no cover — toolchain-gated
    # Probe the *actual* toolchain at resolve time, not just the wrapper
    # module — the wrapper module itself has no import-time dependency
    # on juliacall.
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.coupling._attnres_julia import (
        attnres_modulate_julia,
    )

    return attnres_modulate_julia


def _load_go():  # type: ignore[no-untyped-def]  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.coupling._attnres_go import (
        attnres_modulate_go,
    )

    return attnres_modulate_go


_LOADERS = {
    "rust": _load_rust,
    "mojo": _load_mojo,
    "julia": _load_julia,
    "go": _load_go,
}


def _resolve_backends() -> tuple[str, list[str]]:
    """Probe each candidate backend. Return ``(active, available)``.

    ``available`` lists every backend whose toolchain is present, in
    fastest-first order. ``active`` is the head of that list, or
    ``"python"`` when no compiled backend loads.
    """
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:  # every non-python backend
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")  # terminal fallback
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _softmax_row(logits: NDArray, mask: NDArray) -> NDArray:
    """Softmax over masked entries; rows of zeros return zeros."""
    masked = np.where(mask, logits, -np.inf)
    # Numerically-stable softmax: subtract the row max.
    row_max = np.max(masked, axis=1, keepdims=True)
    # A row with no unmasked entries has row_max = -inf; the subtraction
    # would produce NaN, so guard by replacing those rows with 0.
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exps = np.where(mask, np.exp(masked - row_max), 0.0)
    denom = np.sum(exps, axis=1, keepdims=True)
    # Rows with zero denominator (no unmasked entries) get zero weights.
    denom = np.where(denom > 0.0, denom, 1.0)
    return exps / denom


def attnres_modulate(
    knm: NDArray,
    theta: NDArray,
    *,
    block_size: int = 4,
    temperature: float = 0.1,
    lambda_: float = 0.5,
) -> NDArray:
    """Reshape ``K_nm`` via state-dependent (Hebbian-like) attention.

    Parameters
    ----------
    knm:
        Symmetric coupling matrix, shape ``(N, N)``, with zero diagonal.
    theta:
        Current phase vector, shape ``(N,)``.
    block_size:
        Half-width of the local attention window. Each oscillator
        attends to pairs with ``|i - j| <= block_size``. Must be ≥ 1.
    temperature:
        Softmax temperature. Lower values sharpen the attention onto
        the most-coherent neighbours; higher values flatten towards
        uniform weighting. Must be > 0.
    lambda_:
        Modulation strength. ``lambda_ == 0`` returns ``knm`` unchanged.
        Must be ≥ 0 (negative would *anti*-reinforce, which inverts the
        physical story).

    Returns
    -------
    ``(N, N)`` float64 array. Same dtype & shape as ``knm``; symmetric
    with zero diagonal.

    Raises
    ------
    ValueError
        On non-square ``knm``, shape mismatch with ``theta``, or any
        invalid hyperparameter.
    """
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError(
            f"knm must be square 2-D; got shape {knm.shape}"
        )
    n = knm.shape[0]
    if theta.shape != (n,):
        raise ValueError(
            f"theta shape {theta.shape} does not match knm (N={n})"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be ≥ 1, got {block_size}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if lambda_ < 0.0:
        raise ValueError(f"lambda_ must be ≥ 0, got {lambda_}")

    knm = np.asarray(knm, dtype=np.float64)

    if lambda_ == 0.0:
        return knm.copy()

    if ACTIVE_BACKEND != "python":
        backend_fn = _LOADERS[ACTIVE_BACKEND]()
        knm_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        theta_flat = np.ascontiguousarray(theta, dtype=np.float64)
        return np.asarray(
            backend_fn(knm_flat, theta_flat, n, block_size, temperature, lambda_),
            dtype=np.float64,
        ).reshape(n, n)

    # ────── Python / NumPy fallback ──────
    # Phase-coherence logits: cos(θ_j - θ_i) / temperature.
    # Broadcasting: theta[None, :] - theta[:, None] → (N, N).
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    logits = np.cos(diff) / temperature

    # Local-block mask within ±block_size, excluding the diagonal.
    idx = np.arange(n)
    band = np.abs(idx[np.newaxis, :] - idx[:, np.newaxis]) <= block_size
    np.fill_diagonal(band, False)
    # Further require that K_nm itself is non-zero at that pair — we
    # never create new edges, only reshape the existing topology.
    mask = band & (knm != 0.0)

    attn = _softmax_row(logits, mask)
    factor = 1.0 + lambda_ * attn
    # Mask out the diagonal explicitly: 1 + 0 = 1 (no change), and
    # K_nm[i, i] = 0 so the product is also 0.
    k_mod_rowwise = knm * factor

    # Symmetrise by averaging the forward and backward directions. The
    # per-row softmax is not symmetric in (i, j), so we average the two
    # views to restore ``K_mod[i,j] == K_mod[j,i]``. The diagonal was
    # already zero in knm; symmetrisation preserves that.
    return 0.5 * (k_mod_rowwise + k_mod_rowwise.T)
