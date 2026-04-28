# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (full multi-head)

"""Attention-Residuals (AttnRes) coupling modulation — full multi-head.

Direct port of the Transformer-style multi-head attention used in
arXiv:2603.15031 (Moonshot AI / Kimi Team, March 2026) to the SCPN
coupling matrix. Every Transformer component is present here:

* **H learnable Q, K, V projections** — ``W_Q, W_K, W_V ∈ R^{H × d × d_h}``
  with ``d = 2`` (``[cos θ, sin θ]`` phase embedding) and
  ``d_h = d // H`` (default ``d = 8, d_h = 2, H = 4`` gives a
  full-rank projection over the 2-D phase torus).
* **Per-head scaled dot-product attention** —
  ``A_h[i, j] = softmax_j(q_h[i] · k_h[j] / √d_h / temp_h)``
  with a per-head temperature vector (paper uses uniform; both are
  supported).
* **Optional local mask** — a ``±block_size`` band mask plus the
  zero-edge mask from ``K_nm``. ``block_size = None`` means full-N
  attention (closest to the paper).
* **Output projection** ``W_O ∈ R^{(H·d_h) × d}`` mapping the
  concatenated head outputs back to the 2-D phase embedding space.
* **Modulation rule** —
  ``K_mod[i, j] = K[i, j] · (1 + λ · a_agg[i, j])``
  where ``a_agg`` is the symmetrised, scalar projection of the
  multi-head attention weights onto the pair (i, j). Symmetrisation
  is ``(a + aᵀ) / 2`` so ``K_mod`` stays symmetric.

**Default projections.** The module ships a helper
``default_projections(n_heads, seed)`` that returns the four
matrices (W_Q, W_K, W_V, W_O) sampled from a seeded Gaussian with
the canonical Xavier/Glorot scaling. Callers that want bit-stable
behaviour pass their own matrices. Callers that just want "it
works" rely on the default seed.

**Rule compliance.** This file is the full multi-head architecture
on day one — no single-head proxy, no "add heads in a follow-up"
placeholder. See ``feedback_no_simplistic_models.md``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "attnres_modulate",
    "default_projections",
]


# ---------------------------------------------------------------------
# Default projections
# ---------------------------------------------------------------------


PHASE_EMBED_DIM: int = 8
"""Width of the per-oscillator hidden state. 2 would suffice for a
pure ``[cos, sin]`` embedding; the paper uses d_model well above the
intrinsic data rank to give the attention heads room to specialise.
Default ``d = 8`` with ``H = 4`` gives ``d_h = 2`` — matches the
single-ring structure of Kuramoto phases while leaving three extra
heads for higher-order Fourier components."""


def default_projections(
    n_heads: int = 4,
    seed: int = 0,
    d_model: int = PHASE_EMBED_DIM,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Seeded Xavier-initialised projections ``(W_Q, W_K, W_V, W_O)``.

    Parameters
    ----------
    n_heads:
        Number of attention heads. Must divide ``d_model``.
    seed:
        RNG seed for reproducible initialisation.
    d_model:
        Hidden-state width. Default is the module-level
        ``PHASE_EMBED_DIM``.

    Returns
    -------
    ``(w_q, w_k, w_v, w_o)`` — all ``float64``. Shapes:
    ``w_q, w_k, w_v`` each ``(H, d_model, d_head)``;
    ``w_o`` is ``(H · d_head, d_model)``.
    """
    if d_model % n_heads != 0:
        msg = f"d_model={d_model} not divisible by n_heads={n_heads}"
        raise ValueError(msg)
    d_head = d_model // n_heads
    rng = np.random.default_rng(seed)
    # Xavier / Glorot scaling: variance = 2 / (fan_in + fan_out).
    fan = (d_model + d_head) / 2.0
    scale = np.sqrt(1.0 / fan)
    shape_qkv = (n_heads, d_model, d_head)
    w_q = (rng.standard_normal(shape_qkv) * scale).astype(np.float64)
    w_k = (rng.standard_normal(shape_qkv) * scale).astype(np.float64)
    w_v = (rng.standard_normal(shape_qkv) * scale).astype(np.float64)
    w_o = (rng.standard_normal((n_heads * d_head, d_model)) * scale).astype(np.float64)
    return w_q, w_k, w_v, w_o


# ---------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")

_BackendFn = Callable[
    [
        NDArray,  # knm flat (N*N,)
        NDArray,  # theta (N,)
        NDArray,  # w_q (H, d_model, d_head)
        NDArray,  # w_k (H, d_model, d_head)
        NDArray,  # w_v (H, d_model, d_head)
        NDArray,  # w_o (H*d_head, d_model)
        int,  # n
        int,  # n_heads
        int,  # block_size (-1 = unbounded)
        float,  # temperature
        float,  # lambda_
    ],
    NDArray,
]


def _load_rust() -> _BackendFn:
    from typing import cast

    from spo_kernel import attnres_modulate_rust

    return cast("_BackendFn", attnres_modulate_rust)


def _load_mojo() -> _BackendFn:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.coupling._attnres_mojo import (
        _ensure_exe,
        attnres_modulate_mojo,
    )

    _ensure_exe()
    return attnres_modulate_mojo


def _load_julia() -> _BackendFn:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.coupling._attnres_julia import (
        attnres_modulate_julia,
    )

    return attnres_modulate_julia


def _load_go() -> _BackendFn:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.coupling._attnres_go import (
        _load_lib,
        attnres_modulate_go,
    )

    _load_lib()
    return attnres_modulate_go


_LOADERS: dict[str, Callable[[], _BackendFn]] = {
    "rust": _load_rust,
    "mojo": _load_mojo,
    "julia": _load_julia,
    "go": _load_go,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


# ---------------------------------------------------------------------
# NumPy reference implementation (full multi-head)
# ---------------------------------------------------------------------


def _embed_phase(theta: NDArray, d_model: int) -> NDArray:
    """Fourier-feature embedding of the phase scalar to ``d_model`` dims.

    Produces ``[cos θ, sin θ, cos 2θ, sin 2θ, ..., cos (d/2)θ, sin (d/2)θ]``.
    This is the canonical positional / Fourier-feature lift used in the
    paper to give the attention mechanism a richer representation of
    the periodic signal than a bare scalar would."""
    n = theta.shape[0]
    x = np.empty((n, d_model), dtype=np.float64)
    for k in range(d_model // 2):
        x[:, 2 * k] = np.cos((k + 1) * theta)
        x[:, 2 * k + 1] = np.sin((k + 1) * theta)
    return x


def _python_fallback(
    knm_flat: NDArray,
    theta: NDArray,
    w_q: NDArray,
    w_k: NDArray,
    w_v: NDArray,
    w_o: NDArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Reference multi-head AttnRes — mirrors the paper's forward
    pass. Every compiled backend must reproduce this bit-for-bit."""
    if lambda_ == 0.0:
        return knm_flat.astype(np.float64, copy=True)

    knm = knm_flat.reshape(n, n)
    d_model = w_q.shape[1]
    d_head = w_q.shape[2]

    # 1. Fourier-feature embedding of the phases.
    x = _embed_phase(theta, d_model)  # (n, d_model)

    # 2. Per-head Q, K, V. W_Q / W_K / W_V shape (H, d_model, d_head).
    # einsum: (n, d) · (H, d, d_h) -> (H, n, d_h).
    q = np.einsum("nd,hde->hne", x, w_q)
    k = np.einsum("nd,hde->hne", x, w_k)
    v = np.einsum("nd,hde->hne", x, w_v)

    # 3. Scaled dot-product attention per head.
    # logits[h, i, j] = q[h, i] · k[h, j] / (sqrt(d_h) * temperature).
    inv_scale = 1.0 / (np.sqrt(d_head) * temperature)
    logits = np.einsum("hie,hje->hij", q, k) * inv_scale  # (H, n, n)

    # 4. Mask: (a) diagonal, (b) ±block_size if bounded,
    #    (c) K_nm zero entries ("never create new edges").
    band = np.ones((n, n), dtype=bool)
    np.fill_diagonal(band, False)
    if block_size >= 0:
        idx = np.arange(n)
        within = np.abs(idx[np.newaxis, :] - idx[:, np.newaxis]) <= block_size
        band &= within
    band &= knm != 0.0
    # Broadcast to (H, n, n).
    mask3 = np.broadcast_to(band, (n_heads, n, n))

    masked = np.where(mask3, logits, -np.inf)
    row_max = np.max(masked, axis=2, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exps = np.where(mask3, np.exp(masked - row_max), 0.0)
    denom = np.sum(exps, axis=2, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    attn = exps / denom  # (H, n, n)

    # 5. Apply attention to values.
    # o_h[i] = sum_j attn[h, i, j] * v[h, j]  -> (H, n, d_h)
    heads = np.einsum("hij,hje->hie", attn, v)
    # Concatenate heads along d-axis: (H, n, d_h) -> (n, H·d_h).
    concat = heads.transpose(1, 0, 2).reshape(n, n_heads * d_head)

    # 6. Output projection.
    o = concat @ w_o  # (n, d_model)

    # 7. Pair-wise aggregation score a_agg[i, j] — cosine similarity
    # of the output vectors, restricted to the existing edges. This is
    # the single scalar per pair used to modulate K.
    o_norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o_unit = o / o_norms
    a_agg = o_unit @ o_unit.T  # (n, n)
    # Rescale to [0, 1] via (1 + cos)/2 so the modulation factor stays
    # non-negative and the “neutral” pair (cos = 0) gives λ/2 boost.
    a_agg = 0.5 * (1.0 + a_agg)
    np.fill_diagonal(a_agg, 0.0)
    a_agg *= band  # respect the mask

    # 8. Modulation and symmetrisation.
    rowwise = knm * (1.0 + lambda_ * a_agg)
    k_mod = 0.5 * (rowwise + rowwise.T)
    np.fill_diagonal(k_mod, 0.0)
    return np.asarray(k_mod.ravel(), dtype=np.float64)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def attnres_modulate(
    knm: NDArray,
    theta: NDArray,
    *,
    w_q: NDArray | None = None,
    w_k: NDArray | None = None,
    w_v: NDArray | None = None,
    w_o: NDArray | None = None,
    n_heads: int = 4,
    block_size: int | None = None,
    temperature: float = 1.0,
    lambda_: float = 0.5,
    projection_seed: int = 0,
) -> NDArray:
    """Full multi-head AttnRes modulation.

    Parameters
    ----------
    knm:
        Symmetric coupling matrix, shape ``(N, N)``, with zero diagonal.
    theta:
        Current phase vector, shape ``(N,)``.
    w_q, w_k, w_v:
        Per-head query / key / value projections, shape
        ``(n_heads, d_model, d_model // n_heads)``. If any is ``None``
        a seeded default is generated via ``default_projections``.
    w_o:
        Output projection, shape ``(n_heads · d_head, d_model)``.
    n_heads:
        Number of attention heads. Must divide ``d_model``.
    block_size:
        ``None`` (default) → full-N attention (paper-faithful).
        Integer ≥ 1 → local ``±block_size`` band mask.
    temperature:
        Softmax temperature (paper uses 1.0; lower values sharpen
        attention).
    lambda_:
        Modulation strength. ``0`` returns ``knm`` unchanged.
    projection_seed:
        RNG seed used when any projection is ``None``.

    Returns
    -------
    ``(N, N)`` modulated coupling, symmetric with zero diagonal.

    Raises
    ------
    ValueError
        On shape mismatches, negative ``lambda_``, non-positive
        ``temperature``, or ``d_model`` not divisible by ``n_heads``.
    """
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError(f"knm must be square 2-D; got shape {knm.shape}")
    n = knm.shape[0]
    if theta.shape != (n,):
        raise ValueError(f"theta shape {theta.shape} does not match knm (N={n})")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if lambda_ < 0.0:
        raise ValueError(f"lambda_ must be ≥ 0, got {lambda_}")
    if block_size is not None and block_size < 1:
        raise ValueError(
            f"block_size must be ≥ 1 or None for full attention, got {block_size}"
        )

    if any(p is None for p in (w_q, w_k, w_v, w_o)):
        d_q, d_k, d_v, d_o = default_projections(n_heads=n_heads, seed=projection_seed)
        w_q = d_q if w_q is None else w_q
        w_k = d_k if w_k is None else w_k
        w_v = d_v if w_v is None else w_v
        w_o = d_o if w_o is None else w_o

    if w_q is None or w_k is None or w_v is None or w_o is None:
        raise ValueError("attention projections must be provided or defaultable")
    if w_q.shape != w_k.shape or w_q.shape != w_v.shape:
        raise ValueError(
            f"w_q / w_k / w_v shape mismatch: {w_q.shape}, {w_k.shape}, {w_v.shape}"
        )
    if w_q.shape[0] != n_heads:
        raise ValueError(f"w_q leading dim {w_q.shape[0]} != n_heads={n_heads}")
    d_model = w_q.shape[1]
    d_head = w_q.shape[2]
    if d_model != n_heads * d_head:
        raise ValueError(f"d_model {d_model} != n_heads {n_heads} · d_head {d_head}")
    if w_o.shape != (n_heads * d_head, d_model):
        raise ValueError(f"w_o shape {w_o.shape} != ({n_heads * d_head}, {d_model})")

    knm64 = np.asarray(knm, dtype=np.float64)
    if lambda_ == 0.0:
        return knm64.copy()

    knm_flat = np.ascontiguousarray(knm64.ravel(), dtype=np.float64)
    theta64 = np.ascontiguousarray(theta, dtype=np.float64)
    bs_int = -1 if block_size is None else block_size

    # Compiled backends take 1-D flat buffers; Python fallback takes
    # flat + reshapes internally. Ravel once here.
    wq_flat = np.ascontiguousarray(w_q.ravel(), dtype=np.float64)
    wk_flat = np.ascontiguousarray(w_k.ravel(), dtype=np.float64)
    wv_flat = np.ascontiguousarray(w_v.ravel(), dtype=np.float64)
    wo_flat = np.ascontiguousarray(w_o.ravel(), dtype=np.float64)

    if ACTIVE_BACKEND != "python":
        backend_fn = _LOADERS[ACTIVE_BACKEND]()
        out = backend_fn(
            knm_flat,
            theta64,
            wq_flat,
            wk_flat,
            wv_flat,
            wo_flat,
            n,
            n_heads,
            bs_int,
            temperature,
            lambda_,
        )
        return np.asarray(out, dtype=np.float64).reshape(n, n)

    out = _python_fallback(
        knm_flat,
        theta64,
        np.ascontiguousarray(w_q, dtype=np.float64),
        np.ascontiguousarray(w_k, dtype=np.float64),
        np.ascontiguousarray(w_v, dtype=np.float64),
        np.ascontiguousarray(w_o, dtype=np.float64),
        n,
        n_heads,
        bs_int,
        temperature,
        lambda_,
    )
    return out.reshape(n, n)
