# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent synthetic-connectome reference

"""Replay declared edge equations without importing a production generator."""

from __future__ import annotations

import math
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


def reference_connectome(n_regions: int, seed: int, owner: str) -> NDArray[np.float64]:
    """Return an independent scalar reconstruction of one owner's matrix.

    Parameters
    ----------
    n_regions : int
        At least two synthetic regions, with the extra odd region on the right.
    seed : int
        Unsigned 64-bit seed for the declared owner's original noise law.
    owner : str
        ``python`` uses PCG64 Gaussian noise, ``rust`` uses LCG uniform noise.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Symmetric zero-diagonal weights reconstructed from scalar edge laws.

    Raises
    ------
    ValueError
        If the requested owner is not one of the two implemented generators.

    Notes
    -----
    This is an oracle only. No production dispatch, constants, helper or cache
    is imported. Repeated hub locations retain their historical multiplicities.
    """
    if owner not in {"python", "rust"}:
        raise ValueError("owner must be python or rust")
    half = n_regions // 2
    matrix = np.zeros((n_regions, n_regions), dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    state = seed
    for start, size in [(0, half), (half, n_regions - half)]:
        noise = generator.normal(0.0, 0.02, (size, size)) if owner == "python" else None
        for row in range(size):
            for column in range(size):
                if row == column:
                    continue
                if noise is None:
                    state = (6364136223846793005 * state + 1442695040888963407) % 2**64
                    perturbation = ((state >> 33) / 2**31 - 0.5) * 0.04
                else:
                    perturbation = float(noise[row, column])
                matrix[start + row, start + column] = max(
                    0.0, 0.5 * math.exp(-0.3 * abs(row - column)) + perturbation
                )
    for left in range(half):
        for right in range(min(half, n_regions - half)):
            distance = abs(left - right)
            if distance <= min(3, half):
                weight = 0.15 * math.exp(-0.5 * distance)
                matrix[left, half + right] = matrix[half + right, left] = weight
    hubs = Counter(
        int(fraction * half) + start
        for start in (0, half)
        for fraction in (0.15, 0.45, 0.65, 0.85)
    )
    for row, row_count in hubs.items():
        for column, column_count in hubs.items():
            if row != column:
                matrix[row, column] += 0.3 * row_count * column_count
    return np.asarray((matrix + matrix.T) / 2.0, dtype=np.float64)


def reference_euler_step(
    phases: NDArray[np.float64], knm: NDArray[np.float64], dt: float
) -> NDArray[np.float64]:
    """Integrate the declared unit-frequency unforced Kuramoto equation once.

    Parameters
    ----------
    phases : numpy.typing.NDArray[numpy.float64]
        Initial phase vector in radians.
    knm : numpy.typing.NDArray[numpy.float64]
        Structural zero-diagonal coupling from the public loader.
    dt : float
        Positive Euler interval, in seconds.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Wrapped phase vector from scalar sine sums; no production solver used.
    """
    return np.asarray(
        [
            (
                float(phase)
                + dt
                * (
                    1.0
                    + math.fsum(
                        float(knm[i, j]) * math.sin(float(other - phase))
                        for j, other in enumerate(phases)
                    )
                )
            )
            % (2.0 * math.pi)
            for i, phase in enumerate(phases)
        ],
        dtype=np.float64,
    )


def reference_hcp(n_regions: int = 80) -> NDArray[np.float64]:
    """Read original subject assets independently of the neurolib Dataset class.

    Parameters
    ----------
    n_regions : int
        Top-left cortical region count, from two through eighty.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Subject-wise max-normalised mean of original bundled HCP matrices,
        cortical AAL2 indices only, with a canonical zero diagonal.

    Notes
    -----
    Uses SciPy MATLAB parsing, independent cortical indexing and averaging.
    It imports no production loader, neurolib Dataset or filtering helper.
    """
    module = import_module("neurolib")
    source = module.__file__
    assert source is not None
    root = Path(source).parent / "data/datasets/hcp/subjects"
    files = sorted(root.glob("*/structural/DTI_CM*.mat"))
    assert files
    subjects = []
    for file in files:
        data = cast("dict[str, object]", import_module("scipy.io").loadmat(file))
        raw = np.asarray(data["sc"], dtype=np.float64)
        assert raw.shape == (94, 94)
        keep = [
            i for i in range(94) if i not in set(range(40, 46)) | set(range(74, 82))
        ]
        cortex = raw[np.ix_(keep, keep)]
        subjects.append(cortex / np.max(cortex))
    result = np.asarray(np.mean(subjects, axis=0), dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    return result[:n_regions, :n_regions].copy()


def reference_trajectory(
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
    dt: float,
    n_steps: int,
    method: str,
) -> NDArray[np.float64]:
    """Replay unit-frequency unforced Euler or RK4 with scalar sine reductions.

    Parameters
    ----------
    phases : numpy.typing.NDArray[numpy.float64]
        Original initial phase vector.
    knm : numpy.typing.NDArray[numpy.float64]
        Actual public structural matrix, with zero phase lag and self-coupling.
    dt : float
        Positive fixed integration interval.
    n_steps : int
        Non-negative number of intervals.
    method : str
        ``euler`` or ``rk4`` for the original consumer's chosen method.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Independently integrated phases on the torus.

    Raises
    ------
    ValueError
        If an unimplemented comparison method is requested.
    """
    if method not in {"euler", "rk4"}:
        raise ValueError("reference method must be euler or rk4")

    def derivative(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sum the declared coupling equation without calling a production solver."""
        return np.asarray(
            [
                1.0
                + math.fsum(
                    float(knm[i, j]) * math.sin(float(other - value))
                    for j, other in enumerate(values)
                )
                for i, value in enumerate(values)
            ],
            dtype=np.float64,
        )

    result = phases.copy()
    for _ in range(n_steps):
        first = derivative(result)
        if method == "euler":
            result = (result + dt * first) % (2.0 * math.pi)
        else:
            second = derivative(result + 0.5 * dt * first)
            third = derivative(result + 0.5 * dt * second)
            fourth = derivative(result + dt * third)
            result = (
                result + (dt / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)
            ) % (2.0 * math.pi)
    return result
