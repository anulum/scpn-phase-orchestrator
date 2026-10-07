# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent scalar phase attention oracle

"""Evaluate the documented coupling equations without production intermediates."""

from __future__ import annotations

import math
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class AttnResOptions(TypedDict, total=False):
    """Typed public parameters shared by runtime parity tests."""

    w_q: FloatArray
    w_k: FloatArray
    w_v: FloatArray
    w_o: FloatArray
    n_heads: int
    block_size: int | None
    temperature: float
    lambda_: float
    projection_seed: int


def phase_attention_oracle(
    coupling: FloatArray,
    phases: FloatArray,
    projections: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    *,
    temperature: float = 1.0,
    strength: float = 0.5,
    radius: int | None = None,
) -> FloatArray:
    """Evaluate masked phase attention with scalar sums and an independent layout.

    Parameters
    ----------
    coupling : numpy.ndarray
        Symmetric graph with zero diagonal.
    phases : numpy.ndarray
        Oscillator phases in radians.
    projections : tuple of numpy.ndarray
        Query/key/value tensors followed by the output matrix.
    temperature : float
        Positive softmax temperature.
    strength : float
        Non-negative multiplicative modulation strength.
    radius : int or None
        Optional index-band radius.

    Returns
    -------
    numpy.ndarray
        Modulated symmetric coupling with unchanged absent edges.
    """
    query_weights, key_weights, value_weights, output_weights = projections
    count = len(phases)
    heads, width, head_width = query_weights.shape
    features = [
        [
            function(harmonic * float(phase))
            for harmonic in range(1, width // 2 + 1)
            for function in (math.cos, math.sin)
        ]
        for phase in phases
    ]
    outputs: list[list[float]] = []
    for target in range(count):
        neighbours = [
            source
            for source in range(count)
            if source != target
            and coupling[target, source] != 0.0
            and (radius is None or abs(target - source) <= radius)
        ]
        concatenated: list[float] = []
        for head in range(heads):
            query = [
                math.fsum(
                    features[target][d] * float(query_weights[head, d, e])
                    for d in range(width)
                )
                for e in range(head_width)
            ]
            logits = [
                math.fsum(
                    query[e]
                    * math.fsum(
                        features[source][d] * float(key_weights[head, d, e])
                        for d in range(width)
                    )
                    for e in range(head_width)
                )
                / math.sqrt(head_width)
                / temperature
                for source in neighbours
            ]
            maximum = max(logits, default=0.0)
            weights = [math.exp(logit - maximum) for logit in logits]
            denominator = math.fsum(weights)
            for e in range(head_width):
                concatenated.append(
                    math.fsum(
                        weight
                        / denominator
                        * math.fsum(
                            features[source][d] * float(value_weights[head, d, e])
                            for d in range(width)
                        )
                        for source, weight in zip(neighbours, weights, strict=True)
                    )
                )
        outputs.append(
            [
                math.fsum(
                    concatenated[d] * float(output_weights[d, e]) for d in range(width)
                )
                for e in range(width)
            ]
        )
    norms = [
        math.sqrt(math.fsum(value * value for value in row)) + 1e-12 for row in outputs
    ]
    result = coupling.copy()
    for target in range(count):
        for source in range(count):
            if target == source:
                result[target, source] = 0.0
            elif coupling[target, source] != 0.0 and (
                radius is None or abs(target - source) <= radius
            ):
                cosine = math.fsum(
                    a / norms[target] * b / norms[source]
                    for a, b in zip(outputs[target], outputs[source], strict=True)
                )
                score = min(1.0, max(0.0, (1.0 + cosine) / 2.0))
                result[target, source] *= 1.0 + strength * score
    return result
