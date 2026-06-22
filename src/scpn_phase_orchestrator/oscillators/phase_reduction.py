# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — dependency-light phase-reduction evaluator

"""A pure-NumPy evaluator for a trained phase autoencoder.

The phase autoencoder (``nn.phase_autoencoder``) is trained with JAX, but the
asymptotic phase and the phase-sensitivity function it learns are needed on the
control path, which must stay dependency-light. This module evaluates the trained
encoder/decoder — frozen to plain NumPy weights — without importing JAX.

Given the trained encoder ``g(x) = (Ỹ₁, Ỹ₂, Ỹ₃)`` (a ReLU multilayer
perceptron), the asymptotic phase is ``Θ(x) = atan2(Ỹ₂, Ỹ₁)`` (the unit-circle
normalisation cancels inside ``atan2``). The phase-sensitivity function — the
gradient of the phase with respect to the state, evaluated on the limit cycle —
is

    Z(θ) = ∇ₓ Θ |_{x = decode(cos θ, sin θ, 0)}
         = (∂Θ/∂Ỹ₁) ∇ₓ Ỹ₁ + (∂Θ/∂Ỹ₂) ∇ₓ Ỹ₂,

with ``∂Θ/∂Ỹ₁ = −Ỹ₂/(Ỹ₁²+Ỹ₂²)``, ``∂Θ/∂Ỹ₂ = Ỹ₁/(Ỹ₁²+Ỹ₂²)`` and the encoder
Jacobian computed by exact reverse-mode through the ReLU network. This is the
phase response curve (Nakao 2016) recovered model-free from data.

References
----------
* Yawata, Fukami, Taira & Nakao 2024, *Chaos* 34, 063111 — phase autoencoder.
* Nakao 2016, *Contemp. Phys.* 57, 188 — phase reduction theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["PhaseReducer", "PhaseReductionWeights"]


@dataclass(frozen=True)
class PhaseReductionWeights:
    """Frozen encoder/decoder weights and ``(ω, λ)`` of a phase autoencoder.

    Parameters
    ----------
    encoder_weights, encoder_biases : tuple[numpy.ndarray, ...]
        Per-layer encoder weight matrices ``(out, in)`` and bias vectors.
    decoder_weights, decoder_biases : tuple[numpy.ndarray, ...]
        Per-layer decoder weight matrices and bias vectors.
    omega : float
        The learned angular frequency ``ω``.
    decay : float
        The learned amplitude decay ``λ < 0``.
    state_dim : int
        The oscillator state dimension ``n``.
    """

    encoder_weights: tuple[FloatArray, ...]
    encoder_biases: tuple[FloatArray, ...]
    decoder_weights: tuple[FloatArray, ...]
    decoder_biases: tuple[FloatArray, ...]
    omega: float
    decay: float
    state_dim: int


def _validate_state(value: object, *, name: str, dim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.shape[0] != dim:
        raise ValueError(f"{name} must have {dim} entries, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass(frozen=True)
class PhaseReducer:
    """A dependency-light evaluator of a trained phase autoencoder.

    Parameters
    ----------
    weights : PhaseReductionWeights
        The frozen encoder/decoder weights and ``(ω, λ)``.
    """

    weights: PhaseReductionWeights

    @property
    def omega(self) -> float:
        """The learned angular frequency ``ω``.

        Returns
        -------
        float
            The learned angular frequency ``ω``.
        """
        return self.weights.omega

    @property
    def decay(self) -> float:
        """The learned amplitude decay ``λ < 0``.

        Returns
        -------
        float
            The learned amplitude decay ``λ``, strictly negative.
        """
        return self.weights.decay

    def _encode_raw(self, state: FloatArray) -> FloatArray:
        activation = state
        weights = self.weights.encoder_weights
        biases = self.weights.encoder_biases
        last = len(weights) - 1
        for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
            activation = weight @ activation + bias
            if index != last:
                activation = np.maximum(activation, 0.0)
        return activation

    def _decode(self, latent: FloatArray) -> FloatArray:
        activation = latent
        weights = self.weights.decoder_weights
        biases = self.weights.decoder_biases
        last = len(weights) - 1
        for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
            activation = weight @ activation + bias
            if index != last:
                activation = np.maximum(activation, 0.0)
        return activation

    def _encoder_jacobian(self, state: FloatArray) -> FloatArray:
        """Return the exact encoder Jacobian ``∂g/∂x`` of shape ``(3, n)``."""
        activation = state
        weights = self.weights.encoder_weights
        biases = self.weights.encoder_biases
        last = len(weights) - 1
        jacobian = np.eye(state.shape[0], dtype=np.float64)
        for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
            pre = weight @ activation + bias
            jacobian = weight @ jacobian
            if index != last:
                derivative = (pre > 0.0).astype(np.float64)
                activation = np.maximum(pre, 0.0)
                jacobian = derivative[:, None] * jacobian
        return jacobian

    def asymptotic_phase(self, state: FloatArray) -> float:
        """Return the asymptotic phase ``Θ(x) = atan2(Ỹ₂, Ỹ₁)`` of a state.

        Parameters
        ----------
        state : numpy.ndarray
            The oscillator state ``x`` of shape ``(n,)``.

        Returns
        -------
        float
            The asymptotic phase in ``(−π, π]``.

        Raises
        ------
        ValueError
            If ``state`` is not a finite vector of length ``state_dim``.
        """
        vector = _validate_state(state, name="state", dim=self.weights.state_dim)
        raw = self._encode_raw(vector)
        return float(np.arctan2(raw[1], raw[0]))

    def reconstruct(self, phase: float) -> FloatArray:
        """Reconstruct the on-cycle state at a phase via the decoder.

        Parameters
        ----------
        phase : float
            The phase ``θ`` to reconstruct on the limit cycle.

        Returns
        -------
        numpy.ndarray
            The decoded state ``decode(cos θ, sin θ, 0)`` of shape ``(n,)``.
        """
        latent = np.array(
            [np.cos(float(phase)), np.sin(float(phase)), 0.0], dtype=np.float64
        )
        return np.ascontiguousarray(self._decode(latent), dtype=np.float64)

    def phase_sensitivity(self, phase: float) -> FloatArray:
        """Return the phase-sensitivity function ``Z(θ) = ∇ₓ Θ`` on the cycle.

        Parameters
        ----------
        phase : float
            The phase ``θ`` on the limit cycle at which to evaluate ``Z``.

        Returns
        -------
        numpy.ndarray
            The phase response curve value ``Z(θ)`` of shape ``(n,)``.
        """
        state = self.reconstruct(phase)
        raw = self._encode_raw(state)
        denominator = raw[0] ** 2 + raw[1] ** 2 + 1.0e-12
        dphase_dy1 = -raw[1] / denominator
        dphase_dy2 = raw[0] / denominator
        jacobian = self._encoder_jacobian(state)
        sensitivity = dphase_dy1 * jacobian[0] + dphase_dy2 * jacobian[1]
        return np.ascontiguousarray(sensitivity, dtype=np.float64)
