# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase autoencoder for model-free phase reduction

"""A phase autoencoder for model-free phase reduction of limit-cycle oscillators.

Classical phase reduction needs the vector field. The phase autoencoder of
Yawata, Fukami, Taira & Nakao (2024) learns the asymptotic phase, the isochrons
and the phase-sensitivity function from state time series alone. An encoder maps
the oscillator state ``x`` to a three-component latent ``Y = (Y₁, Y₂, Y₃)`` whose
first two components are constrained to the unit circle ``Y₁² + Y₂² = 1`` so that
``θ = atan2(Y₂, Y₁)`` is the asymptotic phase; the latent then evolves by the
exactly-linear normal-form flow

    Y₁,ₜ₊τ = Y₁ cos(ωτ) − Y₂ sin(ωτ),
    Y₂,ₜ₊τ = Y₁ sin(ωτ) + Y₂ cos(ωτ),
    Y₃,ₜ₊τ = e^{λτ} Y₃,        λ < 0,

with learnable frequency ``ω`` and decay ``λ``; a decoder reconstructs ``x``. The
four-term training objective ties reconstruction, the uniform phase rotation, the
amplitude decay and a centring term that prevents the trivial ``ω = 0`` solution.

The trained encoder/decoder weights and ``(ω, λ)`` are extracted to a
:class:`PhaseReductionWeights` record consumed by the pure-NumPy, dependency-light
evaluator in ``oscillators.phase_reduction`` so the asymptotic phase and the
phase-sensitivity function are available on the control path without JAX.

This module follows the published latent constraint and four-term loss; the
encoder and decoder are plain ReLU multilayer perceptrons (no batch
normalisation, which is a training-stability detail outside the phase-reduction
mathematics and would complicate the frozen-weights evaluator).

References
----------
* Yawata, Fukami, Taira & Nakao 2024, *Chaos* 34, 063111 (arXiv:2403.06992) —
  phase autoencoder for limit-cycle oscillators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.phase_reduction import (
    PhaseReductionWeights,
)

if TYPE_CHECKING:
    import optax

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "PhaseAutoencoder",
    "PhaseReductionWeights",
    "extract_phase_reduction_weights",
    "phase_autoencoder_loss",
    "train_phase_autoencoder",
]

_LATENT_DIM = 3


class PhaseAutoencoder(eqx.Module):
    """Encoder/decoder with a phase-circle latent and normal-form dynamics.

    Parameters
    ----------
    state_dim : int
        The oscillator state dimension ``n``.
    hidden : int
        Hidden width of the encoder and decoder multilayer perceptrons.
    key : jax.Array
        PRNG key for weight initialisation.
    """

    encoder: list[eqx.nn.Linear]
    decoder: list[eqx.nn.Linear]
    raw_omega: jax.Array
    raw_decay: jax.Array
    state_dim: int = eqx.field(static=True)

    def __init__(self, state_dim: int, *, hidden: int = 64, key: jax.Array) -> None:
        keys = jax.random.split(key, 7)
        self.encoder = [
            eqx.nn.Linear(state_dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, _LATENT_DIM, key=keys[2]),
        ]
        self.decoder = [
            eqx.nn.Linear(_LATENT_DIM, hidden, key=keys[3]),
            eqx.nn.Linear(hidden, hidden, key=keys[4]),
            eqx.nn.Linear(hidden, state_dim, key=keys[5]),
        ]
        # ω initialised near 1; λ = −softplus(raw_decay) is strictly negative.
        self.raw_omega = jnp.asarray(1.0)
        self.raw_decay = jnp.asarray(0.0)
        self.state_dim = state_dim

    @property
    def omega(self) -> jax.Array:
        """The learned angular frequency ``ω`` of the latent rotation.

        Returns
        -------
        jax.Array
            The learned angular frequency ``ω``.
        """
        return self.raw_omega

    @property
    def decay(self) -> jax.Array:
        """The learned amplitude decay ``λ = −softplus(raw_decay) < 0``.

        Returns
        -------
        jax.Array
            The learned amplitude decay ``λ``, strictly negative.
        """
        return -jax.nn.softplus(self.raw_decay)

    def encode_raw(self, state: jax.Array) -> jax.Array:
        """Encode a single state to the unnormalised latent ``(Ỹ₁, Ỹ₂, Ỹ₃)``.

        Parameters
        ----------
        state : jax.Array
            The oscillator state of shape ``(state_dim,)``.

        Returns
        -------
        jax.Array
            The unnormalised latent of shape ``(3,)``.
        """
        activation = state
        for layer in self.encoder[:-1]:
            activation = jax.nn.relu(layer(activation))
        return self.encoder[-1](activation)

    def encode(self, state: jax.Array) -> jax.Array:
        """Encode a single state to the phase-circle latent ``(Y₁, Y₂, Y₃)``.

        Parameters
        ----------
        state : jax.Array
            The oscillator state of shape ``(state_dim,)``.

        Returns
        -------
        jax.Array
            The latent of shape ``(3,)`` with ``Y₁² + Y₂² = 1``.
        """
        raw = self.encode_raw(state)
        radius = jnp.sqrt(raw[0] ** 2 + raw[1] ** 2) + 1.0e-9
        return jnp.stack((raw[0] / radius, raw[1] / radius, raw[2]))

    def decode(self, latent: jax.Array) -> jax.Array:
        """Decode a latent ``(Y₁, Y₂, Y₃)`` back to the oscillator state.

        Parameters
        ----------
        latent : jax.Array
            The latent of shape ``(3,)``.

        Returns
        -------
        jax.Array
            The reconstructed oscillator state of shape ``(state_dim,)``.
        """
        activation = latent
        for layer in self.decoder[:-1]:
            activation = jax.nn.relu(layer(activation))
        return self.decoder[-1](activation)

    def advance(self, latent: jax.Array, dt: float) -> jax.Array:
        """Advance a latent by ``dt`` under the exactly-linear normal-form flow.

        Parameters
        ----------
        latent : jax.Array
            The latent of shape ``(3,)``.
        dt : float
            The time increment.

        Returns
        -------
        jax.Array
            The advanced latent of shape ``(3,)``.
        """
        angle = self.omega * dt
        cos = jnp.cos(angle)
        sin = jnp.sin(angle)
        rotated_1 = latent[0] * cos - latent[1] * sin
        rotated_2 = latent[0] * sin + latent[1] * cos
        decayed = jnp.exp(self.decay * dt) * latent[2]
        return jnp.stack((rotated_1, rotated_2, decayed))

    def asymptotic_phase(self, state: jax.Array) -> jax.Array:
        """Return the asymptotic phase ``θ = atan2(Y₂, Y₁)`` of a state.

        Parameters
        ----------
        state : jax.Array
            The oscillator state of shape ``(state_dim,)``.

        Returns
        -------
        jax.Array
            The asymptotic phase in ``(−π, π]``.
        """
        latent = self.encode(state)
        return jnp.arctan2(latent[1], latent[0])


def _layer_arrays(
    layers: list[eqx.nn.Linear],
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    weights = tuple(np.asarray(layer.weight, dtype=np.float64) for layer in layers)
    biases = tuple(
        np.zeros(int(layer.out_features), dtype=np.float64)
        if layer.bias is None
        else np.asarray(layer.bias, dtype=np.float64)
        for layer in layers
    )
    return weights, biases


def extract_phase_reduction_weights(
    model: PhaseAutoencoder,
) -> PhaseReductionWeights:
    """Extract a trained model's weights into a frozen NumPy record.

    Parameters
    ----------
    model : PhaseAutoencoder
        The trained phase autoencoder.

    Returns
    -------
    PhaseReductionWeights
        The frozen encoder/decoder weights and ``(ω, λ)`` for the pure-NumPy
        ``oscillators.phase_reduction`` evaluator.
    """
    encoder_weights, encoder_biases = _layer_arrays(model.encoder)
    decoder_weights, decoder_biases = _layer_arrays(model.decoder)
    return PhaseReductionWeights(
        encoder_weights=encoder_weights,
        encoder_biases=encoder_biases,
        decoder_weights=decoder_weights,
        decoder_biases=decoder_biases,
        omega=float(model.omega),
        decay=float(model.decay),
        state_dim=int(model.state_dim),
    )


def phase_autoencoder_loss(
    model: PhaseAutoencoder,
    windows: jax.Array,
    *,
    dt: float,
    weight_recon: float = 1.0,
    weight_phase: float = 0.5,
    weight_deviation: float = 0.5,
    weight_aux: float = 2.0,
) -> jax.Array:
    """Return the four-term phase-autoencoder training loss (Yawata et al. 2024).

    Parameters
    ----------
    model : PhaseAutoencoder
        The model under training.
    windows : jax.Array
        Trajectory windows of shape ``(batch, K + 1, state_dim)`` — ``K + 1``
        consecutive states sampled at spacing ``dt``.
    dt : float
        The sampling interval between consecutive states in a window.
    weight_recon, weight_phase, weight_deviation, weight_aux : float
        The loss-term weights.

    Returns
    -------
    jax.Array
        The scalar total loss.
    """
    encode = jax.vmap(model.encode)
    decode = jax.vmap(model.decode)

    flat = windows.reshape(-1, model.state_dim)
    reconstruction = jnp.mean((decode(encode(flat)) - flat) ** 2)

    initial = windows[:, 0, :]
    latent = jax.vmap(model.encode)(initial)
    horizon = windows.shape[1] - 1
    phase_loss = jnp.asarray(0.0)
    deviation_loss = jnp.asarray(0.0)
    for step in range(1, horizon + 1):
        latent = jax.vmap(lambda y: model.advance(y, dt))(latent)
        observed = jax.vmap(model.encode)(windows[:, step, :])
        scale = 1.0 / step
        phase_loss = phase_loss + scale * jnp.mean(
            (observed[:, :2] - latent[:, :2]) ** 2
        )
        deviation_loss = deviation_loss + scale * jnp.mean(
            (observed[:, 2] - latent[:, 2]) ** 2
        )

    encoded_initial = jax.vmap(model.encode_raw)(initial)
    auxiliary = (
        jnp.mean(encoded_initial[:, 0]) ** 2 + jnp.mean(encoded_initial[:, 1]) ** 2
    )

    return (
        weight_recon * reconstruction
        + weight_phase * phase_loss
        + weight_deviation * deviation_loss
        + weight_aux * auxiliary
    )


def train_phase_autoencoder(
    windows: jax.Array,
    *,
    dt: float,
    state_dim: int,
    hidden: int = 64,
    epochs: int = 200,
    learning_rate: float = 1.0e-3,
    seed: int = 0,
    loss_kwargs: dict[str, float] | None = None,
) -> tuple[PhaseAutoencoder, jax.Array]:
    """Train a phase autoencoder on trajectory windows.

    Parameters
    ----------
    windows : jax.Array
        Trajectory windows of shape ``(batch, K + 1, state_dim)``.
    dt : float
        The sampling interval within a window.
    state_dim : int
        The oscillator state dimension ``n``.
    hidden : int
        Hidden width of the encoder/decoder.
    epochs : int
        Number of full-batch Adam steps.
    learning_rate : float
        The Adam learning rate.
    seed : int
        PRNG seed for weight initialisation.
    loss_kwargs : dict[str, float] | None
        Optional overrides for the loss-term weights.

    Returns
    -------
    tuple[PhaseAutoencoder, jax.Array]
        The trained model and the final loss value.
    """
    import optax

    overrides = loss_kwargs or {}
    model = PhaseAutoencoder(state_dim, hidden=hidden, key=jax.random.key(seed))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def _loss(candidate: PhaseAutoencoder) -> jax.Array:
        return phase_autoencoder_loss(candidate, windows, dt=dt, **overrides)

    step = _make_train_step(optimizer, _loss)
    loss_value = jnp.asarray(jnp.inf)
    for _ in range(epochs):
        model, opt_state, loss_value = step(model, opt_state)
    return model, loss_value


def _make_train_step(
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[PhaseAutoencoder], jax.Array],
) -> Callable[[PhaseAutoencoder, object], tuple[PhaseAutoencoder, object, jax.Array]]:
    @eqx.filter_jit
    def step(
        model: PhaseAutoencoder, opt_state: object
    ) -> tuple[PhaseAutoencoder, object, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return step
