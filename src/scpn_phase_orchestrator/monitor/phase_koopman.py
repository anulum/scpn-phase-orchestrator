# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — learned phase-autoencoder Koopman observables

"""Use a trained phase autoencoder as the Koopman observable dictionary.

The analytic dictionaries (identity, polynomial, rbf, phase) are fixed feature
maps. A phase autoencoder (``nn.phase_autoencoder``), trained so its latent
evolves by an exactly-linear normal-form flow, learns observables in which a
nonlinear oscillator's dynamics are close to linear — which is exactly what the
Koopman operator wants. :class:`LearnedKoopmanDictionary` wraps the trained
encoder (frozen to the pure-NumPy ``oscillators.phase_reduction`` evaluator) as a
:class:`~scpn_phase_orchestrator.monitor.koopman_edmd.KoopmanObservables` map, so
the EDMD fit and the condensed Koopman MPC consume learned observables with no
change to their machinery and no JAX on the control path.

The lift is state-inclusive — ``ψ(x) = [x, g(x)]`` with the encoder latent
``g(x)`` — so the output map ``C`` reconstructs the state exactly from the
identity block while the learned block sharpens the linear evolution ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanPredictor,
    fit_koopman_predictor,
)
from scpn_phase_orchestrator.oscillators.phase_reduction import PhaseReducer

FloatArray: TypeAlias = NDArray[np.float64]

_LATENT_DIM = 3

__all__ = ["LearnedKoopmanDictionary", "fit_phase_koopman_predictor"]


@dataclass(frozen=True)
class LearnedKoopmanDictionary:
    """A Koopman observable map backed by a trained phase-autoencoder encoder.

    Parameters
    ----------
    reducer : PhaseReducer
        The frozen-weights evaluator of the trained phase autoencoder.
    include_constant : bool
        Whether to prepend a constant observable for the affine term.
    """

    reducer: PhaseReducer
    include_constant: bool = False

    @property
    def state_dim(self) -> int:
        """The original state dimension ``n``.

        Returns
        -------
        int
            The original state dimension ``n``.
        """
        return int(self.reducer.weights.state_dim)

    @property
    def output_dim(self) -> int:
        """The lifted observable dimension ``N``.

        Returns
        -------
        int
            The lifted observable dimension ``N`` (constant + state + latent).
        """
        constant = 1 if self.include_constant else 0
        return constant + self.state_dim + _LATENT_DIM

    def lift(self, states: FloatArray) -> FloatArray:
        """Lift a batch of states ``(K, n)`` to ``[x, g(x)]`` observables.

        Parameters
        ----------
        states : numpy.ndarray
            The state batch of shape ``(K, n)``.

        Returns
        -------
        numpy.ndarray
            The lifted batch of shape ``(K, output_dim)``.

        Raises
        ------
        ValueError
            If ``states`` is not a finite ``(K, state_dim)`` array.
        """
        latent = self.reducer.encode_observables(states)
        features = np.hstack((np.asarray(states, dtype=np.float64), latent))
        if self.include_constant:
            constant = np.ones((features.shape[0], 1), dtype=np.float64)
            features = np.hstack((constant, features))
        return np.ascontiguousarray(features, dtype=np.float64)


def fit_phase_koopman_predictor(
    reducer: PhaseReducer,
    states: FloatArray,
    next_states: FloatArray,
    inputs: FloatArray,
    *,
    include_constant: bool = False,
    regularisation: float = 1.0e-8,
) -> KoopmanPredictor:
    """Fit an EDMD-with-control predictor in learned phase-autoencoder observables.

    Parameters
    ----------
    reducer : PhaseReducer
        The trained phase-autoencoder evaluator providing the observables.
    states, next_states : numpy.ndarray
        Snapshot states ``x_i`` and successors ``y_i`` of shape ``(K, n)``.
    inputs : numpy.ndarray
        Applied controls ``u_i`` of shape ``(K, m)``.
    include_constant : bool
        Whether the lift prepends a constant observable.
    regularisation : float
        Tikhonov regularisation of the least-squares solve.

    Returns
    -------
    KoopmanPredictor
        The fitted predictor over learned observables.
    """
    dictionary = LearnedKoopmanDictionary(reducer, include_constant=include_constant)
    return fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=dictionary,
        regularisation=regularisation,
    )
