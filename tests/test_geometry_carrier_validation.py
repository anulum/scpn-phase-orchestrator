# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry carrier validation contracts

"""
Validation contracts for GeometryCarrier seed, decode, update, and latent-dimension
boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier


def test_u1_geometry_carrier_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="seed must be int or None"):
        GeometryCarrier(4, seed=True)  # type: ignore[arg-type]


def test_u1_geometry_carrier_reset_rejects_boolean_seed() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(TypeError, match="seed must be int or None"):
        carrier.reset(seed=True)  # type: ignore[arg-type]


def test_u1_geometry_carrier_decode_rejects_non_array_input() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        carrier.decode(z=[0.1, 0.2, 0.3, 0.4])  # type: ignore[arg-type]


def test_u1_geometry_carrier_update_rejects_boolean_epsilon() -> None:
    carrier = GeometryCarrier(4)
    with pytest.raises(ValueError, match="finite positive real"):
        carrier.update(cost=0.0, epsilon=True)  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_non_positive_latent_dim() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GeometryCarrier(n_oscillators=4, z_dim=0, lr=0.1)


def test_u1_geometry_carrier_update_rejects_non_positive_epsilon() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="finite positive real"):
        carrier.update(cost=0.1, epsilon=0.0)


def test_u1_geometry_carrier_update_rejects_non_callable_cost_fn() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(TypeError, match="callable or None"):
        carrier.update(cost=0.1, cost_fn=1.0)  # type: ignore[arg-type]


def test_u1_geometry_carrier_decode_rejects_wrong_shape() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="length 2"):
        carrier.decode(np.zeros((2, 1), dtype=float))


def test_u1_geometry_carrier_reset_rejects_non_integer_seed() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(TypeError, match="int or None"):
        carrier.reset(seed="bad")  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_non_integer_oscillator_count() -> None:
    with pytest.raises(TypeError, match="n_oscillators must be a positive integer"):
        GeometryCarrier("four")  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_non_positive_oscillator_count() -> None:
    with pytest.raises(ValueError, match="n_oscillators must be a positive integer"):
        GeometryCarrier(0)


def test_u1_geometry_carrier_rejects_non_integer_latent_dim() -> None:
    with pytest.raises(TypeError, match="z_dim must be a positive integer"):
        GeometryCarrier(4, z_dim="eight")  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_non_real_learning_rate() -> None:
    with pytest.raises(TypeError, match="lr must be a finite positive real"):
        GeometryCarrier(4, lr="fast")  # type: ignore[arg-type]


def test_u1_geometry_carrier_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="lr must be a finite positive real"):
        GeometryCarrier(4, lr=-0.1)


def test_u1_geometry_carrier_decode_rejects_non_finite_latent() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="z must contain only finite values"):
        carrier.decode(np.array([0.0, np.nan]))


def test_u1_geometry_carrier_update_rejects_non_finite_cost() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(TypeError, match="cost must be finite real"):
        carrier.update(cost=float("nan"))
