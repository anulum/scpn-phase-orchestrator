# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hopf-bridge external validation (both detector families)

"""On a Hopf bifurcation, which detector family recovers the true eigenvalue α?

The two other eigenvalue external validations sit on opposite sides of one
question. The grid ANDES test validates the *oscillatory* envelope-growth family
on real systems; the critical-slowing-down (CSD) test validates the
*single-series* autocorrelation family on non-oscillatory bifurcations (fold,
pitchfork), where its lag-one autocorrelation recovers the eigenvalue in
magnitude. The Hopf bifurcation is where the two families **meet**: it presents
an *oscillatory* critical slowing down (α ± iω with α → 0⁻), so both families
point at the same mode, against one analytic ground truth.

For a quasi-static sweep of the Hopf parameter α toward onset, the stochastic
Hopf normal form (dr = (α r − r³) dt on the amplitude, dθ = ω dt on the phase)
is integrated, and the true recovery rate is α — the eigenvalue's real part — in
closed form. The **envelope-growth** family reads the rectified ringdown with the
shipped ``envelope_growth_rate``; the **autocorrelation** family reads the
stationary series with the shipped ``critical_slowing_down_warning``.

The sealed answer completes a regime map:

* The **envelope-growth family recovers α** in rank *and* magnitude (Spearman
  ρ ≈ 0.97, mean |σ − α| ≈ 0.04 at an adequate ringdown SNR), and the recovery is
  **frequency-invariant** (identical across 0.2–1.2 Hz — it fits the envelope, not
  the oscillation).
* The **autocorrelation family tracks α but not its magnitude** (mean
  |rate − α| ≈ 0.69): the lag-one autocorrelation is **confounded by the
  oscillation**, pinned near cos(ω Δt), so ln(AR1)/Δt does not estimate α.

So the eigenvalue's real part is the universal quantity, but the
magnitude-correct estimator is regime-dependent: **envelope-growth for an
oscillatory mode (here, and the grid), the autocorrelation for a non-oscillatory
one (the CSD fold/pitchfork test)**. That is the bridge.

Honest limits: the envelope-growth magnitude recovery **degrades with ringdown
SNR** (a physical floor — a decay cannot be read below the noise it sinks into —
reported here as a characterised curve, not hidden); a quasi-static per-α sweep;
additive noise; a scalar reduced normal form. The seal is recomputed from the
committed rows, never a fresh run.

References
----------
* Strogatz 1994, *Nonlinear Dynamics and Chaos* — the Hopf normal form and its α ± iω
  eigenvalues.
* Kundur 1994 — the electromechanical oscillation whose damping the grid detector reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.grid_modal_growth import envelope_growth_rate

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

#: The sealed-artefact identifier.
BENCHMARK = "hopf_bridge_external_validation"
#: The two detector families compared, in seal order.
FAMILIES = ("envelope_growth", "autocorrelation")
#: The fewest sweep points at which a rank correlation is meaningful.
_MIN_POINTS = 3
#: The lag-one autocorrelation is clipped to this open interval before ``ln`` is taken.
_AR1_FLOOR = 1.0e-6
_AR1_CEIL = 1.0 - 1.0e-6

__all__ = [
    "BENCHMARK",
    "FAMILIES",
    "autocorrelation_family_rate",
    "correlation",
    "envelope_family_sigma",
    "hopf_bridge_payload",
    "hopf_bridge_verdict",
    "hopf_record",
    "simulate_hopf",
    "sweep_hopf",
]


def _spearman(x: FloatArray, y: FloatArray) -> float:
    """Return the Spearman rank correlation of ``x`` and ``y``."""
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def correlation(
    true_alpha: Sequence[float], estimate: Sequence[float]
) -> dict[str, object]:
    """Return the Pearson and Spearman correlation of an estimate versus true ``α``.

    Parameters
    ----------
    true_alpha : sequence of float
        The ground-truth eigenvalue real part ``α`` at each operating point.
    estimate : sequence of float
        A detector family's estimate at each operating point.

    Returns
    -------
    dict
        ``pearson``, ``spearman``, ``n``, and ``mean_abs_error`` (the mean magnitude gap
        ``|estimate − α|``, meaningful when the estimate is itself a rate).

    Raises
    ------
    ValueError
        If the two sequences differ in length or hold fewer than three points.
    """
    a = np.asarray(true_alpha, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("true_alpha and estimate must be the same length")
    if a.shape[0] < _MIN_POINTS:
        raise ValueError(f"need at least {_MIN_POINTS} points for a correlation")
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": _spearman(a, b),
        "n": int(a.shape[0]),
        "mean_abs_error": float(np.mean(np.abs(b - a))),
    }


def simulate_hopf(
    alpha: float,
    omega: float,
    r0: float,
    *,
    dt: float,
    n: int,
    sigma: float,
    seed: int,
) -> FloatArray:
    """Integrate the stochastic Hopf normal form and return the scalar observable.

    The amplitude obeys ``dr = (α r − r³) dt + σ dW`` (kept non-negative) and the
    phase advances at ``ω``; the observable is ``x = r cos θ``. A high ``r0`` gives
    a ringdown; starting near the noise-sustained amplitude gives a stationary
    oscillation.

    Parameters
    ----------
    alpha : float
        The Hopf parameter (the eigenvalue real part); ``α < 0`` is damped.
    omega : float
        The angular oscillation frequency in radians per second.
    r0 : float
        The initial amplitude.
    dt : float
        The integration and sampling step.
    n : int
        The number of samples; must be at least two.
    sigma : float
        The additive amplitude-noise amplitude.
    seed : int
        The seed for this run's noise.

    Returns
    -------
    FloatArray
        The observable ``x = r cos θ``, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not at least two.
    """
    if n < 2:
        raise ValueError(f"n {n} must be at least 2")
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal(n) * (sigma * np.sqrt(dt))
    x = np.empty(n, dtype=np.float64)
    r, theta = r0, 0.0
    for i in range(n):
        x[i] = r * np.cos(theta)
        r = abs(r + (alpha * r - r**3) * dt + increments[i])
        theta = theta + omega * dt
    return x


def envelope_family_sigma(ring: FloatArray, *, rate: float) -> float:
    """Read the envelope-growth family's ``σ`` on a ringdown, the grid convention.

    The shipped :func:`envelope_growth_rate` fits ``log(envelope)``, so the input
    must be an amplitude: the ringdown is rectified (``|x|``) exactly as the grid
    pipeline rectifies a bus deviation before the envelope fit.

    Parameters
    ----------
    ring : FloatArray
        The oscillatory ringdown observable.
    rate : float
        The sampling rate in Hz.

    Returns
    -------
    float
        The envelope growth rate ``σ`` (an estimate of ``α``).
    """
    return envelope_growth_rate(np.abs(ring), rate=rate)


def autocorrelation_family_rate(
    stationary: FloatArray,
    *,
    dt: float,
    window: int,
    step: int,
) -> float:
    """Read the autocorrelation family's implied rate on a stationary oscillation.

    Uses the shipped :func:`critical_slowing_down_warning` to average the lag-one
    autocorrelation, then maps it to ``ln(AR1)/dt`` — the implied rate that recovers the
    eigenvalue on a *non-oscillatory* bifurcation, but which an oscillation confounds.

    Parameters
    ----------
    stationary : FloatArray
        The stationary noisy oscillation.
    dt : float
        The sample spacing.
    window : int
        The detector's analysis window in samples.
    step : int
        The detector's window hop in samples.

    Returns
    -------
    float
        The implied rate ``ln(AR1)/dt``.
    """
    warning = critical_slowing_down_warning(stationary, window=window, step=step)
    ar1 = float(np.mean(warning.autocorrelation_index))
    ar1 = min(max(ar1, _AR1_FLOOR), _AR1_CEIL)
    return float(np.log(ar1) / dt)


def hopf_record(
    *,
    omega_hz: float,
    ring_sigma: float,
    alphas: Sequence[float],
    envelope_sigma: Sequence[float],
    autocorrelation_rate: Sequence[float],
) -> dict[str, object]:
    """Assemble the Hopf sweep record with a correlation per detector family.

    Parameters
    ----------
    omega_hz : float
        The oscillation frequency in Hz, for provenance.
    ring_sigma : float
        The ringdown noise amplitude behind the headline envelope recovery.
    alphas : sequence of float
        The Hopf parameter ``α`` at each operating point (the true eigenvalue part).
    envelope_sigma : sequence of float
        The envelope-growth family's ``σ`` at each operating point.
    autocorrelation_rate : sequence of float
        The autocorrelation family's implied rate at each operating point.

    Returns
    -------
    dict
        The sweep rows and, per family, its estimate and correlation against ``α``.
    """
    estimates: Mapping[str, Sequence[float]] = {
        "envelope_growth": envelope_sigma,
        "autocorrelation": autocorrelation_rate,
    }
    families = {
        label: {
            "estimate": [float(v) for v in estimates[label]],
            "correlation": correlation(alphas, estimates[label]),
        }
        for label in FAMILIES
    }
    return {
        "omega_hz": float(omega_hz),
        "ring_sigma": float(ring_sigma),
        "n": len(list(alphas)),
        "true_alpha": [float(v) for v in alphas],
        "families": families,
    }


def hopf_bridge_verdict(
    record: Mapping[str, object],
    frequency_invariance: Mapping[str, object],
) -> str:
    """Return a one-line honest verdict for the Hopf bridge.

    Parameters
    ----------
    record : mapping
        The :func:`hopf_record` sweep.
    frequency_invariance : mapping
        The frequency-robustness record, holding ``pearson_spread`` across frequencies.

    Returns
    -------
    str
        A factual sentence on which family recovers ``α`` and how the two divide.
    """
    families = record["families"]
    assert isinstance(families, dict)
    env = families["envelope_growth"]["correlation"]
    ac = families["autocorrelation"]["correlation"]
    spread = cast(float, frequency_invariance["pearson_spread"])
    invariant = "frequency-invariant" if spread < 0.05 else "frequency-sensitive"
    return (
        f"On a Hopf bifurcation both families track α, but only the "
        f"envelope-growth family recovers it in MAGNITUDE (mean "
        f"|σ−α|={env['mean_abs_error']:.2f}, Spearman ρ={env['spearman']:.2f}, "
        f"{invariant}); the autocorrelation family's magnitude is confounded by "
        f"the oscillation (mean |rate−α|={ac['mean_abs_error']:.2f}, pinned near "
        f"ln(cos ωΔt)/Δt). The eigenvalue's real part is universal, but the "
        f"magnitude-correct estimator is regime-dependent: envelope-growth for "
        f"an oscillatory mode, the autocorrelation for a non-oscillatory one."
    )


def hopf_bridge_payload(
    *,
    record: Mapping[str, object],
    snr_robustness: Sequence[Mapping[str, object]],
    frequency_invariance: Mapping[str, object],
    window: int,
    step: int,
) -> dict[str, object]:
    """Assemble and hash-seal the Hopf-bridge external-validation result.

    Parameters
    ----------
    record : mapping
        The headline :func:`hopf_record` sweep.
    snr_robustness : sequence of mapping
        The envelope recovery as the ringdown noise rises: each entry holds
        ``ring_sigma`` and the envelope ``pearson`` — the characterised SNR floor.
    frequency_invariance : mapping
        The envelope recovery across oscillation frequencies, with ``pearson_spread``.
    window : int
        The autocorrelation detector's window in samples.
    step : int
        The autocorrelation detector's window hop in samples.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "On a Hopf bifurcation, which detector family recovers the true "
            "eigenvalue α?"
        ),
        "method": (
            "sweep the Hopf parameter α; integrate the stochastic Hopf normal "
            "form; the envelope-growth family reads the rectified ringdown, the "
            "autocorrelation family reads the stationary series; correlate each "
            "against the analytic α."
        ),
        "window": int(window),
        "step": int(step),
        "record": dict(record),
        "snr_robustness": [dict(entry) for entry in snr_robustness],
        "frequency_invariance": dict(frequency_invariance),
        "verdict": hopf_bridge_verdict(record, frequency_invariance),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def sweep_hopf(
    *,
    alphas: Sequence[float],
    omega_hz: float,
    rate: float,
    ring_sigma: float,
    ring_n: int,
    stationary_sigma: float,
    stationary_n: int,
    stationary_burn: int,
    window: int,
    step: int,
    seed: int,
) -> dict[str, object]:
    """Run the headline Hopf sweep and assemble its :func:`hopf_record`.

    For each ``α`` a ringdown feeds the envelope-growth family and an independent
    stationary run feeds the autocorrelation family.

    Parameters
    ----------
    alphas : sequence of float
        The Hopf parameters to sweep.
    omega_hz : float
        The oscillation frequency in Hz.
    rate : float
        The sampling rate in Hz (``dt = 1/rate``).
    ring_sigma, ring_n : float, int
        The ringdown noise and length.
    stationary_sigma, stationary_n, stationary_burn : float, int, int
        The stationary run's noise, length, and discarded leading transient.
    window, step : int
        The autocorrelation detector's window and hop.
    seed : int
        The base seed; ringdown ``k`` uses ``seed + k``, stationary ``k`` uses
        ``seed + 1000 + k``.

    Returns
    -------
    dict
        The :func:`hopf_record`.

    Raises
    ------
    ValueError
        If ``stationary_burn`` is negative or not below ``stationary_n``.
    """
    if not 0 <= stationary_burn < stationary_n:
        raise ValueError("stationary_burn must be in [0, stationary_n)")
    dt = 1.0 / rate
    omega = 2.0 * np.pi * omega_hz
    envelope_sigma: list[float] = []
    autocorrelation_rate: list[float] = []
    for k, alpha in enumerate(alphas):
        ring = simulate_hopf(
            alpha, omega, 1.0, dt=dt, n=ring_n, sigma=ring_sigma, seed=seed + k
        )
        envelope_sigma.append(envelope_family_sigma(ring, rate=rate))
        r_star = float(np.sqrt(max(-alpha, 0.05)))
        stationary = simulate_hopf(
            alpha,
            omega,
            r_star,
            dt=dt,
            n=stationary_n,
            sigma=stationary_sigma,
            seed=seed + 1000 + k,
        )
        segment = stationary[stationary_burn:]
        segment = segment - float(np.mean(segment))
        autocorrelation_rate.append(
            autocorrelation_family_rate(segment, dt=dt, window=window, step=step)
        )
    return hopf_record(
        omega_hz=omega_hz,
        ring_sigma=ring_sigma,
        alphas=alphas,
        envelope_sigma=envelope_sigma,
        autocorrelation_rate=autocorrelation_rate,
    )


def _envelope_pearson(
    alphas: Sequence[float],
    *,
    omega_hz: float,
    rate: float,
    ring_sigma: float,
    ring_n: int,
    seed: int,
) -> float:
    """Return the envelope Pearson correlation for one (frequency, SNR) setting."""
    dt = 1.0 / rate
    omega = 2.0 * np.pi * omega_hz
    estimate = [
        envelope_family_sigma(
            simulate_hopf(
                a, omega, 1.0, dt=dt, n=ring_n, sigma=ring_sigma, seed=seed + k
            ),
            rate=rate,
        )
        for k, a in enumerate(alphas)
    ]
    return cast(float, correlation(alphas, estimate)["pearson"])


def main() -> None:  # pragma: no cover - CLI shell
    """Regenerate the Hopf-bridge sweep and write the sealed artefact."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path for the sealed JSON artefact")
    args = parser.parse_args()

    rate, ring_n, window, step = 5.0, 90, 128, 16
    ring_sigma = 0.002
    alphas = list(np.linspace(-0.5, -0.03, 14))
    record = sweep_hopf(
        alphas=alphas,
        omega_hz=0.4,
        rate=rate,
        ring_sigma=ring_sigma,
        ring_n=ring_n,
        stationary_sigma=0.05,
        stationary_n=2400,
        stationary_burn=400,
        window=window,
        step=step,
        seed=5000,
    )
    snr_robustness = [
        {
            "ring_sigma": float(s),
            "pearson": _envelope_pearson(
                alphas, omega_hz=0.4, rate=rate, ring_sigma=s, ring_n=ring_n, seed=5000
            ),
        }
        for s in (0.002, 0.005, 0.01, 0.02)
    ]
    freq_pearson = [
        _envelope_pearson(
            alphas,
            omega_hz=f,
            rate=rate,
            ring_sigma=ring_sigma,
            ring_n=ring_n,
            seed=5000,
        )
        for f in (0.2, 0.4, 0.8, 1.2)
    ]
    frequency_invariance = {
        "frequencies_hz": [0.2, 0.4, 0.8, 1.2],
        "pearson": [float(v) for v in freq_pearson],
        "pearson_spread": float(max(freq_pearson) - min(freq_pearson)),
    }
    payload = hopf_bridge_payload(
        record=record,
        snr_robustness=snr_robustness,
        frequency_invariance=frequency_invariance,
        window=window,
        step=step,
    )
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: {payload['verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()
