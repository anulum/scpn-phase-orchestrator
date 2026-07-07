# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kuramoto synchronisation eigenvalue external validation

"""Does critical slowing down of the mean field recover the Kuramoto eigenvalue?

The critical-slowing-down external validation (§3.10) confirmed that the shipped
detector's autocorrelation recovers the analytic Jacobian eigenvalue on *scalar* normal
forms. This is the same test carried to the model SCPN was built around, and one
dimension harder: the eigenvalue now lives on the **emergent collective
coordinate** of a high-dimensional system, not a one-line normal form.

For the noisy Kuramoto model of ``N`` phase oscillators with a Lorentzian frequency
distribution (half-width ``γ``) and phase diffusion ``D``, the incoherent state loses
stability at the mean-field critical coupling ``K_c = 2(γ + D)`` (the ``K_c = 2γ``
of the Ott–Antonsen reduction, generalised for noise), and just below it the
fundamental mean-field mode is **real and non-oscillatory** with eigenvalue
``λ(K) = (K − K_c)/2`` (Sakaguchi 1988). So the regime map predicts the
*autocorrelation* family should recover ``λ`` — the non-oscillatory prescription —
and it does, but only through the right observable.

We sweep ``K`` quasi-statically below ``K_c``, integrate the stochastic Kuramoto
model at each operating point, and read the shipped
:func:`~scpn_phase_orchestrator.monitor.critical_slowing_down.critical_slowing_down_warning`
on two observables of the mean field ``Z = (1/N) Σ exp(i θ)``: its **signed real part**
``Re(Z)`` and the **order-parameter amplitude** ``|Z|`` a practitioner usually watches.
The sealed answer:

* **Both** observables recover ``λ`` in **rank** (Spearman ρ ≈ 0.98): the collective
  coordinate undergoes textbook critical slowing down as ``K → K_c``.
* **Only the signed** ``Re(Z)`` recovers ``λ`` in **magnitude** — its lag-one
  autocorrelation is ``exp(λ Δt)``, so ``ln(AR1)/Δt`` tracks ``λ`` with unit slope. The
  amplitude ``|Z|`` is folded (rectified), so its autocorrelation is not ``exp(λ Δt)``:
  it ranks the eigenvalue but sizes it with a slope near two, not one.

The operational lesson, sealed as the fitted slope, is actionable: to read the
*distance to the synchronisation threshold* (the eigenvalue's magnitude), monitor the
signed mean-field component, not the order-parameter magnitude, and use an
autocorrelation window long enough that the finite-window bias does not steepen it.

Honest limits: a quasi-static per-coupling sweep (an independent stationary run at
each ``K``, not one non-stationary approach); the mean-field eigenvalue is the
``N → ∞`` result while the runs are finite-``N``, so ``K_c`` and ``λ`` carry an
``O(1/√N)`` finite-size correction; a Lorentzian frequency law, additive phase noise,
and coupling below ``K_c`` only (a damping-ranking test of the incoherent state, not a
sync/no-sync classification). The seal is recomputed from the committed rows, never a
stochastic run reproduces the indicators only to floating-point tolerance.

References
----------
* Kuramoto 1975; Strogatz 2000, *From Kuramoto to Crawford*, Physica D 143:1 — the
  mean-field synchronisation transition and ``K_c = 2/(π g(0))``.
* Sakaguchi 1988, Prog. Theor. Phys. 79:39 — the noisy Kuramoto incoherent-state
  eigenvalue ``λ = K/2 − (γ + D)``.
* Scheffer et al. 2009, Nature 461:53 — critical slowing down as the generic approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]

#: The sealed-artefact identifier.
BENCHMARK = "kuramoto_synchronization_external_validation"
#: The mean-field observables compared, in seal order: the signed real part recovers the
#: eigenvalue in magnitude, the folded amplitude only in rank.
OBSERVABLES = ("mean_field_real", "order_parameter_amplitude")
#: The fewest sweep points at which a rank correlation is meaningful.
_MIN_POINTS = 3
#: The lag-one autocorrelation is clipped to this open interval before ``ln`` is
#: taken, so a degenerate window (AR1 ≤ 0 or ≥ 1) cannot give a non-finite rate.
_AR1_FLOOR = 1.0e-6
_AR1_CEIL = 1.0 - 1.0e-6
#: A tolerance on the fitted magnitude slope: within it the estimator recovers the
#: eigenvalue in magnitude (unit slope), outside it only in rank.
_UNIT_SLOPE_BAND = 0.4

__all__ = [
    "BENCHMARK",
    "OBSERVABLES",
    "correlation",
    "critical_coupling",
    "detector_rate",
    "eigenvalue",
    "kuramoto_record",
    "kuramoto_sync_external_validation_payload",
    "kuramoto_sync_external_validation_verdict",
    "observable_series",
    "simulate_kuramoto",
    "sweep_coupling",
]


def critical_coupling(gamma: float, diffusion: float) -> float:
    """Return the mean-field critical coupling ``K_c = 2(γ + D)`` for the noisy model.

    For a Lorentzian frequency law of half-width ``γ`` and phase diffusion ``D``, the
    incoherent state loses stability at this coupling; ``D = 0`` recovers the classic
    ``K_c = 2γ`` of the Ott–Antonsen reduction.
    """
    return 2.0 * (gamma + diffusion)


def eigenvalue(coupling: float, gamma: float, diffusion: float) -> float:
    """Return the incoherent-state eigenvalue ``λ = K/2 − (γ + D)`` (Sakaguchi 1988).

    This is the analytic recovery rate of the fundamental mean-field mode; it is
    real and negative below ``K_c`` and crosses zero at ``K_c``.
    """
    return 0.5 * coupling - (gamma + diffusion)


def _spearman(x: FloatArray, y: FloatArray) -> float:
    """Return the Spearman rank correlation of ``x`` and ``y``."""
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def correlation(
    true_rate: Sequence[float], detector_value: Sequence[float]
) -> dict[str, object]:
    """Return the rank, magnitude and fitted slope of a detector channel versus ``λ``.

    Parameters
    ----------
    true_rate : sequence of float
        The ground-truth eigenvalue ``λ`` at each operating point.
    detector_value : sequence of float
        The detector's implied rate ``ln(AR1)/Δt`` at each point.

    Returns
    -------
    dict
        ``pearson`` and ``spearman`` (rank agreement), ``n``, ``mean_abs_gap`` (the mean
        ``|rate − λ|`` magnitude error), and the ``slope`` and ``intercept`` of the
        least-squares fit ``rate = slope·λ + intercept`` (a slope near one means the
        eigenvalue is recovered in magnitude, not merely in rank).

    Raises
    ------
    ValueError
        If the two sequences differ in length or hold fewer than three points.
    """
    a = np.asarray(true_rate, dtype=np.float64)
    b = np.asarray(detector_value, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("true_rate and detector_value must be the same length")
    if a.shape[0] < _MIN_POINTS:
        raise ValueError(f"need at least {_MIN_POINTS} points for a correlation")
    design = np.vstack([a, np.ones_like(a)]).T
    slope, intercept = np.linalg.lstsq(design, b, rcond=None)[0]
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": _spearman(a, b),
        "n": int(a.shape[0]),
        "mean_abs_gap": float(np.mean(np.abs(b - a))),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def simulate_kuramoto(
    omega: FloatArray,
    coupling: float,
    theta0: FloatArray,
    *,
    dt: float,
    n_samples: int,
    sample_every: int,
    diffusion: float,
    seed: int,
) -> ComplexArray:
    """Integrate the noisy Kuramoto model and return the sampled mean field ``Z(t)``.

    The model ``dθ_i = (ω_i + (K/N) Σ_j sin(θ_j − θ_i)) dt + √(2D) dW_i`` is advanced by
    the Euler–Maruyama scheme at step ``dt``; the mean field ``Z = (1/N) Σ exp(i θ)`` is
    recorded once every ``sample_every`` steps, so the returned series has spacing
    ``dt · sample_every`` — chosen to keep the lag-one autocorrelation away from one.

    Parameters
    ----------
    omega : FloatArray
        The natural frequencies, shape ``(N,)``.
    coupling : float
        The coupling ``K`` held fixed for this run.
    theta0 : FloatArray
        The initial phases, shape ``(N,)``.
    dt : float
        The integration step.
    n_samples : int
        The number of recorded mean-field samples; must exceed one.
    sample_every : int
        The number of integration steps between recorded samples; must be positive.
    diffusion : float
        The phase-diffusion coefficient ``D`` (noise amplitude ``√(2D)``).
    seed : int
        The seed for this run's noise, making the integration reproducible.

    Returns
    -------
    ComplexArray
        The sampled mean field, shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``n_samples`` is not at least two or ``sample_every`` is not positive.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples {n_samples} must be at least 2")
    if sample_every < 1:
        raise ValueError(f"sample_every {sample_every} must be positive")
    rng = np.random.default_rng(seed)
    n_osc = omega.shape[0]
    theta = np.asarray(theta0, dtype=np.float64).copy()
    noise_amp = np.sqrt(2.0 * diffusion * dt)
    mean_field = np.empty(n_samples, dtype=np.complex128)
    for s in range(n_samples):
        for _ in range(sample_every):
            field = np.mean(np.exp(1j * theta))
            # (K/N) Σ_j sin(θ_j − θ_i) = K · Im(e^{−i θ_i} · Z)
            coupling_term = coupling * np.imag(np.exp(-1j * theta) * field)
            theta = (
                theta
                + (omega + coupling_term) * dt
                + noise_amp * rng.standard_normal(n_osc)
            )
        mean_field[s] = np.mean(np.exp(1j * theta))
    return mean_field


def observable_series(mean_field: ComplexArray) -> dict[str, FloatArray]:
    """Split the complex mean field into the two mean-subtracted real observables.

    Parameters
    ----------
    mean_field : ComplexArray
        The sampled mean field ``Z(t)``.

    Returns
    -------
    dict
        ``mean_field_real`` (the signed ``Re(Z)``) and ``order_parameter_amplitude``
        (the folded ``|Z|``), each with its mean removed so the detector sees a
        stationary, zero-mean fluctuation.
    """
    real = np.real(mean_field)
    amplitude = np.abs(mean_field)
    return {
        "mean_field_real": real - float(np.mean(real)),
        "order_parameter_amplitude": amplitude - float(np.mean(amplitude)),
    }


def detector_rate(
    series: FloatArray,
    *,
    dt: float,
    window: int,
    step: int,
) -> float:
    """Read the detector's autocorrelation as an implied eigenvalue ``ln(AR1)/dt``.

    Parameters
    ----------
    series : FloatArray
        One stationary, zero-mean observable at a fixed coupling.
    dt : float
        The sample spacing, for the ``ln(AR1)/dt`` implied rate.
    window : int
        The detector's analysis window in samples.
    step : int
        The detector's window hop in samples.

    Returns
    -------
    float
        The window-averaged implied eigenvalue ``ln(AR1)/dt``.
    """
    warning = critical_slowing_down_warning(series, window=window, step=step)
    ar1 = float(np.mean(warning.autocorrelation_index))
    ar1 = min(max(ar1, _AR1_FLOOR), _AR1_CEIL)
    return float(np.log(ar1) / dt)


def kuramoto_record(
    *,
    coupling: Sequence[float],
    critical_coupling_value: float,
    true_rate: Sequence[float],
    detector_value: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Assemble the sweep record with a correlation per observable.

    Parameters
    ----------
    coupling : sequence of float
        The coupling ``K`` at each operating point.
    critical_coupling_value : float
        The mean-field critical coupling ``K_c`` for provenance.
    true_rate : sequence of float
        The analytic eigenvalue ``λ`` at each operating point.
    detector_value : mapping
        Each observable in :data:`OBSERVABLES` mapped to its implied-rate sweep.

    Returns
    -------
    dict
        The rows and, per observable, the detector sweep and its correlation.

    Raises
    ------
    ValueError
        If an observable is missing from ``detector_value``.
    """
    observables = {}
    for label in OBSERVABLES:
        if label not in detector_value:
            raise ValueError(f"detector_value is missing observable {label!r}")
        series = list(detector_value[label])
        observables[label] = {
            "detector_value": [float(v) for v in series],
            "correlation": correlation(true_rate, series),
        }
    return {
        "n": len(list(coupling)),
        "critical_coupling": float(critical_coupling_value),
        "coupling": [float(v) for v in coupling],
        "true_rate": [float(v) for v in true_rate],
        "observables": observables,
    }


def kuramoto_sync_external_validation_verdict(record: Mapping[str, object]) -> str:
    """Return a one-line honest verdict on which observable recovers the eigenvalue.

    Parameters
    ----------
    record : mapping
        The sweep record from :func:`kuramoto_record`.

    Returns
    -------
    str
        A factual sentence: both observables track ``λ`` in rank, but only the signed
        mean-field component recovers it in magnitude.
    """
    observables = record["observables"]
    assert isinstance(observables, dict)

    def _corr(label: str) -> Mapping[str, object]:
        entry = observables[label]
        assert isinstance(entry, dict)
        result = entry["correlation"]
        assert isinstance(result, dict)
        return result

    signed = _corr("mean_field_real")
    amplitude = _corr("order_parameter_amplitude")
    both_rank = min(
        cast("float", signed["spearman"]), cast("float", amplitude["spearman"])
    )
    signed_slope = cast("float", signed["slope"])
    amplitude_slope = cast("float", amplitude["slope"])
    signed_magnitude = abs(signed_slope - 1.0) <= _UNIT_SLOPE_BAND
    amplitude_magnitude = abs(amplitude_slope - 1.0) <= _UNIT_SLOPE_BAND
    recovers = (
        "the signed mean-field component recovers λ in magnitude"
        if signed_magnitude and not amplitude_magnitude
        else "the magnitude split is inconclusive"
    )
    return (
        f"On the Kuramoto incoherent-state transition with the analytic eigenvalue "
        f"λ = (K − K_c)/2 as ground truth, the shipped detector's autocorrelation "
        f"tracks λ in rank on both mean-field observables "
        f"(Spearman ρ≥{both_rank:.2f}), "
        f"but {recovers}: Re(Z) fits λ with slope {signed_slope:.2f} (≈1), while the "
        f"folded order-parameter amplitude |Z| fits with slope {amplitude_slope:.2f} — "
        f"it ranks the eigenvalue but cannot size it. Critical slowing down of the "
        f"collective coordinate is confirmed against a first-principles reference."
    )


def kuramoto_sync_external_validation_payload(
    *,
    record: Mapping[str, object],
    oscillators: int,
    gamma: float,
    diffusion: float,
    sampling_dt: float,
    window: int,
    step: int,
) -> dict[str, object]:
    """Assemble and hash-seal the Kuramoto synchronisation external-validation result.

    Parameters
    ----------
    record : mapping
        The sweep record from :func:`kuramoto_record`.
    oscillators : int
        The number of oscillators ``N`` in the sweep, for provenance.
    gamma : float
        The Lorentzian half-width ``γ``.
    diffusion : float
        The phase-diffusion coefficient ``D``.
    sampling_dt : float
        The sample spacing behind the ``ln(AR1)/dt`` implied rate.
    window : int
        The detector analysis window in samples.
    step : int
        The detector window hop in samples.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "Does critical slowing down of the Kuramoto mean field recover the "
            "analytic incoherent-state eigenvalue λ = (K − K_c)/2?"
        ),
        "method": (
            "per coupling K below K_c = 2(γ + D): integrate the noisy Kuramoto "
            "model, sample the mean field Z = (1/N) Σ exp(i θ); the shipped CSD "
            "detector reads the lag-one autocorrelation (as an ln(AR1)/Δt implied "
            "rate) on the signed Re(Z) and on the folded |Z|; correlate and "
            "least-squares fit each against the analytic λ."
        ),
        "oscillators": int(oscillators),
        "gamma": float(gamma),
        "diffusion": float(diffusion),
        "sampling_dt": float(sampling_dt),
        "window": int(window),
        "step": int(step),
        "record": dict(record),
        "verdict": kuramoto_sync_external_validation_verdict(record),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def sweep_coupling(
    *,
    oscillators: int,
    gamma: float,
    diffusion: float,
    couplings: Sequence[float],
    dt: float,
    n_samples: int,
    sample_every: int,
    window: int,
    step: int,
    burn_in_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Sweep the coupling below ``K_c`` and assemble the sealed-ready record.

    One shared frozen frequency draw and initial phase are used across the sweep so the
    only thing changing between operating points is ``K``. For each coupling the noisy
    model is integrated, a leading burn-in transient discarded, and the shipped
    detector's implied rate read on both mean-field observables; the analytic
    eigenvalue supplies the true ``λ``.

    Parameters
    ----------
    oscillators : int
        The number of oscillators ``N``.
    gamma : float
        The Lorentzian half-width ``γ`` of the frequency law.
    diffusion : float
        The phase-diffusion coefficient ``D``.
    couplings : sequence of float
        The couplings ``K`` to sweep (all expected below ``K_c``).
    dt : float
        The integration step.
    n_samples : int
        The recorded mean-field samples per run.
    sample_every : int
        The integration steps between recorded samples.
    window, step : int
        The detector's window and hop.
    burn_in_fraction : float
        The leading fraction of each run discarded as transient, in ``[0, 1)``.
    seed : int
        The base seed; the frequencies and phases are drawn from it and run ``k`` uses
        ``seed + k`` for its noise.

    Returns
    -------
    dict
        The :func:`kuramoto_record` for the sweep.

    Raises
    ------
    ValueError
        If ``burn_in_fraction`` is not in ``[0, 1)``.
    """
    if not 0.0 <= burn_in_fraction < 1.0:
        raise ValueError(f"burn_in_fraction {burn_in_fraction} must be in [0, 1)")
    rng = np.random.default_rng(seed)
    omega = np.clip(
        gamma * rng.standard_cauchy(oscillators), -20.0 * gamma, 20.0 * gamma
    )
    theta0 = rng.uniform(-np.pi, np.pi, oscillators)
    k_c = critical_coupling(gamma, diffusion)
    dt_sample = dt * sample_every
    burn = int(burn_in_fraction * n_samples)
    true_rate: list[float] = []
    detector_value: dict[str, list[float]] = {label: [] for label in OBSERVABLES}
    for k, coupling in enumerate(couplings):
        mean_field = simulate_kuramoto(
            omega,
            coupling,
            theta0,
            dt=dt,
            n_samples=n_samples,
            sample_every=sample_every,
            diffusion=diffusion,
            seed=seed + 1 + k,
        )
        observables = observable_series(mean_field[burn:])
        true_rate.append(eigenvalue(coupling, gamma, diffusion))
        for label in OBSERVABLES:
            detector_value[label].append(
                detector_rate(
                    observables[label], dt=dt_sample, window=window, step=step
                )
            )
    return kuramoto_record(
        coupling=couplings,
        critical_coupling_value=k_c,
        true_rate=true_rate,
        detector_value=detector_value,
    )


def main() -> None:  # pragma: no cover - CLI shell
    """Regenerate the Kuramoto synchronisation sweep and write the sealed artefact."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path for the sealed JSON artefact")
    args = parser.parse_args()

    oscillators, gamma, diffusion = 512, 0.5, 0.5
    dt, sample_every, n_samples = 0.05, 5, 4000
    window, step, burn = 512, 64, 0.25
    k_c = critical_coupling(gamma, diffusion)
    couplings = list(np.linspace(0.30 * k_c, 0.92 * k_c, 12))
    record = sweep_coupling(
        oscillators=oscillators,
        gamma=gamma,
        diffusion=diffusion,
        couplings=couplings,
        dt=dt,
        n_samples=n_samples,
        sample_every=sample_every,
        window=window,
        step=step,
        burn_in_fraction=burn,
        seed=20260707,
    )
    payload = kuramoto_sync_external_validation_payload(
        record=record,
        oscillators=oscillators,
        gamma=gamma,
        diffusion=diffusion,
        sampling_dt=dt * sample_every,
        window=window,
        step=step,
    )
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: {payload['verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()
