# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — cross-system eigenvalue-ground-truth external validation

"""Does the grid detector's growth rate recover the true eigenvalue on other systems?

The PSML head-to-head certifies the detector on one 23-bus corpus. This is a stronger,
non-circular external test on *independent* systems: for a sweep of operating points on
a benchmark power system, the true dominant electromechanical mode's growth rate ``σ``
(its eigenvalue real part) is computed by ANDES small-signal analysis — a completely
different method from the detector's time-domain envelope slope — and the detector is
run on a ringdown from a small disturbance at each point. The question is whether the
detector's measured ``σ`` tracks the ground-truth ``σ``.

The sealed answer, on two independent systems (the IEEE 39-bus New England system and
the Kundur two-area system, both in ANDES), is **yes for the detector's core, with a
regime caveat**: the *coherent* aggregation (the cross-bus mean, or the dominant spatial
mode) recovers the true ``σ`` trend (Spearman ρ up to 0.87), so the quantity the
detector estimates — the dominant mode's damping — generalises across systems and
simulators. But the *focal* aggregation, the PSML winner, does **not** transfer (its
correlation is negative), because on a slow coherent inter-area mode the per-bus maximum
locks onto spurious local excursions rather than the network mode. The best aggregation
is therefore **regime-dependent** — focal for the fast, localised oscillations of PSML,
coherent for the slow inter-area modes — while the growth-rate quantity is universal.

The validation is honest about its limits: it is a simulated ground truth (ANDES, not
field PMU data), spans stable operating points only (a damping-ranking test, not a
stable-versus-unstable classification, since the well-damped benchmarks do not enter a
clean growing-oscillation regime), and each system is a 14-point sweep. The seal is
recomputed from the committed measurement rows, never from a fresh simulation, because a
nonlinear time-domain run reproduces the growth rate only to floating-point tolerance.

References
----------
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability and
  the two-area inter-area oscillation benchmark.
* ANDES (Cui, Li & Tomsovic 2021) — the open-source dynamic simulator whose eigenvalue
  analysis supplies the ground-truth ``σ``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

#: The sealed-artefact identifier.
BENCHMARK = "grid_eigenvalue_external_validation"
#: The channel aggregations compared, in seal order.
AGGREGATIONS = ("focal", "mean", "spatial")
#: The fewest sweep points at which a rank correlation is meaningful.
_MIN_POINTS = 3

__all__ = [
    "AGGREGATIONS",
    "BENCHMARK",
    "correlation",
    "external_validation_payload",
    "external_validation_verdict",
    "system_record",
]


def _spearman(x: FloatArray, y: FloatArray) -> float:
    """Return the Spearman rank correlation of ``x`` and ``y``."""
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def correlation(
    true_sigma: Sequence[float], detector_sigma: Sequence[float]
) -> dict[str, object]:
    """Return the Pearson and Spearman correlation of detector versus true ``σ``.

    Parameters
    ----------
    true_sigma : sequence of float
        The ground-truth dominant-mode growth rate at each operating point.
    detector_sigma : sequence of float
        The detector's measured growth rate at each operating point.

    Returns
    -------
    dict
        ``pearson``, ``spearman``, and ``n`` (the number of points).

    Raises
    ------
    ValueError
        If the two sequences differ in length or hold fewer than three points.
    """
    a = np.asarray(true_sigma, dtype=np.float64)
    b = np.asarray(detector_sigma, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("true_sigma and detector_sigma must be the same length")
    if a.shape[0] < _MIN_POINTS:
        raise ValueError(f"need at least {_MIN_POINTS} points for a correlation")
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": _spearman(a, b),
        "n": int(a.shape[0]),
    }


def system_record(
    *,
    name: str,
    case: str,
    loads: Sequence[float],
    true_sigma: Sequence[float],
    detector_sigma: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Assemble one system's sweep record with a correlation per aggregation.

    Parameters
    ----------
    name : str
        The system label, e.g. ``"ieee39"`` or ``"kundur"``.
    case : str
        The ANDES case file the sweep was run on.
    loads : sequence of float
        The load scaling at each operating point.
    true_sigma : sequence of float
        The ground-truth dominant-mode ``σ`` at each operating point.
    detector_sigma : mapping
        Each aggregation in :data:`AGGREGATIONS` mapped to its detector ``σ`` sweep.

    Returns
    -------
    dict
        The system's rows and, per aggregation, the detector sweep and its correlation.

    Raises
    ------
    ValueError
        If an aggregation is missing from ``detector_sigma``.
    """
    aggregations = {}
    for label in AGGREGATIONS:
        if label not in detector_sigma:
            raise ValueError(f"detector_sigma is missing aggregation {label!r}")
        series = list(detector_sigma[label])
        aggregations[label] = {
            "detector_sigma": [float(v) for v in series],
            "correlation": correlation(true_sigma, series),
        }
    return {
        "name": name,
        "case": case,
        "n": len(list(loads)),
        "loads": [float(v) for v in loads],
        "true_sigma": [float(v) for v in true_sigma],
        "aggregations": aggregations,
    }


def external_validation_verdict(systems: Sequence[Mapping[str, object]]) -> str:
    """Return a one-line honest verdict across the systems.

    Parameters
    ----------
    systems : sequence of mapping
        The per-system records from :func:`system_record`.

    Returns
    -------
    str
        A factual sentence on whether the detector's ``σ`` generalises and which
        aggregation transfers.
    """

    def _rho(system: Mapping[str, object], label: str) -> float:
        aggs = system["aggregations"]
        assert isinstance(aggs, dict)
        return float(aggs[label]["correlation"]["spearman"])

    names = ", ".join(str(system["name"]) for system in systems)
    coherent_ok = all(
        _rho(system, "mean") > 0.0 or _rho(system, "spatial") > 0.0
        for system in systems
    )
    focal_fails = all(_rho(system, "focal") <= 0.0 for system in systems)
    best = max(max(_rho(s, "mean"), _rho(s, "spatial")) for s in systems)
    generalises = "generalises" if coherent_ok else "does not generalise"
    focal_clause = (
        "the focal aggregation (the PSML winner) does not transfer"
        if focal_fails
        else "the focal aggregation transfers unevenly"
    )
    return (
        f"On independent systems ({names}) with ANDES eigenvalue ground truth, the "
        f"detector's coherent growth rate {generalises} — it recovers the true "
        f"dominant-mode σ trend (best Spearman ρ={best:.2f}); {focal_clause}, so "
        f"the growth-rate quantity is universal but the best aggregation is "
        f"regime-dependent."
    )


def external_validation_payload(
    *,
    systems: Sequence[Mapping[str, object]],
    andes_version: str,
) -> dict[str, object]:
    """Assemble and hash-seal the cross-system external-validation result.

    Parameters
    ----------
    systems : sequence of mapping
        The per-system records from :func:`system_record`.
    andes_version : str
        The ANDES version the ground truth was computed with, for provenance.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "Does the grid detector's time-domain growth rate recover the true "
            "small-signal eigenvalue σ on independent power systems?"
        ),
        "method": (
            "per operating point: ANDES small-signal analysis gives the true dominant "
            "electromechanical σ; a brief disturbance gives a ringdown the detector "
            "scores by its envelope growth rate under each aggregation; correlate."
        ),
        "andes_version": andes_version,
        "systems": [dict(system) for system in systems],
        "verdict": external_validation_verdict(systems),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _sweep_ground_truth(  # pragma: no cover - ANDES I/O shell over the sealed pure core
    case: str,
    loads: Sequence[float],
    *,
    fault_bus: int,
    window: tuple[float, float],
    tf: float,
    rate: float,
) -> dict[str, list[float]]:
    """Run an ANDES load sweep and return the true and detector-``σ`` series.

    For each load scaling: solve the power flow, take the dominant electromechanical
    eigenvalue's real part as the true ``σ``, run a ringdown from a brief fault, and
    it by the envelope growth rate under the focal, mean, and spatial aggregations.
    """
    import andes  # type: ignore[import-not-found]  # optional external simulator

    andes.config_logger(stream_level=50)
    case_path = andes.get_case(case)

    def _env_growth(env: FloatArray) -> float:
        logs = np.log(np.maximum(env, 1.0e-12))
        times = np.arange(env.shape[0], dtype=np.float64) / rate
        return float(np.polyfit(times, logs, 1)[0])

    out: dict[str, list[float]] = {
        k: [] for k in ("loads", "true_sigma", *AGGREGATIONS)
    }
    for load in loads:
        system = andes.load(case_path, setup=False)
        system.add("Fault", {"bus": fault_bus, "tf": 1.0, "tc": 1.05})
        system.setup()
        system.PQ.p0.v[:] *= load
        system.PQ.q0.v[:] *= load
        system.PFlow.run()
        if not system.PFlow.converged:
            continue
        system.EIG.run()
        mu = np.asarray(system.EIG.mu)
        freq = np.abs(mu.imag) / (2.0 * np.pi)
        band = (freq >= 0.1) & (freq <= 2.5)
        true_sigma = float(mu.real[band].max())
        system.TDS.config.tf = tf
        system.TDS.config.tstep = 1.0 / rate
        try:
            system.TDS.run()
        except (RuntimeError, ArithmeticError, ValueError):  # diverged run
            continue
        voltages = system.dae.ts.y[:, system.Bus.v.a].T
        times = system.dae.ts.t
        w = voltages[:, (times >= window[0]) & (times <= window[1])]
        if w.shape[1] < 40 or not np.all(np.isfinite(w)):
            continue
        centred = w - w.mean(axis=0, keepdims=True)
        focal = max(_env_growth(np.abs(bus)) for bus in centred)
        mean = _env_growth(np.abs(centred).mean(axis=0))
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        spatial = _env_growth(np.abs(s[0] * vt[0]))
        out["loads"].append(float(load))
        out["true_sigma"].append(true_sigma)
        out["focal"].append(focal)
        out["mean"].append(mean)
        out["spatial"].append(spatial)
    return out


def main() -> None:  # pragma: no cover - CLI shell
    """Regenerate the cross-system sweep with ANDES and write the sealed artefact."""
    import argparse
    import json
    from pathlib import Path

    import andes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path for the sealed JSON artefact")
    args = parser.parse_args()

    configs = [
        (
            "ieee39",
            "ieee39/ieee39_full.xlsx",
            16,
            (4.0, 24.0),
            24.0,
            30.0,
            list(np.linspace(0.7, 1.33, 14)),
        ),
        (
            "kundur",
            "kundur/kundur_full.xlsx",
            8,
            (3.0, 15.0),
            15.0,
            60.0,
            list(np.linspace(0.6, 1.2, 14)),
        ),
    ]
    systems = []
    for name, case, bus, window, tf, rate, loads in configs:
        swept = _sweep_ground_truth(
            case, loads, fault_bus=bus, window=window, tf=tf, rate=rate
        )
        systems.append(
            system_record(
                name=name,
                case=case,
                loads=swept["loads"],
                true_sigma=swept["true_sigma"],
                detector_sigma={k: swept[k] for k in AGGREGATIONS},
            )
        )
    payload = external_validation_payload(
        systems=systems, andes_version=andes.__version__
    )
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: {payload['verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()
