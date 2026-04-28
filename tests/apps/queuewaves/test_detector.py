# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves detector tests

from __future__ import annotations

from scpn_phase_orchestrator.apps.queuewaves.config import ThresholdConfig
from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PipelineSnapshot,
    ServiceSnapshot,
)


def _snap(
    r_bad: float = 0.2,
    plv: list[list[float]] | None = None,
    imprints: dict[str, float] | None = None,
    tick: int = 1,
) -> PipelineSnapshot:
    svcs = [
        ServiceSnapshot(
            "svc-a", "micro", 0.0, 1.0, 1.0, (imprints or {}).get("svc-a", 0.0)
        ),
        ServiceSnapshot(
            "svc-b", "micro", 1.0, 1.0, 1.0, (imprints or {}).get("svc-b", 0.0)
        ),
        ServiceSnapshot(
            "tput", "macro", 2.0, 1.0, 1.0, (imprints or {}).get("tput", 0.0)
        ),
    ]
    return PipelineSnapshot(
        tick=tick,
        timestamp=0.0,
        r_good=0.8,
        r_bad=r_bad,
        regime="nominal",
        services=svcs,
        plv_matrix=plv or [[0.0, 0.0], [0.0, 0.0]],
        layer_states=[{"R": 0.5, "psi": 0.0}],
        boundary_violations=[],
        actions=[],
    )


def test_no_anomalies_below_thresholds() -> None:
    det = AnomalyDetector(ThresholdConfig())
    anomalies = det.detect(_snap(r_bad=0.3))
    assert anomalies == []


def test_retry_storm_warning() -> None:
    det = AnomalyDetector(ThresholdConfig(r_bad_warn=0.50))
    anomalies = det.detect(_snap(r_bad=0.55))
    assert len(anomalies) == 1
    assert anomalies[0].type == "retry_storm_forming"
    assert anomalies[0].severity == "warning"


def test_retry_storm_critical() -> None:
    det = AnomalyDetector(ThresholdConfig(r_bad_critical=0.70))
    anomalies = det.detect(_snap(r_bad=0.75))
    assert len(anomalies) == 1
    assert anomalies[0].severity == "critical"


def test_cascade_propagation() -> None:
    det = AnomalyDetector(ThresholdConfig(plv_cascade=0.80))
    plv = [[0.0, 0.90], [0.90, 0.0]]
    anomalies = det.detect(_snap(plv=plv))
    cascade = [a for a in anomalies if a.type == "cascade_propagation"]
    assert len(cascade) == 1
    assert cascade[0].value == 0.90


def test_chronic_degradation() -> None:
    det = AnomalyDetector(ThresholdConfig(imprint_chronic=1.0))
    anomalies = det.detect(_snap(imprints={"svc-a": 1.5}))
    chronic = [a for a in anomalies if a.type == "chronic_degradation"]
    assert len(chronic) == 1
    assert chronic[0].service == "svc-a"


def test_multiple_anomaly_types() -> None:
    det = AnomalyDetector(
        ThresholdConfig(
            r_bad_warn=0.40,
            plv_cascade=0.80,
            imprint_chronic=1.0,
        )
    )
    plv = [[0.0, 0.90], [0.90, 0.0]]
    anomalies = det.detect(_snap(r_bad=0.45, plv=plv, imprints={"svc-b": 2.0}))
    types = {a.type for a in anomalies}
    assert "retry_storm_forming" in types
    assert "cascade_propagation" in types
    assert "chronic_degradation" in types


def test_critical_supersedes_warning() -> None:
    det = AnomalyDetector(ThresholdConfig(r_bad_warn=0.40, r_bad_critical=0.60))
    anomalies = det.detect(_snap(r_bad=0.65))
    assert len(anomalies) == 1
    assert anomalies[0].severity == "critical"
