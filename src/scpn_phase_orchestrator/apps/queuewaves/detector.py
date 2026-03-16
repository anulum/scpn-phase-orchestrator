# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves anomaly detector

from __future__ import annotations

from dataclasses import dataclass

from scpn_phase_orchestrator.apps.queuewaves.config import ThresholdConfig
from scpn_phase_orchestrator.apps.queuewaves.pipeline import PipelineSnapshot

__all__ = ["Anomaly", "AnomalyDetector"]


@dataclass(frozen=True)
class Anomaly:
    type: str  # retry_storm_forming | cascade_propagation | chronic_degradation
    severity: str  # warning | critical
    service: str
    value: float
    threshold: float
    tick: int
    message: str


def _storm_msg(r_bad: float, threshold: float, label: str) -> str:
    return f"R_bad={r_bad:.3f} > {threshold} — {label}"


class AnomalyDetector:
    """Detects three anomaly types from pipeline snapshots."""

    def __init__(self, thresholds: ThresholdConfig):
        self._t = thresholds

    def detect(self, snap: PipelineSnapshot) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        anomalies.extend(self._check_retry_storm(snap))
        anomalies.extend(self._check_cascade(snap))
        anomalies.extend(self._check_chronic(snap))
        return anomalies

    def _check_retry_storm(self, snap: PipelineSnapshot) -> list[Anomaly]:
        r_bad = snap.r_bad
        if r_bad > self._t.r_bad_critical:
            return [
                Anomaly(
                    type="retry_storm_forming",
                    severity="critical",
                    service="cluster",
                    value=r_bad,
                    threshold=self._t.r_bad_critical,
                    tick=snap.tick,
                    message=_storm_msg(
                        r_bad, self._t.r_bad_critical, "retry storm imminent"
                    ),
                )
            ]
        if r_bad > self._t.r_bad_warn:
            return [
                Anomaly(
                    type="retry_storm_forming",
                    severity="warning",
                    service="cluster",
                    value=r_bad,
                    threshold=self._t.r_bad_warn,
                    tick=snap.tick,
                    message=_storm_msg(
                        r_bad, self._t.r_bad_warn, "retry storm forming"
                    ),
                )
            ]
        return []

    def _check_cascade(self, snap: PipelineSnapshot) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        plv = snap.plv_matrix
        n = len(plv)
        for i in range(n):
            for j in range(i + 1, n):
                val = plv[i][j]
                if val > self._t.plv_cascade:
                    msg = f"PLV(L{i},L{j})={val:.3f} > {self._t.plv_cascade} — cascade"
                    anomalies.append(
                        Anomaly(
                            type="cascade_propagation",
                            severity="warning",
                            service=f"layer_{i}_to_{j}",
                            value=val,
                            threshold=self._t.plv_cascade,
                            tick=snap.tick,
                            message=msg,
                        )
                    )
        return anomalies

    def _check_chronic(self, snap: PipelineSnapshot) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        for svc in snap.services:
            if svc.imprint > self._t.imprint_chronic:
                msg = (
                    f"{svc.name} imprint={svc.imprint:.3f}"
                    f" > {self._t.imprint_chronic} — chronic"
                )
                anomalies.append(
                    Anomaly(
                        type="chronic_degradation",
                        severity="warning",
                        service=svc.name,
                        value=svc.imprint,
                        threshold=self._t.imprint_chronic,
                        tick=snap.tick,
                        message=msg,
                    )
                )
        return anomalies
