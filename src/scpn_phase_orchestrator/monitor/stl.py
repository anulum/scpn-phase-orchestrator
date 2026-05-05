# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime STL monitor

"""Signal Temporal Logic monitor backed by rtamt.

rtamt is an optional dependency: ``pip install rtamt``
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["STLMonitor", "STLTraceResult", "HAS_RTAMT"]

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import rtamt

    HAS_RTAMT = True
except ImportError:
    rtamt = None
    HAS_RTAMT = False


_SIMPLE_SPEC_RE = re.compile(r"^(always|eventually)\s*\((.*)\)\s*$")
_PREDICATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|==)\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class STLTraceResult:
    """Robustness summary for an STL monitor evaluation."""

    spec: str
    robustness: float
    satisfied: bool
    backend: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable STL audit payload."""
        return {
            "spec": self.spec,
            "robustness": self.robustness,
            "satisfied": self.satisfied,
            "backend": self.backend,
        }


class STLMonitor:
    """Evaluate STL specifications against numeric traces.

    Parameters
    ----------
    spec : str
        An rtamt STL specification string, e.g.
        ``"always (sync_error <= 0.3)"``.
    """

    # IEC 62443 / Kuramoto safety: order-parameter must stay above threshold
    SYNC_THRESHOLD = "always (R >= 0.3)"
    # Coupling gain bounded to prevent instability; Kuramoto 1984
    COUPLING_BOUND = "always (K <= 10.0)"

    def __init__(self, spec: str) -> None:
        self._spec_str = spec
        self._simple = _parse_simple_spec(spec)
        self._stl = rtamt.StlDiscreteTimeSpecification() if rtamt is not None else None
        self._parsed = False

    def evaluate(self, trace: dict[str, list[float]]) -> float:
        """Return the robustness value of *spec* over *trace*.

        A positive value means the specification is satisfied; negative
        means violated.  The magnitude indicates how far from the boundary.
        """
        if not trace:
            raise ValueError("trace must contain at least one signal")

        lengths = {len(v) for v in trace.values()}
        if len(lengths) != 1:
            raise ValueError("all signals in trace must have equal length")

        length = lengths.pop()
        if length == 0:
            raise ValueError("trace signals must be non-empty")

        if self._simple is not None:
            return _evaluate_simple(self._simple, trace)

        if self._stl is None:
            raise ImportError(
                "rtamt is required for this STL syntax. Install: pip install rtamt"
            )

        if not self._parsed:
            for name in trace:
                self._stl.declare_var(name, "float")
            self._stl.spec = self._spec_str
            self._stl.parse()
            self._parsed = True

        # rtamt discrete-time offline: flat lists per signal + 'time' key
        datasets: dict[str, list[float]] = {}
        for name, values in trace.items():
            datasets[name] = [float(v) for v in values]
        if "time" not in datasets:
            datasets["time"] = [float(t) for t in range(length)]

        robustness = self._stl.evaluate(datasets)
        # rtamt returns [[time, robustness], ...]; min is worst-case
        if isinstance(robustness, list) and robustness:
            return float(min(r[1] for r in robustness))
        return float(robustness)

    def evaluate_result(self, trace: dict[str, list[float]]) -> STLTraceResult:
        """Evaluate and return robustness plus audit metadata."""
        robustness = self.evaluate(trace)
        backend = "builtin" if self._simple is not None else "rtamt"
        return STLTraceResult(
            spec=self._spec_str,
            robustness=robustness,
            satisfied=robustness >= 0.0,
            backend=backend,
        )


def _parse_simple_spec(spec: str) -> tuple[str, list[tuple[str, str, float]]] | None:
    match = _SIMPLE_SPEC_RE.match(spec.strip())
    if match is None:
        return None
    temporal_op = match.group(1)
    predicates: list[tuple[str, str, float]] = []
    for raw_predicate in re.split(r"\s+(?:and|&&)\s+", match.group(2)):
        predicate_match = _PREDICATE_RE.match(raw_predicate)
        if predicate_match is None:
            return None
        signal, op, threshold = predicate_match.groups()
        predicates.append((signal, op, float(threshold)))
    return temporal_op, predicates


def _evaluate_simple(
    parsed: tuple[str, list[tuple[str, str, float]]],
    trace: dict[str, list[float]],
) -> float:
    temporal_op, predicates = parsed
    per_predicate = [
        _predicate_robustness(signal, op, threshold, trace)
        for signal, op, threshold in predicates
    ]
    pointwise = np.min(np.vstack(per_predicate), axis=0)
    if temporal_op == "always":
        return float(np.min(pointwise))
    if temporal_op == "eventually":
        return float(np.max(pointwise))
    raise ValueError(f"unsupported STL temporal operator {temporal_op!r}")


def _predicate_robustness(
    signal: str,
    op: str,
    threshold: float,
    trace: dict[str, list[float]],
) -> np.ndarray:
    if signal not in trace:
        raise ValueError(f"trace missing signal {signal!r}")
    values = np.asarray(trace[signal], dtype=np.float64)
    if op in {">=", ">"}:
        return values - threshold
    if op in {"<=", "<"}:
        return threshold - values
    if op == "==":
        return -np.abs(values - threshold)
    raise ValueError(f"unsupported STL comparison operator {op!r}")
