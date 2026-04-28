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

__all__ = ["STLMonitor", "HAS_RTAMT"]

try:
    import rtamt

    HAS_RTAMT = True
except Exception:  # rtamt's antlr4 dep breaks on Python >=3.12
    rtamt = None
    HAS_RTAMT = False


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
        if rtamt is None:
            raise ImportError(
                "rtamt is required for STL monitoring. Install: pip install rtamt"
            )
        self._spec_str = spec
        self._stl = rtamt.StlDiscreteTimeSpecification()
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
