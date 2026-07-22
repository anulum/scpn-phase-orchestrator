# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime STL monitor and trace evaluation

"""rtamt-backed STL monitor, trace results, and predicate robustness evaluation."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import rtamt

    HAS_RTAMT = True
except ImportError:
    rtamt = None
    HAS_RTAMT = False


_SIMPLE_SPEC_RE = re.compile(r"^(always|eventually)\s*\((.*)\)\s*$")


_BOUNDED_SPEC_RE = re.compile(
    r"^(always|eventually)\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\((.*)\)\s*$"
)


_PREDICATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|==)\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


# Parsed unbounded builtin specification: temporal operator + atomic predicates.
_ParsedSimple: TypeAlias = tuple[str, list[tuple[str, str, float]]]

# Parsed bounded builtin specification: the above plus a discrete step window.
_ParsedBounded: TypeAlias = tuple[str, list[tuple[str, str, float]], tuple[int, int]]


@dataclass(frozen=True)
class STLTraceResult:
    """Robustness summary for an STL monitor evaluation."""

    spec: str
    robustness: float
    satisfied: bool
    backend: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable STL audit payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable STL audit payload.
        """
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
        An STL specification string, e.g. ``"always (sync_error <= 0.3)"``.
        The builtin backend evaluates the unbounded operators
        ``always``/``eventually`` and their bounded forms
        ``always[a,b]``/``eventually[a,b]`` (integer discrete step window,
        ``0 <= a <= b``) over a conjunction of atomic predicates; ``until``,
        nesting, and other syntax require the optional ``rtamt`` backend.
    """

    # IEC 62443 / Kuramoto safety: order-parameter must stay above threshold
    SYNC_THRESHOLD = "always (R >= 0.3)"
    # Coupling gain bounded to prevent instability; Kuramoto 1984
    COUPLING_BOUND = "always (K <= 10.0)"

    def __init__(self, spec: str) -> None:
        self._spec_str = spec
        self._simple = _parse_simple_spec(spec)
        self._bounded = _parse_bounded_spec(spec) if self._simple is None else None
        self._stl = rtamt.StlDiscreteTimeSpecification() if rtamt is not None else None
        self._parsed = False

    @property
    def _is_builtin(self) -> bool:
        """Return whether this spec evaluates on the builtin robustness backend."""
        return self._simple is not None or self._bounded is not None

    def evaluate(self, trace: dict[str, list[float]]) -> float:
        """Return the robustness value of *spec* over *trace*.

        A positive value means the specification is satisfied; negative
        means violated.  The magnitude indicates how far from the boundary.

        Parameters
        ----------
        trace : dict[str, list[float]]
            Signal trace keyed by variable name, each a list of floats.

        Returns
        -------
        float
            The robustness value of the specification over the trace.

        Raises
        ------
        ImportError
            If the rtamt STL backend is not installed.
        """
        _validate_trace(trace)
        length = len(next(iter(trace.values())))

        if self._simple is not None:
            return _evaluate_simple(self._simple, trace)

        if self._bounded is not None:
            return _evaluate_bounded(self._bounded, trace)

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
        for name in trace:
            datasets[name] = _trace_signal_array(name, trace).tolist()
        if "time" not in datasets:
            datasets["time"] = [float(t) for t in range(length)]

        robustness = self._stl.evaluate(datasets)
        # rtamt returns [[time, robustness], ...]; min is worst-case
        if isinstance(robustness, list) and robustness:
            return float(min(r[1] for r in robustness))
        return float(robustness)

    def evaluate_result(self, trace: dict[str, list[float]]) -> STLTraceResult:
        """Evaluate and return robustness plus audit metadata.

        Parameters
        ----------
        trace : dict[str, list[float]]
            Signal trace keyed by variable name, each a list of floats.

        Returns
        -------
        STLTraceResult
            The robustness value plus audit metadata.
        """
        robustness = self.evaluate(trace)
        backend = "builtin" if self._is_builtin else "rtamt"
        return STLTraceResult(
            spec=self._spec_str,
            robustness=robustness,
            satisfied=robustness >= 0.0,
            backend=backend,
        )


def _parse_predicates(body: str) -> list[tuple[str, str, float]] | None:
    """Parse a conjunction of atomic predicates, or ``None`` if unsupported."""
    predicates: list[tuple[str, str, float]] = []
    for raw_predicate in re.split(r"\s+(?:and|&&)\s+", body):
        predicate_match = _PREDICATE_RE.match(raw_predicate)
        if predicate_match is None:
            return None
        signal, op, threshold = predicate_match.groups()
        predicates.append((signal, op, float(threshold)))
    return predicates


def _parse_simple_spec(spec: str) -> _ParsedSimple | None:
    """Parse an unbounded ``always``/``eventually`` STL specification string."""
    match = _SIMPLE_SPEC_RE.match(spec.strip())
    if match is None:
        return None
    predicates = _parse_predicates(match.group(2))
    if predicates is None:
        return None
    return match.group(1), predicates


def _parse_bounded_spec(spec: str) -> _ParsedBounded | None:
    """Parse a bounded ``always[a,b]``/``eventually[a,b]`` specification string.

    Returns the temporal operator, atomic predicates, and the integer discrete
    step window ``(a, b)`` with ``0 <= a <= b``. Any other syntax (``until``,
    nesting, inverted bounds, unsupported predicates) returns ``None`` so
    evaluation falls through to the ``rtamt`` backend.
    """
    match = _BOUNDED_SPEC_RE.match(spec.strip())
    if match is None:
        return None
    lower, upper = int(match.group(2)), int(match.group(3))
    if lower > upper:
        return None
    predicates = _parse_predicates(match.group(4))
    if predicates is None:
        return None
    return match.group(1), predicates, (lower, upper)


def _reduce_temporal(temporal_op: str, window: FloatArray) -> float:
    """Reduce a robustness window by the temporal operator.

    An empty window yields ``+inf`` for ``always`` and ``-inf`` for
    ``eventually`` — the vacuous quantifier over no samples, matching the
    ``rtamt`` offline convention for windows clamped past the trace end.
    """
    if temporal_op == "always":
        return float(np.min(window)) if window.size else float("inf")
    if temporal_op == "eventually":
        return float(np.max(window)) if window.size else float("-inf")
    raise ValueError(f"unsupported STL temporal operator {temporal_op!r}")


def _evaluate_simple(
    parsed: _ParsedSimple,
    trace: dict[str, list[float]],
) -> float:
    """Evaluate a parsed unbounded STL specification over a trace."""
    temporal_op, predicates = parsed
    return _reduce_temporal(temporal_op, _pointwise_robustness(predicates, trace))


def _evaluate_bounded(
    parsed: _ParsedBounded,
    trace: dict[str, list[float]],
) -> float:
    """Evaluate a parsed bounded STL specification at the initial time.

    The temporal reduction is restricted to the discrete step window ``[a, b]``
    (clamped to the trace end, matching the ``rtamt`` offline convention), so
    the returned value equals the robustness ``rtamt`` reports at time zero.
    """
    temporal_op, predicates, (lower, upper) = parsed
    pointwise = _pointwise_robustness(predicates, trace)
    return _reduce_temporal(temporal_op, pointwise[lower : upper + 1])


def _validate_trace(trace: dict[str, list[float]]) -> None:
    """Return the validated signal trace, else raise ``ValueError``."""
    if not trace:
        raise ValueError("trace must contain at least one signal")

    lengths = {len(v) for v in trace.values()}
    if len(lengths) != 1:
        raise ValueError("all signals in trace must have equal length")

    length = lengths.pop()
    if length == 0:
        raise ValueError("trace signals must be non-empty")
    for signal in trace:
        _trace_signal_array(signal, trace)


def _trace_signal_array(signal: str, trace: dict[str, list[float]]) -> FloatArray:
    """Return the named signal as a finite float array from the trace."""
    if signal not in trace:
        raise ValueError(f"trace missing signal {signal!r}")

    raw = np.asarray(trace[signal], dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"trace signal {signal!r} must be a 1-D numeric signal")
    if _contains_boolean_alias(raw):
        raise ValueError(f"trace signal {signal!r} must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError(f"trace signal {signal!r} must contain real-valued samples")

    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trace signal {signal!r} must be numeric") from exc

    if not np.all(np.isfinite(values)):
        raise ValueError(f"trace signal {signal!r} must contain only finite values")
    return np.ascontiguousarray(values, dtype=np.float64)


def _contains_boolean_alias(raw: NDArray[np.generic]) -> bool:
    """Return whether the value contains any boolean alias."""
    if raw.dtype == np.bool_:
        return True
    if raw.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)
    return False


def _contains_complex_alias(raw: NDArray[np.generic]) -> bool:
    """Return whether the value contains any complex-number alias."""
    if np.iscomplexobj(raw):
        return True
    if raw.dtype == object:
        return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)
    return False


def _pointwise_robustness(
    predicates: list[tuple[str, str, float]],
    trace: dict[str, list[float]],
) -> FloatArray:
    """Return the pointwise STL robustness signal over the trace."""
    per_predicate = [
        _predicate_robustness(signal, op, threshold, trace)
        for signal, op, threshold in predicates
    ]
    pointwise: FloatArray = np.asarray(
        np.min(np.vstack(per_predicate), axis=0),
        dtype=np.float64,
    )
    return pointwise


def _predicate_robustness(
    signal: str,
    op: str,
    threshold: float,
    trace: dict[str, list[float]],
) -> FloatArray:
    """Return the robustness signal of an STL predicate over the trace."""
    values = _trace_signal_array(signal, trace)
    if op in {">=", ">"}:
        return values - threshold
    if op in {"<=", "<"}:
        return threshold - values
    if op == "==":
        return -np.abs(values - threshold)
    raise ValueError(f"unsupported STL comparison operator {op!r}")


def _format_threshold(threshold: float) -> str:
    """Return a stable string rendering of a predicate threshold."""
    if threshold.is_integer():
        return str(int(threshold))
    return f"{threshold:g}"


def _require_non_empty(value: str, name: str) -> None:
    """Return ``value`` if it is a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
