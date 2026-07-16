# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared non-negative scalar validation

"""Canonical non-negative scalar validators shared across the package.

One implementation of the non-negative real / integer parameter checks
that were previously duplicated (with drifting keyword names and error
texts) across ~20 modules. Per-call error context is carried by the
mandatory ``name`` keyword; the error text after the name is canonical
and stable, so callers keep meaningful, greppable failures without
re-implementing the check.

Booleans (including ``numpy.bool_``) are rejected explicitly: ``bool``
is an ``Integral`` subclass and silently coercing ``True`` to ``1``
has repeatedly masked caller bugs. NumPy scalar types need no special
cases — ``numpy.float64`` registers as :class:`numbers.Real` and
``numpy.integer`` as :class:`numbers.Integral`, while ``numpy.bool_``
registers as neither.

Scope
-----
This module is the canonical home for the free-standing non-negative
scalar checks: new code, and any module whose validator matches these
semantics, should import from here rather than re-implement the pattern.
The K-QUAL1 consolidation migrated the nine exact-match, free-standing
defs (predictive, topology, morphogenetic, multiverse, reward,
fusion_core_bridge, modbus_tls).

Some modules deliberately keep a local validation family rather than
delegate here, and this is by design, not oversight. Each such check is
one leaf of a cohesive ``_positive_*`` / ``_non_negative_*`` pair sharing
a private type-check helper, and it carries a contract these canonical
``ValueError`` helpers do not model: ``actuation/koopman_mpc`` raises
:class:`TypeError` on a non-real input, ``supervisor/policy_rules`` raises
its ``PolicyRuleError``, and ``adapters/hardware_io`` emits domain text
("address must be >= 0"). Folding those into this primitive would break
the exception contracts their tests pin, or split a module's validation
vocabulary, so they are retained.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

__all__ = [
    "non_negative_int",
    "non_negative_real",
]


def non_negative_real(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise.

    Parameters
    ----------
    value : object
        Candidate scalar. Booleans are rejected; any
        :class:`numbers.Real` instance is accepted.
    name : str
        Parameter name used as the error context.

    Returns
    -------
    float
        The validated value converted to ``float``.

    Raises
    ------
    ValueError
        If ``value`` is boolean, not a real number, not finite, or
        negative.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def non_negative_int(value: object, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise.

    Parameters
    ----------
    value : object
        Candidate scalar. Booleans are rejected; any
        :class:`numbers.Integral` instance is accepted.
    name : str
        Parameter name used as the error context.

    Returns
    -------
    int
        The validated value converted to ``int``.

    Raises
    ------
    ValueError
        If ``value`` is boolean, not an integer, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
