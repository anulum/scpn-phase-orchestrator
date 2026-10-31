# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Where each real compute owner's contracts run

"""Route each real compute owner's test cases to the CI profiles that have it.

The Python owner needs nothing and runs in every profile. The Rust owner
needs the compiled ``spo_kernel``. Its cases run wherever the kernel is
installed: the native branch profile and the FFI jobs. They skip, with a
reason, only where the kernel is genuinely not installed, as in the absent
branch profile. An installed but defective kernel is not skipped, so it
fails. The Mojo, Julia and Go owners need their toolchains. Their cases
carry the ``native_runtime`` marker, so only the jobs that provide that
toolchain select them.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.util import find_spec

import pytest
from _pytest.mark.structures import ParameterSet

KERNEL_INSTALLED = find_spec("spo_kernel") is not None
"""Whether the compiled ``spo_kernel`` distribution is installed here."""

NEEDS_KERNEL = pytest.mark.skipif(
    not KERNEL_INSTALLED, reason="the compiled spo_kernel is not installed here"
)
"""Skip a case only where the compiled kernel is genuinely absent."""

TOOLCHAIN_OWNERS = frozenset({"mojo", "julia", "go"})
"""Owners whose cases run only in the job that provides their toolchain."""


def owner_param(owner: str) -> ParameterSet:
    """Return one owner as a parameter routed to the profiles that have it.

    Parameters
    ----------
    owner : str
        Owner name: ``"python"``, ``"rust"``, ``"mojo"``, ``"julia"`` or
        ``"go"``.

    Returns
    -------
    ParameterSet
        The owner with no mark (Python), :data:`NEEDS_KERNEL` (Rust) or the
        ``native_runtime`` mark (a toolchain owner).

    Raises
    ------
    ValueError
        If ``owner`` is not one of the five known owners.
    """
    if owner == "python":
        return pytest.param(owner, id=owner)
    if owner == "rust":
        return pytest.param(owner, marks=NEEDS_KERNEL, id=owner)
    if owner in TOOLCHAIN_OWNERS:
        return pytest.param(owner, marks=pytest.mark.native_runtime, id=owner)
    raise ValueError(f"unknown compute owner: {owner!r}")


def owner_params(owners: Iterable[str]) -> list[ParameterSet]:
    """Return each owner routed by :func:`owner_param`, in the given order.

    Parameters
    ----------
    owners : Iterable[str]
        Owner names.

    Returns
    -------
    list[ParameterSet]
        One routed parameter per owner.
    """
    return [owner_param(owner) for owner in owners]


__all__ = [
    "KERNEL_INSTALLED",
    "NEEDS_KERNEL",
    "TOOLCHAIN_OWNERS",
    "owner_param",
    "owner_params",
]
