# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Canonical NumPy phase projection

"""Project NumPy UPDE phases onto the half-open floating-point torus."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI


def wrap_phases(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return wrapped phases with a canonical positive zero.

    Parameters
    ----------
    phases : NDArray[np.float64]
        Phase state in radians; the caller retains ownership.

    Returns
    -------
    NDArray[np.float64]
        A new array in the half-open interval for finite inputs. A negative
        remainder that rounds to the upper endpoint maps to positive zero;
        other interior values are preserved. Nonfinite values remain nonfinite
        for the caller's numerical-output validation.
    """
    wrapped: NDArray[np.float64] = np.remainder(phases, TWO_PI)
    wrapped[(wrapped >= TWO_PI) | (wrapped == 0.0)] = 0.0
    return wrapped
