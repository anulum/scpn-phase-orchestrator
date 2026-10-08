# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JuliaCall boundary type declarations

"""Declare the installed JuliaCall exception boundary and dynamic Main module."""

Main: object

class JuliaError(Exception):
    """Represent the actual JuliaCall native exception and optional backtrace."""

    def __init__(self, exception: object, backtrace: object | None = None) -> None:
        """Wrap the native Julia exception without replacing its identity.

        Parameters
        ----------
        exception : object
            Actual Julia exception carried by PythonCall.
        backtrace : object or None
            Native Julia backtrace, if supplied.
        """
        ...

    @property
    def exception(self) -> object:
        """Return the original native exception.

        Returns
        -------
        object
            Actual Julia exception used by Main.isa for precise translation.
        """
        ...

    @property
    def backtrace(self) -> object | None:
        """Return the native backtrace when present.

        Returns
        -------
        object or None
            JuliaCall backtrace associated with the original exception.
        """
        ...
