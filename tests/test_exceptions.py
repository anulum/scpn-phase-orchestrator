# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Exception tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.exceptions import (
    AuditError,
    BindingError,
    EngineError,
    ExtractorError,
    PolicyError,
    SPOError,
    ValidationError,
)


@pytest.mark.parametrize(
    "exc_cls,bases",
    [
        (SPOError, (Exception,)),
        (BindingError, (SPOError, ValueError)),
        (ValidationError, (SPOError, ValueError)),
        (ExtractorError, (SPOError, RuntimeError)),
        (EngineError, (SPOError, RuntimeError)),
        (PolicyError, (SPOError, ValueError)),
        (AuditError, (SPOError, RuntimeError)),
    ],
)
def test_exception_hierarchy(exc_cls, bases):
    for base in bases:
        assert issubclass(exc_cls, base)


def test_spo_error_catches_all_subtypes():
    for cls in (
        BindingError,
        ValidationError,
        ExtractorError,
        EngineError,
        PolicyError,
        AuditError,
    ):
        with pytest.raises(SPOError):
            raise cls("test")


def test_exception_message_preserved():
    msg = "spec parse failed"
    exc = BindingError(msg)
    assert str(exc) == msg
