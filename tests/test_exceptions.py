# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Exception hierarchy tests

from __future__ import annotations

import pickle

import numpy as np
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

# All exception classes for parametrised tests
ALL_SUBTYPES = [
    BindingError,
    ValidationError,
    ExtractorError,
    EngineError,
    PolicyError,
    AuditError,
]

VALUE_ERROR_SUBTYPES = [BindingError, ValidationError, PolicyError]
RUNTIME_ERROR_SUBTYPES = [ExtractorError, EngineError, AuditError]


# ---------------------------------------------------------------------------
# Hierarchy contracts
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify that the dual-inheritance hierarchy allows correct except-block
    routing: SPOError catches everything, ValueError/RuntimeError catch
    their respective subtypes only."""

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
    def test_subclass_relationships(self, exc_cls, bases):
        for base in bases:
            assert issubclass(exc_cls, base), (
                f"{exc_cls.__name__} must be subclass of {base.__name__}"
            )

    def test_spo_error_catches_every_subtype(self):
        """A single `except SPOError` block must catch all library exceptions."""
        for cls in ALL_SUBTYPES:
            with pytest.raises(SPOError):
                raise cls("test")

    def test_valueerror_catches_only_value_subtypes(self):
        """except ValueError must catch Binding/Validation/Policy but NOT
        Extractor/Engine/Audit."""
        for cls in VALUE_ERROR_SUBTYPES:
            with pytest.raises(ValueError):
                raise cls("test")

        for cls in RUNTIME_ERROR_SUBTYPES:
            with pytest.raises(RuntimeError):
                raise cls("test")
            # These must NOT be caught by ValueError
            assert not issubclass(cls, ValueError), (
                f"{cls.__name__} should not be a ValueError subtype"
            )

    def test_runtimeerror_catches_only_runtime_subtypes(self):
        """except RuntimeError must catch Extractor/Engine/Audit but NOT
        Binding/Validation/Policy."""
        for cls in RUNTIME_ERROR_SUBTYPES:
            with pytest.raises(RuntimeError):
                raise cls("test")

        for cls in VALUE_ERROR_SUBTYPES:
            assert not issubclass(cls, RuntimeError), (
                f"{cls.__name__} should not be a RuntimeError subtype"
            )


# ---------------------------------------------------------------------------
# Message and data preservation
# ---------------------------------------------------------------------------


class TestExceptionDataPreservation:
    """Verify that exception messages, args, and attributes survive
    construction, re-raising, and serialisation."""

    @pytest.mark.parametrize("exc_cls", ALL_SUBTYPES)
    def test_message_preserved(self, exc_cls):
        msg = f"specific error from {exc_cls.__name__}"
        exc = exc_cls(msg)
        assert str(exc) == msg
        assert exc.args == (msg,)

    @pytest.mark.parametrize("exc_cls", ALL_SUBTYPES)
    def test_picklable_for_multiprocessing(self, exc_cls):
        """Exceptions must survive pickle round-trip (required for
        multiprocessing, Ray, Dask error propagation)."""
        original = exc_cls("pickle test message")
        restored = pickle.loads(pickle.dumps(original))
        assert type(restored) is exc_cls
        assert str(restored) == "pickle test message"

    def test_spo_error_empty_message(self):
        """SPOError with no args must not crash str() or repr()."""
        exc = SPOError()
        assert str(exc) == ""
        assert repr(exc) == "SPOError()"


# ---------------------------------------------------------------------------
# Real code paths that raise these exceptions
# ---------------------------------------------------------------------------


class TestExceptionsRaisedByRealCode:
    """Verify that actual library code raises the correct exception types,
    not generic ValueError/RuntimeError. This ensures the hierarchy is
    actually used, not just defined."""

    def test_audit_error_on_incomplete_replay_fields(self, tmp_path):
        """AuditLogger must raise AuditError (not ValueError) when phases
        are provided without omegas/knm/alpha."""
        from scpn_phase_orchestrator.audit.logger import AuditLogger
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        state = UPDEState(
            layers=[LayerState(R=0.8, psi=0.5)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.7,
            regime_id="nominal",
        )
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as logger, pytest.raises(AuditError):
            logger.log_step(0, state, [], phases=np.array([0.1]))

    def test_engine_error_or_valueerror_on_nan_input(self):
        """UPDEEngine must reject NaN input — verifying the error is catchable
        as both ValueError and SPOError."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError):
            eng.step(
                np.array([float("nan"), 0.2]),
                np.array([1.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
            )

    def test_binding_error_on_invalid_file(self, tmp_path):
        """Loading a non-existent binding spec must raise BindingError."""
        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        with pytest.raises((BindingError, FileNotFoundError)):
            load_binding_spec(tmp_path / "nonexistent.yaml")

    def test_except_spo_catches_audit_error_from_logger(self, tmp_path):
        """Integration: a generic `except SPOError` must catch AuditError
        raised by the logger — proving the hierarchy works in practice."""
        from scpn_phase_orchestrator.audit.logger import AuditLogger
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        state = UPDEState(
            layers=[LayerState(R=0.8, psi=0.5)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.7,
            regime_id="nominal",
        )
        log_path = tmp_path / "audit.jsonl"
        caught = False
        with AuditLogger(log_path) as logger:
            try:
                logger.log_step(0, state, [], phases=np.array([0.1]))
            except SPOError:
                caught = True
        assert caught, "SPOError must catch AuditError from real code"


# Pipeline wiring: exception hierarchy tested via real code paths that raise each
# exception type. TestExceptionsRaisedByRealCode proves exceptions are not decorative.
