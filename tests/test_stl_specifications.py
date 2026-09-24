# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Curated STL specification catalogue tests

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.monitor.stl import (
    PHASE_FIELD_SPECIFICATIONS,
    PhaseFieldSpecification,
    STLMonitor,
    STLTraceResult,
    phase_field_specification,
    phase_field_specification_names,
)
from scpn_phase_orchestrator.monitor.stl.specifications import (
    _render_threshold,
    _require_non_empty,
)


class TestRenderThreshold:
    def test_integral_threshold_renders_with_decimal_point(self) -> None:
        assert _render_threshold(10.0) == "10.0"
        assert _render_threshold(1.0) == "1.0"

    def test_fractional_threshold_renders_with_significant_digits(self) -> None:
        assert _render_threshold(0.3) == "0.3"
        assert _render_threshold(0.5) == "0.5"

    def test_irrational_threshold_renders_the_exact_float(self) -> None:
        rendered = _render_threshold(math.pi / 2.0)
        assert rendered == "1.5707963267948966"
        assert float(rendered) == math.pi / 2.0
        # The rendering must remain parseable by the builtin predicate grammar.
        assert (
            STLMonitor(f"always (phase_lag <= {rendered})").evaluate(
                {"phase_lag": [0.1, 0.2]}
            )
            > 0
        )


class TestRequireNonEmpty:
    def test_accepts_non_empty_string(self) -> None:
        _require_non_empty("R", "signal")  # must not raise

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _require_non_empty("   ", "name")

    def test_rejects_non_string(self) -> None:
        # Direct call exercises the ``not isinstance`` guard the frozen
        # dataclass relies on; the runtime type is deliberately wrong here.
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _require_non_empty(123, "name")  # type: ignore[arg-type] # wrong type on purpose


class TestPhaseFieldSpecificationValidation:
    def _valid_kwargs(self) -> dict[str, object]:
        return {
            "name": "demo",
            "signal": "R",
            "temporal_op": "always",
            "comparison": ">=",
            "threshold": 0.3,
            "rationale": "demo rationale",
        }

    def test_valid_specification_constructs(self) -> None:
        spec = PhaseFieldSpecification(**self._valid_kwargs())  # type: ignore[arg-type]
        assert spec.severity == "soft"

    def test_rejects_empty_name(self) -> None:
        kwargs = self._valid_kwargs() | {"name": ""}
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_empty_signal(self) -> None:
        kwargs = self._valid_kwargs() | {"signal": ""}
        with pytest.raises(ValueError, match="signal must be a non-empty string"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_empty_rationale(self) -> None:
        kwargs = self._valid_kwargs() | {"rationale": ""}
        with pytest.raises(ValueError, match="rationale must be a non-empty string"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_unknown_temporal_op(self) -> None:
        kwargs = self._valid_kwargs() | {"temporal_op": "until"}
        with pytest.raises(ValueError, match="temporal_op must be one of"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_unknown_comparison(self) -> None:
        kwargs = self._valid_kwargs() | {"comparison": "!="}
        with pytest.raises(ValueError, match="comparison must be one of"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_unknown_severity(self) -> None:
        kwargs = self._valid_kwargs() | {"severity": "critical"}
        with pytest.raises(ValueError, match="severity must be one of"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]

    def test_rejects_non_finite_threshold(self) -> None:
        kwargs = self._valid_kwargs() | {"threshold": math.inf}
        with pytest.raises(ValueError, match="threshold must be finite"):
            PhaseFieldSpecification(**kwargs)  # type: ignore[arg-type]


class TestPhaseFieldSpecificationBehaviour:
    def test_spec_renders_builtin_formula(self) -> None:
        spec = PhaseFieldSpecification(
            name="demo",
            signal="R",
            temporal_op="always",
            comparison=">=",
            threshold=0.3,
            rationale="demo",
        )
        assert spec.spec == "always (R >= 0.3)"

    def test_eventually_operator_renders(self) -> None:
        spec = PhaseFieldSpecification(
            name="demo",
            signal="R",
            temporal_op="eventually",
            comparison=">=",
            threshold=0.8,
            rationale="demo",
        )
        assert spec.spec == "eventually (R >= 0.8)"

    def test_monitor_returns_bound_stl_monitor(self) -> None:
        spec = phase_field_specification("order_parameter_floor")
        monitor = spec.monitor()
        assert isinstance(monitor, STLMonitor)
        assert monitor.evaluate({"R": [0.9, 0.8, 0.7]}) > 0

    def test_evaluate_returns_builtin_backed_result(self) -> None:
        spec = phase_field_specification("order_parameter_floor")
        result = spec.evaluate({"R": [0.9, 0.8, 0.7]})
        assert isinstance(result, STLTraceResult)
        assert result.backend == "builtin"
        assert result.satisfied

    def test_evaluate_flags_violation(self) -> None:
        spec = phase_field_specification("order_parameter_floor")
        result = spec.evaluate({"R": [0.9, 0.1, 0.7]})
        assert not result.satisfied
        assert result.robustness < 0

    def test_evaluate_missing_signal_raises(self) -> None:
        spec = phase_field_specification("order_parameter_floor")
        with pytest.raises(ValueError, match="missing signal"):
            spec.evaluate({"K": [1.0, 2.0]})


class TestCatalogue:
    def test_names_are_the_five_curated_keys_in_order(self) -> None:
        assert phase_field_specification_names() == (
            "order_parameter_floor",
            "coupling_gain_ceiling",
            "chimera_index_ceiling",
            "phase_lag_bound",
            "winding_stability",
        )

    def test_lookup_returns_matching_specification(self) -> None:
        spec = phase_field_specification("winding_stability")
        assert spec.signal == "winding_number"
        assert spec.spec == "always (winding_number <= 1.0)"

    def test_lookup_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="unknown phase-field specification"):
            phase_field_specification("does_not_exist")

    @pytest.mark.parametrize("spec", PHASE_FIELD_SPECIFICATIONS, ids=lambda s: s.name)
    def test_every_catalogue_spec_is_builtin_evaluable(
        self, spec: PhaseFieldSpecification
    ) -> None:
        # Each curated spec must evaluate on the builtin backend (no rtamt),
        # so a single benign sample must produce a finite builtin robustness.
        result = spec.evaluate({spec.signal: [0.05, 0.05, 0.05]})
        assert result.backend == "builtin"
        assert math.isfinite(result.robustness)

    def test_severities_are_within_the_permitted_tiers(self) -> None:
        assert {spec.severity for spec in PHASE_FIELD_SPECIFICATIONS} <= {
            "soft",
            "hard",
        }

    def test_phase_lag_bound_uses_sakaguchi_frustration_limit(self) -> None:
        spec = phase_field_specification("phase_lag_bound")
        assert spec.threshold == pytest.approx(math.pi / 2.0)


class TestSpecificationCarriesTheStoredThreshold:
    """The rendered formula evaluates exactly the threshold the object holds."""

    @pytest.mark.parametrize("name", phase_field_specification_names())
    def test_catalogue_formula_round_trips_its_threshold(self, name: str) -> None:
        from scpn_phase_orchestrator.monitor.stl.monitor import _parse_simple_spec

        spec = phase_field_specification(name)
        parsed = _parse_simple_spec(spec.spec)
        assert parsed is not None
        assert parsed[1] == [(spec.signal, spec.comparison, spec.threshold)]

    @pytest.mark.parametrize(
        "threshold", [0.1 + 0.2, 1e-20, 1e22, -2.5e-7, 10, math.pi]
    )
    def test_threshold_round_trips_through_the_formula(self, threshold: float) -> None:
        spec = PhaseFieldSpecification(
            name="probe",
            signal="R",
            temporal_op="always",
            comparison="<=",
            threshold=threshold,
            rationale="round-trip probe",
        )
        result = spec.evaluate({"R": [threshold]})
        assert result.backend == "builtin"
        assert result.robustness == 0.0
        assert result.satisfied

    def test_phase_lag_bound_is_exactly_half_pi(self) -> None:
        spec = phase_field_specification("phase_lag_bound")
        assert spec.spec == "always (phase_lag <= 1.5707963267948966)"
        assert spec.evaluate({"phase_lag": [math.pi / 2.0]}).satisfied


class TestSpecificationFieldValidation:
    @pytest.mark.parametrize("signal", ["R value", "R)", "1R", "R-1", "R.x"])
    def test_signal_must_be_a_predicate_identifier(self, signal: str) -> None:
        with pytest.raises(ValueError, match="signal must be an identifier"):
            PhaseFieldSpecification(
                name="probe",
                signal=signal,
                temporal_op="always",
                comparison=">=",
                threshold=0.3,
                rationale="probe",
            )

    @pytest.mark.parametrize(
        "threshold", [True, False, "0.3", None, float("nan"), float("inf")]
    )
    def test_threshold_must_be_a_finite_real(self, threshold: object) -> None:
        with pytest.raises(
            ValueError, match="threshold must be finite and a real number"
        ):
            PhaseFieldSpecification(
                name="probe",
                signal="R",
                temporal_op="always",
                comparison=">=",
                threshold=threshold,  # type: ignore[arg-type]
                rationale="probe",
            )
