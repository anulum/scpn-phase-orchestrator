# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — causal coupling inference tests

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.coupling import (
    CouplingInferenceConfig,
    auto_coupling_estimation,
    infer_coupling_from_timeseries,
)
from scpn_phase_orchestrator.runtime.cli import main


def _directed_phase_series(n_steps: int = 240) -> np.ndarray:
    """Return phases where oscillator 0 drives oscillator 1."""

    time = np.arange(n_steps, dtype=np.float64)
    driver = 0.13 * time + 0.7 * np.sin(0.05 * time)
    driven = np.empty_like(driver)
    driven[:2] = driver[:2] + 0.4
    for idx in range(2, n_steps):
        driven[idx] = 0.72 * driven[idx - 1] + 0.28 * driver[idx - 1] + 0.03
    independent = 1.7 + 0.091 * time + 0.35 * np.sin(0.113 * time + 0.8)
    return np.vstack([driver, driven, independent]) % (2.0 * np.pi)


def test_transfer_entropy_inference_recovers_directed_support() -> None:
    result = infer_coupling_from_timeseries(
        _directed_phase_series(),
        config=CouplingInferenceConfig(
            method="transfer_entropy",
            n_bins=8,
            threshold_quantile=0.65,
            normalisation="max",
        ),
    )

    assert result.method == "transfer_entropy"
    assert result.package == "auto-coupling-estimation"
    assert result.knm.shape == (3, 3)
    assert result.score_matrix.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(result.knm), 0.0)
    assert result.knm[0, 1] > result.knm[1, 0]
    assert result.support_mask[0, 1]
    assert result.edge_count >= 1
    assert 0.0 < result.density <= 1.0


def test_inference_result_audit_record_is_json_safe() -> None:
    result = auto_coupling_estimation(_directed_phase_series(), n_bins=6)

    record = result.to_audit_record()
    encoded = json.dumps(record, sort_keys=True)

    assert "auto-coupling-estimation" in encoded
    assert record["shape"] == [3, 240]
    assert record["method"] == "transfer_entropy"
    assert record["score_kind"] == "transfer_entropy"
    assert len(record["knm"]) == 3
    assert record["diagnostics"]["edge_count"] == result.edge_count


def test_inference_rejects_invalid_inputs_and_unsupported_methods() -> None:
    with pytest.raises(ValueError, match="finite 2-D"):
        infer_coupling_from_timeseries(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="boolean"):
        infer_coupling_from_timeseries(np.array([[True, False, True, False]] * 2))
    with pytest.raises(ValueError, match="finite 2-D"):
        infer_coupling_from_timeseries(np.array([[0.1 + 0.1j, 0.2, 0.3, 0.4]] * 2))
    with pytest.raises(ValueError, match="at least 4 timesteps"):
        infer_coupling_from_timeseries(np.ones((2, 3)))
    with pytest.raises(NotImplementedError, match="notears"):
        infer_coupling_from_timeseries(
            _directed_phase_series(),
            config=CouplingInferenceConfig(method="notears"),
        )


def test_cli_auto_coupling_estimation_outputs_json(tmp_path) -> None:
    csv_path = tmp_path / "phases.csv"
    np.savetxt(csv_path, _directed_phase_series().T, delimiter=",")

    result = CliRunner().invoke(
        main,
        [
            "auto-coupling-estimation",
            str(csv_path),
            "--orientation",
            "time-by-oscillator",
            "--n-bins",
            "8",
            "--threshold-quantile",
            "0.65",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "transfer_entropy"
    assert payload["shape"] == [3, 240]
    assert payload["diagnostics"]["edge_count"] >= 1
    assert payload["knm"][0][1] > payload["knm"][1][0]


def test_transfer_entropy_backend_negative_scores_fail_closed(monkeypatch) -> None:
    import scpn_phase_orchestrator.coupling.infer as infer_module

    def negative_scores(series, *, n_bins):
        return np.array([[0.0, -0.25], [0.1, 0.0]], dtype=np.float64)

    monkeypatch.setattr(infer_module, "transfer_entropy_matrix", negative_scores)

    with pytest.raises(RuntimeError, match="negative scores"):
        infer_coupling_from_timeseries(np.ones((2, 8)))


def test_transfer_entropy_backend_self_scores_fail_closed(monkeypatch) -> None:
    import scpn_phase_orchestrator.coupling.infer as infer_module

    def self_scores(series, *, n_bins):
        return np.array([[0.2, 0.1], [0.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(infer_module, "transfer_entropy_matrix", self_scores)

    with pytest.raises(RuntimeError, match="self scores"):
        infer_coupling_from_timeseries(np.ones((2, 8)))


@pytest.mark.parametrize(
    "config",
    [
        CouplingInferenceConfig(threshold_quantile=True),
        CouplingInferenceConfig(threshold_quantile=False),
        CouplingInferenceConfig(threshold_absolute=True),
        CouplingInferenceConfig(threshold_absolute=np.nan),
        CouplingInferenceConfig(min_timesteps=4.5),  # type: ignore[arg-type]
    ],
)
def test_inference_rejects_invalid_threshold_and_timestep_config(config) -> None:
    with pytest.raises((TypeError, ValueError), match="threshold|min_timesteps"):
        infer_coupling_from_timeseries(_directed_phase_series(), config=config)


def test_result_density_is_zero_for_a_single_oscillator() -> None:
    from scpn_phase_orchestrator.coupling.infer import CouplingInferenceResult

    result = CouplingInferenceResult(
        knm=np.zeros((1, 1)),
        score_matrix=np.zeros((1, 1)),
        support_mask=np.zeros((1, 1), dtype=bool),
        method="transfer_entropy",
        score_kind="transfer_entropy",
        n_bins=8,
        threshold=0.0,
        normalisation="max",
        shape=(1, 1),
    )
    assert result.density == 0.0
    assert result.edge_count == 0


def test_to_upde_knm_returns_the_transpose() -> None:
    result = infer_coupling_from_timeseries(_directed_phase_series())
    np.testing.assert_allclose(result.to_upde_knm(), result.knm.T)


@pytest.mark.parametrize(
    ("series", "match"),
    [
        (np.ones((3, 240), dtype=bool), "boolean"),
        (np.empty((0, 3), dtype=bool), "boolean"),
        (np.array([["a", "b", "c"], ["d", "e", "f"]]), "finite 2-D array"),
        (np.ones((1, 240)), "at least 2 oscillators"),
        (np.full((3, 240), np.inf), "finite 2-D array"),
    ],
)
def test_inference_rejects_invalid_phase_series(series, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        infer_coupling_from_timeseries(series)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (CouplingInferenceConfig(method="bogus"), "method must be one of"),  # type: ignore[arg-type]
        (CouplingInferenceConfig(n_bins=2.5), "n_bins must be an integer"),  # type: ignore[arg-type]
        (CouplingInferenceConfig(n_bins=1), "n_bins must be greater"),
        (CouplingInferenceConfig(threshold_quantile=1.5), "must lie in"),
        (CouplingInferenceConfig(threshold_absolute=-1.0), "must be non-negative"),
        (CouplingInferenceConfig(normalisation="bogus"), "must be one of"),  # type: ignore[arg-type]
        (CouplingInferenceConfig(min_timesteps=2), "min_timesteps must be at least 4"),
    ],
)
def test_inference_rejects_invalid_config(config, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        infer_coupling_from_timeseries(_directed_phase_series(), config=config)


def test_absolute_threshold_and_no_normalisation_paths() -> None:
    result = infer_coupling_from_timeseries(
        _directed_phase_series(),
        config=CouplingInferenceConfig(threshold_absolute=0.001, normalisation="none"),
    )
    assert result.threshold == pytest.approx(0.001)
    assert result.normalisation == "none"


def test_all_zero_scores_yield_zero_threshold_and_empty_coupling(monkeypatch) -> None:
    from scpn_phase_orchestrator.coupling import infer as infer_mod

    monkeypatch.setattr(
        infer_mod,
        "transfer_entropy_matrix",
        lambda series, n_bins: np.zeros((series.shape[0], series.shape[0])),
    )
    result = infer_coupling_from_timeseries(
        _directed_phase_series(),
        config=CouplingInferenceConfig(threshold_quantile=None),
    )
    assert result.threshold == 0.0
    assert not np.any(result.knm)


def test_backend_unexpected_shape_is_rejected(monkeypatch) -> None:
    from scpn_phase_orchestrator.coupling import infer as infer_mod

    monkeypatch.setattr(
        infer_mod,
        "transfer_entropy_matrix",
        lambda series, n_bins: np.zeros((series.shape[0] + 1, series.shape[0])),
    )
    with pytest.raises(RuntimeError, match="unexpected matrix shape"):
        infer_coupling_from_timeseries(_directed_phase_series())


def test_backend_non_finite_scores_are_rejected(monkeypatch) -> None:
    from scpn_phase_orchestrator.coupling import infer as infer_mod

    monkeypatch.setattr(
        infer_mod,
        "transfer_entropy_matrix",
        lambda series, n_bins: np.full((series.shape[0], series.shape[0]), np.inf),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        infer_coupling_from_timeseries(_directed_phase_series())
