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

from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.coupling import (
    CouplingInferenceConfig,
    auto_coupling_estimation,
    infer_coupling_from_timeseries,
)


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
