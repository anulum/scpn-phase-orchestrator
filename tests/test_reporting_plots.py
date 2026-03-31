# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reporting plot tests

"""Tests for CoherencePlot matplotlib visualisations."""

from __future__ import annotations

import importlib.util

import pytest

from scpn_phase_orchestrator.reporting.plots import CoherencePlot

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _make_log(n_steps: int = 20, n_layers: int = 3) -> list[dict]:
    records = []
    regimes = ["NOMINAL"] * 10 + ["DEGRADED"] * 5 + ["RECOVERY"] * 5
    for i in range(n_steps):
        record = {
            "step": i,
            "regime": regimes[i % len(regimes)],
            "stability": 0.5 + 0.01 * i,
            "layers": [
                {"R": 0.3 + 0.02 * i + 0.01 * j, "psi": 0.0} for j in range(n_layers)
            ],
            "actions": [],
            "mean_amplitude": 0.5 + 0.01 * i,
            "subcritical_fraction": max(0.0, 0.4 - 0.02 * i),
        }
        if i % 5 == 0:
            record["actions"] = [
                {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                    "justification": "test",
                }
            ]
        records.append(record)
    return records


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
class TestCoherencePlot:
    def test_r_timeline(self, tmp_path) -> None:
        out = tmp_path / "r_timeline.png"
        plot = CoherencePlot(_make_log())
        result = plot.plot_r_timeline(out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_regime_timeline(self, tmp_path) -> None:
        out = tmp_path / "regime.png"
        plot = CoherencePlot(_make_log())
        result = plot.plot_regime_timeline(out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_action_audit(self, tmp_path) -> None:
        out = tmp_path / "actions.png"
        plot = CoherencePlot(_make_log())
        result = plot.plot_action_audit(out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_amplitude_timeline(self, tmp_path) -> None:
        out = tmp_path / "amplitude.png"
        plot = CoherencePlot(_make_log())
        result = plot.plot_amplitude_timeline(out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_pac_heatmap(self, tmp_path) -> None:
        n = 4
        matrix = [0.1 * (i * n + j) / (n * n) for i in range(n) for j in range(n)]
        log_data = _make_log()
        log_data.append({"event": "pac_snapshot", "pac_matrix": matrix, "n": n})
        out = tmp_path / "pac_heatmap.png"
        plot = CoherencePlot(log_data)
        result = plot.plot_pac_heatmap(out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_pac_heatmap_missing_data(self) -> None:
        plot = CoherencePlot(_make_log())
        with pytest.raises(ValueError, match="No pac_matrix"):
            plot.plot_pac_heatmap("/dev/null")


class TestExtractors:
    """Data extraction runs without matplotlib — always covered in CI."""

    def test_extract_r_series(self) -> None:
        plot = CoherencePlot(_make_log(n_steps=10, n_layers=2))
        x, n_layers, series = plot._extract_r_series()
        assert len(x) == 10
        assert n_layers == 2
        assert len(series) == 2
        assert len(series[0]) == 10

    def test_extract_regime_epochs(self) -> None:
        plot = CoherencePlot(_make_log())
        epochs = plot._extract_regime_epochs()
        assert len(epochs) >= 2
        regimes = [e[0] for e in epochs]
        assert "NOMINAL" in regimes
        assert "DEGRADED" in regimes

    def test_extract_actions(self) -> None:
        plot = CoherencePlot(_make_log())
        x, r_global, knob_steps = plot._extract_actions()
        assert len(x) == 20
        assert len(r_global) == 20
        assert "K" in knob_steps
        assert all(0.0 <= r <= 1.5 for r in r_global)

    def test_extract_amplitude(self) -> None:
        plot = CoherencePlot(_make_log())
        x, amps, sub_frac = plot._extract_amplitude()
        assert len(x) == 20
        assert all(a >= 0.0 for a in amps)
        assert all(0.0 <= sf <= 1.0 for sf in sub_frac)

    def test_extract_pac_matrix(self) -> None:
        n = 4
        matrix = [0.1 * (i * n + j) / (n * n) for i in range(n) for j in range(n)]
        log = _make_log()
        log.append({"event": "pac_snapshot", "pac_matrix": matrix, "n": n})
        plot = CoherencePlot(log)
        n_out, mat = plot._extract_pac_matrix()
        assert n_out == 4
        assert mat.shape == (4, 4)

    def test_extract_pac_matrix_missing_raises(self) -> None:
        plot = CoherencePlot(_make_log())
        with pytest.raises(ValueError, match="No pac_matrix"):
            plot._extract_pac_matrix()

    def test_empty_log_raises(self) -> None:
        plot = CoherencePlot([])
        with pytest.raises(ValueError, match="No step records"):
            plot._extract_r_series()

    def test_require_steps_shared(self) -> None:
        plot = CoherencePlot([])
        with pytest.raises(ValueError, match="No step records"):
            plot._require_steps()


def test_no_matplotlib_guard() -> None:
    """Empty log raises ValueError before matplotlib is even needed."""
    plot = CoherencePlot([])
    with pytest.raises((ValueError, ImportError)):
        plot.plot_r_timeline("/dev/null")


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
