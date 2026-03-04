# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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


def test_no_matplotlib_guard() -> None:
    """Empty log raises before matplotlib check."""
    plot = CoherencePlot([])
    # Empty log → ValueError before matplotlib is even needed
    with pytest.raises((ValueError, ImportError)):
        plot.plot_r_timeline("/dev/null")
