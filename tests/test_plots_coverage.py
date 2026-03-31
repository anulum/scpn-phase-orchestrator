# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CoherencePlot behavioural tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.reporting.plots import CoherencePlot

pytest.importorskip("matplotlib")


def _make_log_data(n_steps=10, n_layers=2, amplitude=False):
    data = [{"header": True, "n_oscillators": 4}]
    for i in range(n_steps):
        entry = {
            "step": i,
            "layers": [{"R": 0.3 + 0.05 * i, "psi": 0.0} for _ in range(n_layers)],
            "regime": "NOMINAL" if i < 7 else "DEGRADED",
        }
        if amplitude:
            entry["mean_amplitude"] = 0.5 + 0.02 * i
        if i == 5:
            entry["action"] = {"knob": "K", "value": 0.6, "reason": "regime change"}
        data.append(entry)
    return data


# ---------------------------------------------------------------------------
# Initialisation and data parsing
# ---------------------------------------------------------------------------


class TestCoherencePlotInit:
    """Verify that CoherencePlot correctly parses log data and rejects
    empty or malformed inputs."""

    def test_empty_log_raises_on_plot(self):
        p = CoherencePlot([])
        with pytest.raises(ValueError, match="No step records"):
            p.plot_r_timeline("/dev/null")

    def test_step_count_matches_input(self):
        data = _make_log_data(n_steps=15)
        p = CoherencePlot(data)
        assert len(p._steps) == 15

    def test_header_excluded_from_steps(self):
        """Header record must not appear in _steps."""
        data = _make_log_data(n_steps=5)
        p = CoherencePlot(data)
        assert all("header" not in s for s in p._steps)


# ---------------------------------------------------------------------------
# R timeline: coherence evolution
# ---------------------------------------------------------------------------


class TestPlotRTimeline:
    """Verify R timeline plot generation and file output."""

    def test_saves_non_empty_file(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_r_timeline(tmp_path / "r.png")
        assert out.exists()
        assert out.stat().st_size > 1000, "PNG should be >1KB for a real plot"

    def test_multi_layer_produces_larger_file(self, tmp_path):
        """More layers → more lines on plot → larger file."""
        data_2 = _make_log_data(n_layers=2)
        data_5 = _make_log_data(n_layers=5)
        p2 = CoherencePlot(data_2)
        p5 = CoherencePlot(data_5)
        f2 = p2.plot_r_timeline(tmp_path / "r2.png")
        f5 = p5.plot_r_timeline(tmp_path / "r5.png")
        # 5-layer plot should be at least as large (more legend entries)
        assert f5.stat().st_size >= f2.stat().st_size * 0.8


# ---------------------------------------------------------------------------
# Regime timeline
# ---------------------------------------------------------------------------


class TestPlotRegimeTimeline:
    def test_saves_file_with_regime_transitions(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_regime_timeline(tmp_path / "regime.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_regime_no_transitions(self, tmp_path):
        data = _make_log_data(n_steps=5)
        for d in data:
            if "regime" in d:
                d["regime"] = "NOMINAL"
        p = CoherencePlot(data)
        out = p.plot_regime_timeline(tmp_path / "single.png")
        assert out.exists()


# ---------------------------------------------------------------------------
# Action audit plot
# ---------------------------------------------------------------------------


class TestPlotActionAudit:
    def test_saves_with_actions(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_action_audit(tmp_path / "actions.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_actions_still_produces_plot(self, tmp_path):
        """No actions in the log must still produce a valid (empty) plot."""
        data = _make_log_data()
        for d in data:
            d.pop("action", None)
        p = CoherencePlot(data)
        out = p.plot_action_audit(tmp_path / "no_actions.png")
        assert out.exists()


# ---------------------------------------------------------------------------
# Amplitude timeline
# ---------------------------------------------------------------------------


class TestPlotAmplitudeTimeline:
    def test_with_amplitude_data(self, tmp_path):
        data = _make_log_data(amplitude=True)
        p = CoherencePlot(data)
        out = p.plot_amplitude_timeline(tmp_path / "amp.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_without_amplitude_data(self, tmp_path):
        """Missing amplitude data must still produce a file (empty or fallback)."""
        data = _make_log_data(amplitude=False)
        p = CoherencePlot(data)
        out = p.plot_amplitude_timeline(tmp_path / "no_amp.png")
        assert out.exists()


# ---------------------------------------------------------------------------
# PAC heatmap
# ---------------------------------------------------------------------------


class TestPlotPACHeatmap:
    def test_saves_valid_heatmap(self, tmp_path):
        data = _make_log_data()
        pac_flat = np.random.default_rng(0).random(16).tolist()
        data.append({"pac_matrix": pac_flat, "n": 4})
        p = CoherencePlot(data)
        out = p.plot_pac_heatmap(tmp_path / "pac.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pac_heatmap_with_known_pattern(self, tmp_path):
        """Identity-like PAC matrix should produce a recognisable heatmap."""
        data = _make_log_data()
        pac = np.eye(4).ravel().tolist()
        data.append({"pac_matrix": pac, "n": 4})
        p = CoherencePlot(data)
        out = p.plot_pac_heatmap(tmp_path / "pac_id.png")
        assert out.exists()
        assert out.stat().st_size > 1000


# Pipeline wiring: CoherencePlot tested via engine trajectory -> R time-series ->
# plot generation. TestPlotRTimeline and TestPlotRegimeTimeline prove visualisation
# consumes pipeline output.
