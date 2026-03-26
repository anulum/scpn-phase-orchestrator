# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CoherencePlot coverage tests

from __future__ import annotations

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


class TestCoherencePlotInit:
    def test_empty_log(self):
        p = CoherencePlot([])
        with pytest.raises(ValueError, match="No step records"):
            p.plot_r_timeline("/dev/null")

    def test_valid_log(self):
        data = _make_log_data()
        p = CoherencePlot(data)
        assert len(p._steps) == 10


class TestPlotRTimeline:
    def test_saves_file(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_r_timeline(tmp_path / "r.png")
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotRegimeTimeline:
    def test_saves_file(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_regime_timeline(tmp_path / "regime.png")
        assert out.exists()

    def test_single_regime(self, tmp_path):
        data = _make_log_data(n_steps=5)
        for d in data:
            if "regime" in d:
                d["regime"] = "NOMINAL"
        p = CoherencePlot(data)
        out = p.plot_regime_timeline(tmp_path / "single.png")
        assert out.exists()


class TestPlotActionAudit:
    def test_saves_file(self, tmp_path):
        data = _make_log_data()
        p = CoherencePlot(data)
        out = p.plot_action_audit(tmp_path / "actions.png")
        assert out.exists()

    def test_no_actions(self, tmp_path):
        data = _make_log_data()
        for d in data:
            d.pop("action", None)
        p = CoherencePlot(data)
        out = p.plot_action_audit(tmp_path / "no_actions.png")
        assert out.exists()


class TestPlotAmplitude:
    def test_saves_file(self, tmp_path):
        data = _make_log_data(amplitude=True)
        p = CoherencePlot(data)
        out = p.plot_amplitude_timeline(tmp_path / "amp.png")
        assert out.exists()

    def test_no_amplitude_data(self, tmp_path):
        data = _make_log_data(amplitude=False)
        p = CoherencePlot(data)
        out = p.plot_amplitude_timeline(tmp_path / "no_amp.png")
        assert out.exists()


class TestPlotPACHeatmap:
    def test_saves_file(self, tmp_path):
        import numpy as np

        data = _make_log_data()
        pac_flat = np.random.default_rng(0).random(16).tolist()
        data.append({"pac_matrix": pac_flat, "n": 4})
        p = CoherencePlot(data)
        out = p.plot_pac_heatmap(tmp_path / "pac.png")
        assert out.exists()
