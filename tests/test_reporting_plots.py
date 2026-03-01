from __future__ import annotations

import pytest

from scpn_phase_orchestrator.reporting.plots import CoherencePlot


def test_coherence_plot_init():
    plot = CoherencePlot(log_data=[{"step": 0}])
    assert len(plot._data) == 1


def test_plot_methods_not_implemented():
    plot = CoherencePlot([])
    with pytest.raises(NotImplementedError):
        plot.plot_r_timeline("out.png")
    with pytest.raises(NotImplementedError):
        plot.plot_regime_timeline("out.png")
    with pytest.raises(NotImplementedError):
        plot.plot_action_audit("out.png")
