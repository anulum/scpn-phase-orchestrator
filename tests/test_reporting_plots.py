# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

    def test_extract_r_series_ignores_malformed_layer_containers(self) -> None:
        plot = CoherencePlot(
            [
                {"step": 0, "regime": "NOMINAL", "layers": None},
                {
                    "step": 1,
                    "regime": "DEGRADED",
                    "layers": ["not-a-layer", {"R": 0.4}],
                },
            ]
        )
        x, n_layers, series = plot._extract_r_series()
        assert x == [0, 1]
        assert n_layers == 1
        assert series == [[0.0, 0.4]]

    def test_extract_r_series_ignores_malformed_layer_r_values(self) -> None:
        plot = CoherencePlot(
            [
                {
                    "step": 0,
                    "regime": "NOMINAL",
                    "layers": [{"R": True}, {"R": float("nan")}],
                },
                {
                    "step": 1,
                    "regime": "DEGRADED",
                    "layers": [{"R": float("inf")}, {"R": 0.4}],
                },
            ]
        )
        x, n_layers, series = plot._extract_r_series()
        assert x == [0, 1]
        assert n_layers == 2
        assert series == [[0.0, 0.0], [0.0, 0.4]]

    def test_extract_regime_epochs(self) -> None:
        plot = CoherencePlot(_make_log())
        epochs = plot._extract_regime_epochs()
        assert len(epochs) >= 2
        regimes = [e[0] for e in epochs]
        assert "NOMINAL" in regimes
        assert "DEGRADED" in regimes

    def test_extractors_ignore_malformed_step_identifiers(self) -> None:
        plot = CoherencePlot(
            [
                {"step": True, "regime": "NOMINAL", "layers": [{"R": 0.2}]},
                {
                    "step": "not-an-int",
                    "regime": "DEGRADED",
                    "layers": [{"R": 0.4}],
                    "actions": [{"knob": "K"}],
                    "mean_amplitude": 0.5,
                    "subcritical_fraction": 0.25,
                },
            ]
        )
        x, _, _ = plot._extract_r_series()
        epochs = plot._extract_regime_epochs()
        action_x, _, knob_steps = plot._extract_actions()
        amplitude_x, _, _ = plot._extract_amplitude()

        assert x == [0, 0]
        assert epochs == [("NOMINAL", 0, 0), ("DEGRADED", 0, 1)]
        assert action_x == [0, 0]
        assert knob_steps == {"K": [0]}
        assert amplitude_x == [0, 0]

    def test_extract_actions(self) -> None:
        plot = CoherencePlot(_make_log())
        x, r_global, knob_steps = plot._extract_actions()
        assert len(x) == 20
        assert len(r_global) == 20
        assert "K" in knob_steps
        assert all(0.0 <= r <= 1.5 for r in r_global)

    def test_extract_actions_ignores_malformed_layer_containers(self) -> None:
        plot = CoherencePlot(
            [
                {"step": 0, "regime": "NOMINAL", "layers": None, "actions": []},
                {
                    "step": 1,
                    "regime": "DEGRADED",
                    "layers": ["not-a-layer", {"R": 0.4}],
                    "actions": [{"knob": "K"}],
                },
            ]
        )
        x, r_global, knob_steps = plot._extract_actions()
        assert x == [0, 1]
        assert r_global == [0.0, 0.4]
        assert knob_steps == {"K": [1]}

    def test_extract_actions_ignores_malformed_action_payloads(self) -> None:
        plot = CoherencePlot(
            [
                {
                    "step": 0,
                    "regime": "NOMINAL",
                    "layers": [{"R": 0.2}],
                    "actions": None,
                },
                {
                    "step": 1,
                    "regime": "DEGRADED",
                    "layers": [{"R": 0.4}],
                    "actions": ["not-an-action", {"scope": "global"}, {"knob": "K"}],
                },
            ]
        )
        x, r_global, knob_steps = plot._extract_actions()
        assert x == [0, 1]
        assert r_global == [0.2, 0.4]
        assert knob_steps == {"K": [1]}

    def test_extract_amplitude(self) -> None:
        plot = CoherencePlot(_make_log())
        x, amps, sub_frac = plot._extract_amplitude()
        assert len(x) == 20
        assert all(a >= 0.0 for a in amps)
        assert all(0.0 <= sf <= 1.0 for sf in sub_frac)

    def test_extract_amplitude_ignores_malformed_numeric_values(self) -> None:
        plot = CoherencePlot(
            [
                {
                    "step": 0,
                    "regime": "NOMINAL",
                    "layers": [{"R": 0.2}],
                    "mean_amplitude": True,
                    "subcritical_fraction": float("nan"),
                },
                {
                    "step": 1,
                    "regime": "DEGRADED",
                    "layers": [{"R": 0.4}],
                    "mean_amplitude": float("inf"),
                    "subcritical_fraction": 0.25,
                },
            ]
        )
        x, amps, sub_frac = plot._extract_amplitude()
        assert x == [0, 1]
        assert amps == [0.0, 0.0]
        assert sub_frac == [0.0, 0.25]

    def test_extract_pac_matrix(self) -> None:
        n = 4
        matrix = [0.1 * (i * n + j) / (n * n) for i in range(n) for j in range(n)]
        log = _make_log()
        log.append({"event": "pac_snapshot", "pac_matrix": matrix, "n": n})
        plot = CoherencePlot(log)
        n_out, mat = plot._extract_pac_matrix()
        assert n_out == 4
        assert mat.shape == (4, 4)

    def test_extract_pac_matrix_rejects_malformed_dimensions(self) -> None:
        log = _make_log()
        log.append({"event": "pac_snapshot", "pac_matrix": [0.1, 0.2, 0.3], "n": 2})
        plot = CoherencePlot(log)
        with pytest.raises(ValueError, match="pac_matrix length"):
            plot._extract_pac_matrix()

    def test_extract_pac_matrix_ignores_malformed_numeric_values(self) -> None:
        log = _make_log()
        log.append(
            {
                "event": "pac_snapshot",
                "pac_matrix": [True, float("nan"), float("inf"), 0.25],
                "n": 2,
            }
        )
        plot = CoherencePlot(log)
        n_out, mat = plot._extract_pac_matrix()
        assert n_out == 2
        assert mat.tolist() == [[0.0, 0.0], [0.0, 0.25]]

    @pytest.mark.parametrize("n_value", [True, float("nan"), float("inf"), 2.5, "2"])
    def test_extract_pac_matrix_rejects_malformed_n_metadata(
        self, n_value: object
    ) -> None:
        log = _make_log()
        log.append(
            {"event": "pac_snapshot", "pac_matrix": [0.1, 0.2, 0.3, 0.4], "n": n_value}
        )
        plot = CoherencePlot(log)
        with pytest.raises(ValueError, match="pac_matrix n"):
            plot._extract_pac_matrix()

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


def test_matplotlib_not_imported_on_module_import() -> None:
    """Importing plots module must not load matplotlib (V8 lazy-import).

    Verifies the module can be imported on CLI/server paths that never plot
    anything, without paying the matplotlib backend init cost.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scpn_phase_orchestrator.reporting.plots as _; "
            'print("loaded" if "matplotlib" in sys.modules else "absent")',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "absent", (
        f"matplotlib was loaded on module import: {result.stdout!r}"
    )


class TestReportingPlotsPipelineWiring:
    """Pipeline: audit log → CoherencePlot → R time series."""

    def test_audit_log_to_coherence_plot(self):
        """AuditLogger log → CoherencePlot → _extract_r_series.
        Proves reporting consumes audit pipeline output."""
        log = _make_log()
        plot = CoherencePlot(log)
        steps, n_layers, r_series = plot._extract_r_series()
        assert len(steps) > 0
        assert n_layers >= 1
        assert len(r_series) == n_layers
        for layer_r in r_series:
            assert all(0.0 <= r <= 1.0 for r in layer_r)
