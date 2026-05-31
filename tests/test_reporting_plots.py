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
import sys
from pathlib import Path
from types import ModuleType

import pytest

import scpn_phase_orchestrator.reporting.plots as plots_module
from scpn_phase_orchestrator.reporting.plots import CoherencePlot

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


class _FakeAxis:
    def __init__(self) -> None:
        self.vertical_markers: list[int] = []

    def plot(self, *args, **kwargs):
        return [object()]

    def set_xlabel(self, *args, **kwargs) -> None:
        return None

    def set_ylabel(self, *args, **kwargs) -> None:
        return None

    def set_ylim(self, *args, **kwargs) -> None:
        return None

    def set_xlim(self, *args, **kwargs) -> None:
        return None

    def set_yticks(self, *args, **kwargs) -> None:
        return None

    def legend(self, *args, **kwargs) -> None:
        return None

    def add_patch(self, *args, **kwargs) -> None:
        return None

    def axvline(self, step_x, *args, **kwargs) -> None:
        self.vertical_markers.append(step_x)

    def tick_params(self, *args, **kwargs) -> None:
        return None

    def twinx(self):
        return _FakeAxis()

    def get_legend_handles_labels(self):
        return [], []

    def imshow(self, *args, **kwargs):
        return object()


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, output_path, *args, **kwargs) -> None:
        with Path(output_path).open("wb") as handle:
            handle.write(b"fake-rendered-plot")

    def colorbar(self, *args, **kwargs) -> None:
        return None


class _FakePyplot:
    def __init__(self) -> None:
        self.closed: list[_FakeFigure] = []

    def subplots(self, *args, **kwargs):
        return _FakeFigure(), _FakeAxis()

    def close(self, fig) -> None:
        self.closed.append(fig)


def _fake_rectangle(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


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


class TestMatplotlibLoaderContracts:
    def test_cached_matplotlib_handles_are_reused(self, monkeypatch) -> None:
        fake_pyplot = _FakePyplot()
        monkeypatch.setattr(plots_module, "_plt", fake_pyplot)
        monkeypatch.setattr(plots_module, "_Rectangle", _fake_rectangle)

        assert plots_module._require_matplotlib() == (fake_pyplot, _fake_rectangle)

    def test_missing_matplotlib_raises_actionable_extra_message(self, monkeypatch):
        monkeypatch.setattr(plots_module, "_plt", None)
        monkeypatch.setattr(plots_module, "_HAS_MPL", False)

        with pytest.raises(ImportError, match=r"scpn-phase-orchestrator\[plot\]"):
            plots_module._require_matplotlib()

    def test_uncached_loader_selects_headless_backend(self, monkeypatch) -> None:
        backend_updates: list[str] = []

        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.get_backend = lambda: "TkAgg"
        fake_matplotlib.use = backend_updates.append

        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = _FakePyplot().subplots
        fake_pyplot.close = _FakePyplot().close

        fake_patches = ModuleType("matplotlib.patches")
        fake_patches.Rectangle = _fake_rectangle

        monkeypatch.setattr(plots_module, "_plt", None)
        monkeypatch.setattr(plots_module, "_Rectangle", None)
        monkeypatch.setattr(plots_module, "_HAS_MPL", True)
        monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
        monkeypatch.setitem(sys.modules, "matplotlib.patches", fake_patches)

        assert plots_module._require_matplotlib() == (fake_pyplot, _fake_rectangle)
        assert backend_updates == ["Agg"]


class TestCoherencePlotHeadlessRendering:
    def test_all_plot_methods_write_outputs_with_headless_renderer(
        self, monkeypatch, tmp_path
    ) -> None:
        fake_pyplot = _FakePyplot()
        monkeypatch.setattr(
            plots_module,
            "_require_matplotlib",
            lambda: (fake_pyplot, _fake_rectangle),
        )

        log_data = _make_log()
        log_data.append({"event": "pac_snapshot", "pac_matrix": [0.0, 0.1, 0.2, 0.3]})
        plot = CoherencePlot(log_data)

        outputs = [
            plot.plot_r_timeline(tmp_path / "r.png"),
            plot.plot_regime_timeline(tmp_path / "regime.png"),
            plot.plot_action_audit(tmp_path / "actions.png"),
            plot.plot_amplitude_timeline(tmp_path / "amplitude.png"),
            plot.plot_pac_heatmap(tmp_path / "pac.png"),
        ]

        assert all(path.exists() for path in outputs)
        assert all(path.read_bytes() == b"fake-rendered-plot" for path in outputs)
        assert len(fake_pyplot.closed) == len(outputs)


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
                "corrupted-jsonl-line",
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

    def test_constructor_ignores_non_dict_audit_payloads(self) -> None:
        plot = CoherencePlot(
            [
                None,
                17,
                {"event": "bootstrap"},
                {"step": 0, "regime": "NOMINAL", "layers": [{"R": 0.2}]},
            ]
        )

        x, n_layers, series = plot._extract_r_series()

        assert x == [0]
        assert n_layers == 1
        assert series == [[0.2]]

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

    def test_extract_regime_epochs_stringifies_malformed_regime_values(self) -> None:
        plot = CoherencePlot(
            [
                {"step": 0, "regime": ["NOMINAL"], "layers": [{"R": 0.2}]},
                {"step": 1, "regime": {"bad": "shape"}, "layers": [{"R": 0.4}]},
            ]
        )

        assert plot._extract_regime_epochs() == [
            ("['NOMINAL']", 0, 1),
            ("{'bad': 'shape'}", 1, 2),
        ]

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

    def test_extract_pac_matrix_infers_dimension_when_n_is_missing(self) -> None:
        log = _make_log()
        log.append({"event": "pac_snapshot", "pac_matrix": [0.11, 0.12, 0.13, 0.14]})
        plot = CoherencePlot(log)
        n_out, mat = plot._extract_pac_matrix()
        assert n_out == 2
        assert mat.shape == (2, 2)

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

    def test_extract_pac_matrix_skips_non_dict_records_and_rejects_non_list_matrix(
        self,
    ) -> None:
        plot = CoherencePlot(
            [
                "corrupted-jsonl-line",
                {"event": "pac_snapshot", "pac_matrix": "not-a-flat-list", "n": 2},
            ]
        )

        with pytest.raises(ValueError, match="pac_matrix must be a flat list"):
            plot._extract_pac_matrix()

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


def test_require_matplotlib_fails_closed_when_optional_dependency_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plots_module, "_HAS_MPL", False)
    monkeypatch.setattr(plots_module, "_plt", None)
    monkeypatch.setattr(plots_module, "_Rectangle", None)

    with pytest.raises(ImportError, match="matplotlib required"):
        plots_module._require_matplotlib()


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
