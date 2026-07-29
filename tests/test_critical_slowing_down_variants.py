# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — tests for critical-slowing-down variants

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
    critical_slowing_down_multiscale_warning,
    surrogate_score_threshold,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_variant_module():
    """Load the bench driver as a module without adding ``bench`` to ``sys.path``."""
    target = REPO_ROOT / "bench" / "critical_slowing_down_variants.py"
    spec = importlib.util.spec_from_file_location("csd_variants", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["csd_variants"] = module
    spec.loader.exec_module(module)
    return module


csd_variants = _load_variant_module()


def test_multiscale_warning_returns_compatible_record():
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((4, 1024))
    warning = critical_slowing_down_multiscale_warning(
        signals,
        windows=(64, 128),
        step=16,
        baseline_fraction=0.25,
        z_threshold=1.0,
        rise_threshold=0.0,
        persistence=1,
    )
    assert isinstance(warning, CriticalSlowingDownWarning)
    assert warning.combined_z.shape[0] == warning.window_starts.shape[0]
    assert warning.variance_index.shape == warning.combined_z.shape
    assert warning.autocorrelation_index.shape == warning.combined_z.shape


def test_multiscale_aggregation_modes_differ():
    rng = np.random.default_rng(7)
    signals = rng.standard_normal((2, 512))
    max_warning = critical_slowing_down_multiscale_warning(
        signals,
        windows=(64, 128),
        step=16,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=1,
        aggregation="max",
    )
    mean_warning = critical_slowing_down_multiscale_warning(
        signals,
        windows=(64, 128),
        step=16,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=1,
        aggregation="mean",
    )
    assert max_warning.combined_z.max() >= mean_warning.combined_z.max()


def test_surrogate_threshold_is_positive_and_finite():
    rng = np.random.default_rng(99)
    signals = rng.standard_normal((1, 512))
    threshold = surrogate_score_threshold(
        signals,
        n_surrogates=10,
        percentile=90.0,
        window=64,
        step=16,
        rng=rng,
    )
    assert np.isfinite(threshold)
    assert threshold >= 0.0


@pytest.mark.parametrize("windows", [[], (), "64", 64])
def test_multiscale_requires_non_empty_window_sequence(windows):
    with pytest.raises(ValueError, match="windows must be a non-empty sequence"):
        critical_slowing_down_multiscale_warning(
            np.zeros((1, 512)),
            windows=windows,
        )


def test_multiscale_default_windows_are_usable() -> None:
    warning = critical_slowing_down_multiscale_warning(
        np.zeros((1, 300)),
        step=16,
    )

    assert warning.window == 256


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"windows": (8, 8)}, "windows must be unique"),
        ({"windows": (8,), "aggregation": "median"}, "aggregation"),
        ({"windows": (2,)}, "at least 3"),
        ({"windows": (128,)}, "exceeds the series length"),
    ],
)
def test_multiscale_rejects_invalid_scale_geometry(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        critical_slowing_down_multiscale_warning(
            np.zeros((1, 64)),
            step=4,
            **kwargs,
        )


@pytest.mark.parametrize("rng", [True, np.bool_(True), -1, "7"])
def test_surrogate_rng_requires_canonical_seed_or_generator(rng):
    with pytest.raises(ValueError, match="rng must be a non-negative integer"):
        surrogate_score_threshold(
            np.zeros((1, 64)),
            n_surrogates=1,
            window=8,
            step=4,
            rng=rng,
        )


def test_surrogate_accepts_numpy_integer_seed() -> None:
    signals = np.arange(64, dtype=np.float64).reshape(1, -1)

    first = surrogate_score_threshold(
        signals,
        n_surrogates=2,
        window=8,
        step=4,
        rng=np.int64(7),
    )
    second = surrogate_score_threshold(
        signals,
        n_surrogates=2,
        window=8,
        step=4,
        rng=np.int64(7),
    )

    assert first == second


def test_surrogate_rejects_percentile_above_one_hundred() -> None:
    with pytest.raises(ValueError, match="percentile must be <= 100"):
        surrogate_score_threshold(
            np.zeros((1, 64)),
            n_surrogates=1,
            percentile=100.1,
            window=8,
            step=4,
        )


def test_surrogate_accepts_explicit_block_length() -> None:
    threshold = surrogate_score_threshold(
        np.arange(64, dtype=np.float64).reshape(1, -1),
        n_surrogates=1,
        block_length=4,
        window=8,
        step=4,
        rng=7,
    )

    assert np.isfinite(threshold)


def test_surrogate_rejects_empty_signals_before_bootstrap() -> None:
    with pytest.raises(ValueError, match="at least one node and one sample"):
        surrogate_score_threshold(
            np.empty((1, 0)),
            n_surrogates=1,
            window=3,
            step=1,
        )


def test_multiscale_improves_over_baseline_on_explosive_transition(tmp_path):
    """End-to-end check that the multi-scale variant out-detects the baseline.

    This uses the same vanilla Kuramoto simulator as
    ``bench/early_warning_leadtime.py`` but with a small ensemble so the test
    stays fast. The seed is fixed for determinism.
    """
    output_dir = tmp_path / "csd_variant_synthetic"
    rc = csd_variants.main(
        [
            "--n",
            "24",
            "--steps",
            "2000",
            "--n-realisations",
            "8",
            "--output-dir",
            str(output_dir),
            "--seed",
            "20260710",
        ]
    )
    assert rc == 0
    aggregate_path = output_dir / "csd_variant_synthetic_results.json"
    assert aggregate_path.exists()
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    baseline_rate = aggregate["critical_slowing_down_baseline"]["mean_detection_rate"]
    multiscale_rate = aggregate["critical_slowing_down_multiscale"][
        "mean_detection_rate"
    ]
    assert multiscale_rate > baseline_rate


def test_driver_aggregate_schema_is_meta_analysis_friendly(tmp_path):
    output_dir = tmp_path / "csd_variant_synthetic"
    rc = csd_variants.main(
        [
            "--n",
            "16",
            "--steps",
            "1000",
            "--n-realisations",
            "4",
            "--output-dir",
            str(output_dir),
            "--seed",
            "20260711",
        ]
    )
    assert rc == 0
    aggregate = json.loads(
        (output_dir / "csd_variant_synthetic_results.json").read_text(encoding="utf-8")
    )
    assert aggregate["benchmark"] == "csd_variant_synthetic"
    for variant in (
        "critical_slowing_down_baseline",
        "critical_slowing_down_multiscale",
        "critical_slowing_down_surrogate",
    ):
        assert variant in aggregate
        assert "mean_detection_rate" in aggregate[variant]
        assert 0.0 <= aggregate[variant]["mean_detection_rate"] <= 1.0
