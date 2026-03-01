# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState


def _make_state(r_values, lock_pairs=None):
    layers = []
    for i, r in enumerate(r_values):
        sigs = {}
        if lock_pairs:
            for (a, b), plv in lock_pairs.items():
                if a == i:
                    sigs[f"{a}_{b}"] = LockSignature(
                        source_layer=a,
                        target_layer=b,
                        plv=plv,
                        mean_lag=0.0,
                    )
        layers.append(LayerState(R=r, psi=0.0, lock_signatures=sigs))
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=float(np.mean(r_values)),
        regime_id="nominal",
    )


def test_r_good_from_correct_subset():
    monitor = CoherenceMonitor(good_layers=[0, 1], bad_layers=[2])
    state = _make_state([0.9, 0.8, 0.3])
    r_good = monitor.compute_r_good(state)
    np.testing.assert_allclose(r_good, 0.85, atol=1e-12)


def test_r_bad_from_correct_subset():
    monitor = CoherenceMonitor(good_layers=[0], bad_layers=[1, 2])
    state = _make_state([0.9, 0.3, 0.4])
    r_bad = monitor.compute_r_bad(state)
    np.testing.assert_allclose(r_bad, 0.35, atol=1e-12)


def test_detect_phase_lock_found():
    pairs = {(0, 1): 0.95}
    state = _make_state([0.9, 0.8], lock_pairs=pairs)
    monitor = CoherenceMonitor(good_layers=[0], bad_layers=[1])
    locked = monitor.detect_phase_lock(state, threshold=0.9)
    assert (0, 1) in locked


def test_detect_phase_lock_below_threshold():
    pairs = {(0, 1): 0.5}
    state = _make_state([0.9, 0.8], lock_pairs=pairs)
    monitor = CoherenceMonitor(good_layers=[0], bad_layers=[1])
    locked = monitor.detect_phase_lock(state, threshold=0.9)
    assert locked == []


def test_empty_good_layers_returns_zero():
    monitor = CoherenceMonitor(good_layers=[], bad_layers=[0])
    state = _make_state([0.5])
    assert monitor.compute_r_good(state) == 0.0
