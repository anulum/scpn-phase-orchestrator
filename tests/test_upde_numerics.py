from __future__ import annotations

from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability


def test_config_defaults():
    cfg = IntegrationConfig(dt=0.001)
    assert cfg.substeps == 1
    assert cfg.method == "euler"
    assert cfg.max_dt == 0.01


def test_stable_small_dt():
    assert check_stability(dt=0.001, max_omega=2.0, max_coupling=1.0) is True


def test_unstable_large_dt():
    assert check_stability(dt=2.0, max_omega=5.0, max_coupling=5.0) is False


def test_zero_deriv_always_stable():
    assert check_stability(dt=100.0, max_omega=0.0, max_coupling=0.0) is True
