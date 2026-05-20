# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public roadmap status tests

from pathlib import Path

PUBLIC_ROADMAP = Path("docs/roadmap.md")


def test_public_roadmap_reflects_benchmark_gated_auto_binding_status() -> None:
    text = PUBLIC_ROADMAP.read_text(encoding="utf-8")

    assert (
        "larger live-dataset benchmarks and domain-specific "
        "automatic-acceptance thresholds remain open"
    ) not in text
    assert "benchmark-gated domain thresholds" in text
    assert (
        "deterministic domain-like fixtures now enforce domain-specific gates"
        in text
    )
    assert "private or partner live datasets remain future work" in text


def test_public_roadmap_reflects_bayesian_posterior_fit_status() -> None:
    text = PUBLIC_ROADMAP.read_text(encoding="utf-8")

    assert (
        "posterior fitting and benchmarked NumPyro/BlackJAX samplers remain open"
        not in text
    )
    assert "Gaussian posterior fitting from observed Kuramoto trajectories" in text
    assert (
        "reserved NumPyro/BlackJAX sampler names are benchmarked as fail-closed"
        in text
    )
