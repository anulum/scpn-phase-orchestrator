# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — public capability-claim boundary guards

"""Prevent public capability and toolchain docs from outrunning evidence."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = ROOT / "docs" / "galleries" / "capabilities.md"
INSTALLATION = ROOT / "docs" / "getting-started" / "installation.md"
RUST_GUIDE = ROOT / "docs" / "guide" / "rust_ffi.md"
PERFORMANCE = ROOT / "docs" / "guide" / "performance.md"
VALIDATION = ROOT / "VALIDATION.md"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"
PUBLIC_DOCS = ROOT / "docs"


def _rust_msrv() -> str:
    cargo = tomllib.loads(
        (ROOT / "spo-kernel" / "Cargo.toml").read_text(encoding="utf-8")
    )
    return cargo["workspace"]["package"]["rust-version"]


def test_public_rust_prerequisites_match_workspace_msrv() -> None:
    msrv = _rust_msrv()
    assert msrv == "1.83.0"
    assert f"Rust {msrv} build" in VALIDATION.read_text(encoding="utf-8")
    for path in (INSTALLATION, RUST_GUIDE):
        assert "Rust 1.83+" in path.read_text(encoding="utf-8")


def test_capability_page_separates_implementation_from_claims() -> None:
    text = CAPABILITIES.read_text(encoding="utf-8")
    for required in (
        "Capability maturity at a glance",
        "Externally checked scientific niche",
        "Domain scaffold",
        "Review-only integration",
        "Where the market value can come from",
        "What SPO does not establish",
    ):
        assert required in text

    for unsupported in (
        "predicts systemic phase transitions",
        "milliseconds before macroscopic failure",
        "N = 10^6",
        "Digital Earth synchronization",
        "Sub-millisecond sparse UPDE integrations",
    ):
        assert unsupported not in text


def test_hardware_and_performance_docs_deny_portable_deadline_claims() -> None:
    assert "sub-15μs" not in README.read_text(encoding="utf-8")
    assert "sub-15μs" not in ARCHITECTURE.read_text(encoding="utf-8")

    performance = PERFORMANCE.read_text(encoding="utf-8")
    rust = RUST_GUIDE.read_text(encoding="utf-8")
    assert "not portable throughput or real-time" in performance
    assert "not establish a 250 Hz control-loop deadline" in rust


def test_primary_public_docs_reject_stale_or_unsupported_performance_claims() -> None:
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PUBLIC_DOCS.rglob("*.md")
        if "internal" not in path.parts and "superpowers" not in path.parts
    )
    for unsupported in (
        "53 engine modules, 2-96x speedup",
        "Sub-microsecond Control",
        "real-time operation to ~10³ oscillators",
        "enabling real-time spiking control loops at ~250 Hz",
        "Current: **v0.4.1**",
        "handles N=1000+ oscillators at <1ms/step",
        "QueueWaves detects the R drop 10-30 seconds",
        "SPO predicts cascade failures",
        "SPO detects mode locking precursors",
        "This paper presents the first hazard analysis",
        "No other project has addressed them either",
    ):
        assert unsupported not in public_text
