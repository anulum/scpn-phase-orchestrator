# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network security helper tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.network_security import (
    FixedWindowRateLimiter,
    env_int,
    is_production_mode,
)


def test_is_production_mode_uses_service_specific_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUEUEWAVES_ENV", "production")
    monkeypatch.setenv("SPO_ENV", "development")

    assert is_production_mode("QUEUEWAVES") is True


def test_is_production_mode_uses_generic_spo_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPO_GRPC_ENV", raising=False)
    monkeypatch.setenv("SPO_PROFILE", "production")

    assert is_production_mode("SPO_GRPC") is True


def test_is_production_mode_ignores_non_production_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_ENV", "development")
    monkeypatch.setenv("SPO_PROFILE", "local")

    assert is_production_mode("SPO") is False


def test_env_int_default_and_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPO_RATE_LIMIT_PER_MINUTE", raising=False)
    assert env_int("SPO_RATE_LIMIT_PER_MINUTE", 120) == 120

    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "7")
    assert env_int("SPO_RATE_LIMIT_PER_MINUTE", 120) == 7


def test_env_int_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        env_int("SPO_RATE_LIMIT_PER_MINUTE", 120)

    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        env_int("SPO_RATE_LIMIT_PER_MINUTE", 120)


def test_rate_limiter_allows_until_limit_then_blocks() -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=2)

    assert limiter.allow("client-a", now=10.0) is True
    assert limiter.allow("client-a", now=11.0) is True
    assert limiter.allow("client-a", now=12.0) is False


def test_rate_limiter_is_per_identity() -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=1)

    assert limiter.allow("client-a", now=10.0) is True
    assert limiter.allow("client-a", now=11.0) is False
    assert limiter.allow("client-b", now=11.0) is True


def test_rate_limiter_resets_on_new_window() -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=1)

    assert limiter.allow("client-a", now=59.0) is True
    assert limiter.allow("client-a", now=59.5) is False
    assert limiter.allow("client-a", now=60.0) is True


def test_rate_limiter_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        FixedWindowRateLimiter(limit_per_minute=0)
