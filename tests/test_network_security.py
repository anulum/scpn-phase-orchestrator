# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network security helper tests

from __future__ import annotations

import importlib.util
from typing import cast

import pytest

from scpn_phase_orchestrator.runtime.network_security import (
    FixedWindowRateLimiter,
    TokenBucketRateLimiter,
    env_int,
    is_production_mode,
)


def test_presplit_network_security_module_is_removed() -> None:
    assert importlib.util.find_spec("scpn_phase_orchestrator.network_security") is None


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


@pytest.mark.parametrize("prefix", ["", " ", True])
def test_is_production_mode_rejects_malformed_prefix(prefix: object) -> None:
    with pytest.raises(ValueError, match="prefix"):
        is_production_mode(cast(str, prefix))


def test_env_int_default_and_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPO_RATE_LIMIT_PER_MINUTE", raising=False)
    assert env_int("SPO_RATE_LIMIT_PER_MINUTE", 120) == 120

    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "7")
    assert env_int("SPO_RATE_LIMIT_PER_MINUTE", 120) == 7


@pytest.mark.parametrize("name", ["", " ", True])
def test_env_int_rejects_malformed_name(name: object) -> None:
    with pytest.raises(ValueError, match="name"):
        env_int(cast(str, name), 120)


def test_env_int_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        env_int("SPO_RATE_LIMIT_PER_MINUTE", 120)

    monkeypatch.setenv("SPO_RATE_LIMIT_PER_MINUTE", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        env_int("SPO_RATE_LIMIT_PER_MINUTE", 120)


@pytest.mark.parametrize("default", [True, -1, "5"])
def test_env_int_rejects_malformed_default(
    monkeypatch: pytest.MonkeyPatch,
    default: object,
) -> None:
    monkeypatch.delenv("SPO_RATE_LIMIT_PER_MINUTE", raising=False)

    with pytest.raises(ValueError, match="default"):
        env_int("SPO_RATE_LIMIT_PER_MINUTE", cast(int, default))


def test_rate_limiter_uses_token_bucket_burst_capacity() -> None:
    limiter = TokenBucketRateLimiter(limit_per_minute=60, burst_capacity=2)

    assert limiter.allow("client-a", now=10.0) is True
    assert limiter.allow("client-a", now=11.0) is True
    assert limiter.allow("client-a", now=11.5) is False
    assert limiter.allow("client-a", now=12.0) is True


def test_compat_rate_limiter_does_not_allow_full_minute_burst() -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=60)

    allowed = sum(1 for _ in range(60) if limiter.allow("client-a", now=10.0))

    assert allowed < 60


def test_rate_limiter_is_per_identity() -> None:
    limiter = TokenBucketRateLimiter(limit_per_minute=1, burst_capacity=1)

    assert limiter.allow("client-a", now=10.0) is True
    assert limiter.allow("client-a", now=11.0) is False
    assert limiter.allow("client-b", now=11.0) is True


def test_rate_limiter_resets_on_new_window() -> None:
    limiter = TokenBucketRateLimiter(limit_per_minute=60, burst_capacity=1)

    assert limiter.allow("client-a", now=59.0) is True
    assert limiter.allow("client-a", now=59.5) is False
    assert limiter.allow("client-a", now=60.0) is True


@pytest.mark.parametrize("now", [True, float("nan"), float("inf"), "10"])
def test_rate_limiter_rejects_malformed_timestamp(now: object) -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=1)

    with pytest.raises(ValueError, match="now"):
        limiter.allow("client-a", now=cast(float, now))


@pytest.mark.parametrize("identity", ["", " ", True, 7])
def test_rate_limiter_rejects_malformed_identity(identity: object) -> None:
    limiter = FixedWindowRateLimiter(limit_per_minute=1)

    with pytest.raises(ValueError, match="identity"):
        limiter.allow(cast(str, identity), now=10.0)


def test_rate_limiter_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        FixedWindowRateLimiter(limit_per_minute=0)


@pytest.mark.parametrize("limit", [True, 1.5, "10"])
def test_rate_limiter_rejects_malformed_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="limit_per_minute"):
        FixedWindowRateLimiter(limit_per_minute=cast(int, limit))


@pytest.mark.parametrize("burst_capacity", [0, True, 2.5, "2"])
def test_rate_limiter_rejects_malformed_burst_capacity(
    burst_capacity: object,
) -> None:
    with pytest.raises(ValueError, match="burst_capacity"):
        TokenBucketRateLimiter(
            limit_per_minute=10,
            burst_capacity=cast(int, burst_capacity),
        )


def test_rate_limiter_rejects_burst_above_minute_limit() -> None:
    with pytest.raises(ValueError, match="burst_capacity"):
        TokenBucketRateLimiter(limit_per_minute=2, burst_capacity=3)
