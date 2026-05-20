# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_non_negative_int,
    require_tcp_port,
)
from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore


def test_schema_validators_normalise_and_reject_invalid_values() -> None:
    assert require_non_empty_str("  review  ", field="mode") == "review"
    with pytest.raises(ValueError, match="non-empty string"):
        require_non_empty_str("   ", field="mode")
    with pytest.raises(ValueError, match="must be an integer"):
        require_non_negative_int(None, field="count")

    assert require_non_negative_int(0, field="count") == 0
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        require_non_negative_int(-3, field="count")
    with pytest.raises(ValueError, match="must be an integer"):
        require_non_negative_int(True, field="count")

    assert require_tcp_port(5000, field="Modbus port") == 5000
    with pytest.raises(ValueError, match="must be in the range 1..65535"):
        require_tcp_port(70000, field="Modbus port")


def test_redis_state_store_validation_and_payload_rejection() -> None:
    client = MagicMock()
    client.get.return_value = "[]"
    store = RedisStateStore(client=client, ssl=False)
    with pytest.raises(ValueError, match="Redis payload must be a JSON object"):
        store.load_state()

    with pytest.raises(ValueError, match="JSON serializable"):
        store.save_state({"state": {1, 2}})

    with pytest.raises(ValueError, match="non-empty string"):
        RedisStateStore(host="", client=client)


def test_redis_state_store_rejects_plaintext_to_remote_host() -> None:
    with pytest.raises(
        ValueError,
        match="plaintext Redis connections are allowed only for loopback hosts",
    ):
        RedisStateStore(host="redis.internal", ssl=False, client=MagicMock())
