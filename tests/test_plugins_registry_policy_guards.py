# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin runtime policy validation guards

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
)

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


_HEX = "a" * 64


class TestPluginRuntimeLoadPolicy:
    def test_accepts_defaults(self) -> None:
        policy = PluginRuntimeLoadPolicy()
        assert policy.loading_permitted is False

    def test_rejects_non_boolean_loading_permitted(self) -> None:
        with pytest.raises(TypeError, match="loading_permitted must be a boolean"):
            _call(PluginRuntimeLoadPolicy, loading_permitted="yes")

    def test_rejects_non_boolean_require_package_target(self) -> None:
        with pytest.raises(TypeError, match="require_package_target must be a boolean"):
            _call(PluginRuntimeLoadPolicy, require_package_target="yes")

    def test_rejects_empty_allowed_kinds(self) -> None:
        with pytest.raises(ValueError, match="allowed_kinds must not be empty"):
            PluginRuntimeLoadPolicy(allowed_kinds=())

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(ValueError, match="unsupported runtime load kind"):
            _call(PluginRuntimeLoadPolicy, allowed_kinds=("wizard",))


class TestPluginRuntimeExecutionPolicy:
    def test_accepts_defaults(self) -> None:
        policy = PluginRuntimeExecutionPolicy()
        assert policy.execution_permitted is False

    def test_rejects_non_boolean_execution_permitted(self) -> None:
        with pytest.raises(TypeError, match="execution_permitted must be a boolean"):
            _call(PluginRuntimeExecutionPolicy, execution_permitted="yes")

    def test_rejects_non_boolean_require_target_hash_approval(self) -> None:
        with pytest.raises(
            TypeError, match="require_target_hash_approval must be a boolean"
        ):
            _call(PluginRuntimeExecutionPolicy, require_target_hash_approval="yes")

    def test_rejects_invalid_approved_target_hash(self) -> None:
        with pytest.raises(ValueError, match="approved target hash"):
            PluginRuntimeExecutionPolicy(approved_target_hashes=("not-a-digest",))

    def test_rejects_required_approval_without_hashes(self) -> None:
        with pytest.raises(
            ValueError, match="approved_target_hashes must not be empty"
        ):
            PluginRuntimeExecutionPolicy(
                require_target_hash_approval=True,
                approved_target_hashes=(),
            )

    def test_accepts_required_approval_with_hashes(self) -> None:
        policy = PluginRuntimeExecutionPolicy(
            require_target_hash_approval=True,
            approved_target_hashes=(_HEX,),
        )
        assert policy.require_target_hash_approval is True
