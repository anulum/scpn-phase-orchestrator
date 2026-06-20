# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin runtime load and execution policies

"""Runtime load and execution policy records governing plugin capability access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._shared import _VALID_KINDS, _validate_sha256

if TYPE_CHECKING:
    from ._shared import PluginKind


_DEFAULT_RUNTIME_LOAD_KINDS: tuple[PluginKind, ...] = (
    "actuator",
    "bridge",
    "extractor",
    "monitor",
)


_DEFAULT_RUNTIME_LOAD_POLICY: PluginRuntimeLoadPolicy


@dataclass(frozen=True)
class PluginRuntimeLoadPolicy:
    """Explicit policy gate for Python-owned plugin runtime loading."""

    loading_permitted: bool = False
    allowed_kinds: tuple[PluginKind, ...] = _DEFAULT_RUNTIME_LOAD_KINDS
    require_package_target: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.loading_permitted, bool):
            raise TypeError("loading_permitted must be a boolean")
        if not isinstance(self.require_package_target, bool):
            raise TypeError("require_package_target must be a boolean")
        if not self.allowed_kinds:
            raise ValueError("allowed_kinds must not be empty")
        for kind in self.allowed_kinds:
            if kind not in _VALID_KINDS:
                raise ValueError(f"unsupported runtime load kind: {kind}")


@dataclass(frozen=True)
class PluginRuntimeExecutionPolicy:
    """Explicit policy gate for invoking Python-owned plugin runtime targets."""

    loading_permitted: bool = False
    execution_permitted: bool = False
    allowed_kinds: tuple[PluginKind, ...] = _DEFAULT_RUNTIME_LOAD_KINDS
    require_package_target: bool = True
    approved_target_hashes: tuple[str, ...] = ()
    require_target_hash_approval: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.execution_permitted, bool):
            raise TypeError("execution_permitted must be a boolean")
        if not isinstance(self.require_target_hash_approval, bool):
            raise TypeError("require_target_hash_approval must be a boolean")
        for target_hash in self.approved_target_hashes:
            _validate_sha256(target_hash, "approved target hash")
        if self.require_target_hash_approval and not self.approved_target_hashes:
            raise ValueError(
                "approved_target_hashes must not be empty when target hash "
                "approval is required"
            )
        PluginRuntimeLoadPolicy(
            loading_permitted=self.loading_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
        )

    def to_load_policy(self) -> PluginRuntimeLoadPolicy:
        """Return the corresponding load policy for target resolution.

        Returns
        -------
        PluginRuntimeLoadPolicy
            The corresponding load policy for target resolution.
        """
        return PluginRuntimeLoadPolicy(
            loading_permitted=self.loading_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
        )


_DEFAULT_RUNTIME_LOAD_POLICY = PluginRuntimeLoadPolicy()
