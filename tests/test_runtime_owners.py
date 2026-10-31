# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Owner routing of real-runtime test cases

"""The owner router sends each case only to profiles that have its runtime."""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from tests._runtime_owners import (
    KERNEL_INSTALLED,
    NEEDS_KERNEL,
    TOOLCHAIN_OWNERS,
    owner_param,
    owner_params,
)


def test_python_owner_carries_no_mark() -> None:
    """The Python owner runs in every profile."""
    param = owner_param("python")
    assert param.values == ("python",)
    assert param.marks == ()
    assert param.id == "python"


def test_rust_owner_skips_only_without_the_installed_kernel() -> None:
    """The Rust owner skips exactly where the kernel distribution is absent."""
    param = owner_param("rust")
    assert param.values == ("rust",)
    assert param.marks == (NEEDS_KERNEL,)
    assert NEEDS_KERNEL.mark.args == (not KERNEL_INSTALLED,)
    assert KERNEL_INSTALLED is (find_spec("spo_kernel") is not None)


@pytest.mark.parametrize("owner", sorted(TOOLCHAIN_OWNERS))
def test_toolchain_owners_run_only_where_selected(owner: str) -> None:
    """A toolchain owner carries the native_runtime marker."""
    param = owner_param(owner)
    assert param.values == (owner,)
    assert [mark.name for mark in param.marks] == ["native_runtime"]
    assert {"mojo", "julia", "go"} == TOOLCHAIN_OWNERS


def test_owner_params_keep_order() -> None:
    """The list helper routes each owner in the given order."""
    params = owner_params(["go", "python", "rust"])
    assert [p.values for p in params] == [("go",), ("python",), ("rust",)]


def test_unknown_owner_is_refused() -> None:
    """An unknown owner name is refused rather than silently unmarked."""
    with pytest.raises(ValueError, match="unknown compute owner: 'cuda'"):
        owner_param("cuda")
