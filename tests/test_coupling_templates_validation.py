# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling template validation contracts

"""
Validation contracts for KnmTemplate and KnmTemplateSet matrix/template boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet


def test_u1_knm_template_set_rejects_non_square_template() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="square"):
        reg.add(
            KnmTemplate(
                name="bad",
                knm=np.ones((2, 3), dtype=float),
                alpha=np.ones((2, 3), dtype=float),
                description="invalid",
            )
        )


def test_u1_knm_template_set_add_rejects_non_template_payload() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(TypeError, match="template must be KnmTemplate"):
        reg.add(object())  # type: ignore[arg-type]


def test_u1_knm_template_set_add_rejects_blank_template_name() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="template name must be a non-empty string"):
        reg.add(
            KnmTemplate(
                name=" ",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.ones((2, 2), dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_rejects_blank_description() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="description"):
        reg.add(
            KnmTemplate(
                name="k",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.ones((2, 2), dtype=float),
                description="",
            )
        )


def test_u1_knm_template_set_rejects_non_float_dtype() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="floating-point dtypes"):
        reg.add(
            KnmTemplate(
                name="bad_dtype",
                knm=np.ones((2, 2), dtype=object),
                alpha=np.ones((2, 2), dtype=object),
                description="invalid",
            )
        )


def test_u1_knm_template_set_add_rejects_non_2d_knm() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="must be 2D matrices"):
        reg.add(
            KnmTemplate(
                name="non2d_knm",
                knm=np.ones((2,), dtype=float),
                alpha=np.ones((2, 2), dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_add_rejects_non_2d_alpha() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="must be 2D matrices"):
        reg.add(
            KnmTemplate(
                name="non2d_alpha",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.ones((2,), dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_add_rejects_shape_mismatch() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="must have identical shapes"):
        reg.add(
            KnmTemplate(
                name="shape_mismatch",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.ones((3, 3), dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_add_rejects_non_finite_knm() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="contain only finite values"):
        reg.add(
            KnmTemplate(
                name="non_finite_knm",
                knm=np.array([[1.0, 0.0], [0.0, np.nan]], dtype=float),
                alpha=np.ones((2, 2), dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_add_rejects_non_finite_alpha() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="contain only finite values"):
        reg.add(
            KnmTemplate(
                name="non_finite_alpha",
                knm=np.ones((2, 2), dtype=float),
                alpha=np.array([[1.0, 0.0], [0.0, np.inf]], dtype=float),
                description="ok",
            )
        )


def test_u1_knm_template_set_get_rejects_blank_name() -> None:
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("")


def test_u1_knm_template_set_get_rejects_whitespace_name() -> None:
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("   ")


def test_u1_knm_template_set_get_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown template"):
        KnmTemplateSet().get("missing")


def test_u1_knm_template_set_get_strips_lookup_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name="k",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    stored = reg.get(" k ")
    assert stored.name == "k"
    np.testing.assert_array_equal(stored.knm, tpl.knm)
    np.testing.assert_array_equal(stored.alpha, tpl.alpha)
    assert stored.description == tpl.description


def test_u1_knm_template_set_add_strips_storage_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name=" k ",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.list_names() == ["k"]


def test_u1_knm_template_set_add_strips_tabbed_storage_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name="\tk\t",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.list_names() == ["k"]


def test_u1_knm_template_set_stores_canonical_template_name() -> None:
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name=" k ",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
