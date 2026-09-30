# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling template validation contracts

"""Exercise public coupling-template admission, isolation and runtime use."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet
from scpn_phase_orchestrator.upde.engine import UPDEEngine


def test_u1_knm_template_set_rejects_non_square_template() -> None:
    """Refuse a coupling matrix that cannot describe a square topology."""
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
    """Refuse an unrelated object through the public registry boundary."""
    reg = KnmTemplateSet()
    with pytest.raises(TypeError, match="template must be KnmTemplate"):
        reg.add(cast(KnmTemplate, object()))


def test_u1_knm_template_set_add_rejects_blank_template_name() -> None:
    """Refuse a name that becomes empty after canonicalisation."""
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
    """Require a meaningful description before registering a template."""
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
    """Refuse object matrices rather than coercing their elements."""
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
    """Require a matrix rather than a coupling vector."""
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
    """Require a matrix rather than a phase-lag vector."""
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
    """Refuse coupling and phase-lag matrices with different cardinalities."""
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
    """Refuse a coupling matrix containing a missing numerical value."""
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
    """Refuse a phase-lag matrix containing an infinite numerical value."""
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
    """Refuse an empty lookup name without inventing a default template."""
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("")


def test_u1_knm_template_set_get_rejects_whitespace_name() -> None:
    """Refuse a lookup containing only whitespace."""
    with pytest.raises(KeyError, match="non-empty string"):
        KnmTemplateSet().get("   ")


def test_u1_knm_template_set_get_rejects_unknown_name() -> None:
    """Report an unregistered name through the public lookup error."""
    with pytest.raises(KeyError, match="Unknown template"):
        KnmTemplateSet().get("missing")


def test_u1_knm_template_set_get_strips_lookup_name() -> None:
    """Resolve a padded lookup to the registered canonical template."""
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
    """Canonicalise surrounding spaces when storing a template name."""
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
    """Canonicalise surrounding tabs when storing a template name."""
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
    """Return the canonical name inside the retrieved template record."""
    reg = KnmTemplateSet()
    tpl = KnmTemplate(
        name=" k ",
        knm=np.ones((2, 2), dtype=float),
        alpha=np.ones((2, 2), dtype=float),
        description="ok",
    )
    reg.add(tpl)
    assert reg.get("k").name == "k"


@pytest.mark.parametrize("field", ["knm", "alpha"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("replace_existing", [False, True])
@pytest.mark.parametrize("error_mode", ["warn", "raise"])
@pytest.mark.parametrize("dtype", ["float64", "longdouble"])
def test_float64_overflow_refuses_without_changing_registry(
    field: str,
    sign: int,
    replace_existing: bool,
    error_mode: Literal["warn", "raise"],
    dtype: str,
) -> None:
    """Refuse raw or narrowing overflow without mutating either registration."""
    registry = KnmTemplateSet()
    baseline = np.array([[0.0, 0.25], [0.25, 0.0]])
    registry.add(KnmTemplate("baseline", baseline, np.zeros((2, 2)), "original"))
    name = "baseline" if replace_existing else "overflow"
    source_dtype = np.dtype(dtype)
    with np.errstate(over="ignore"):
        magnitude = source_dtype.type(np.finfo(np.float64).max) * source_dtype.type(
            2 * sign
        )
    oversized = np.array([[0.0, magnitude], [magnitude, 0.0]], dtype=dtype)
    coupling = oversized if field == "knm" else baseline.copy()
    lag = oversized if field == "alpha" else np.zeros((2, 2))
    coupling_before = coupling.copy()
    lag_before = lag.copy()
    source_is_finite = np.finfo(source_dtype).maxexp > np.finfo(np.float64).maxexp
    assert bool(np.isfinite(oversized).all()) == source_is_finite
    if not source_is_finite:
        refusal = "contain only finite values"
    elif field == "knm":
        refusal = "remain finite in float64"
    else:
        refusal = "alpha must contain finite real values"

    with (
        np.errstate(over=error_mode, invalid="raise"),
        pytest.raises(ValueError, match=refusal),
    ):
        registry.add(KnmTemplate(name, coupling, lag, "unrepresentable"))

    np.testing.assert_array_equal(coupling, coupling_before)
    np.testing.assert_array_equal(lag, lag_before)
    assert registry.list_names() == ["baseline"]
    preserved = registry.get("baseline")
    assert preserved.description == "original"
    np.testing.assert_array_equal(preserved.knm, baseline)
    np.testing.assert_array_equal(preserved.alpha, np.zeros((2, 2)))
    if not replace_existing:
        with pytest.raises(KeyError, match="Unknown template"):
            registry.get(name)

    recovery = np.array([[0.0, 0.5], [0.5, 0.0]])
    registry.add(KnmTemplate(name, recovery, np.zeros((2, 2)), "recovered"))
    stored = registry.get(name)
    assert stored.description == "recovered"
    phase = UPDEEngine(2, dt=0.01, method="euler").step(
        np.array([0.0, np.pi / 2]),
        np.zeros(2),
        stored.knm,
        0.0,
        0.0,
        stored.alpha,
    )
    np.testing.assert_allclose(phase, [0.005, np.pi / 2 - 0.005], atol=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64", "longdouble"])
def test_representable_templates_keep_copy_isolation_and_phase_lag(
    dtype: str,
) -> None:
    """Normalise real noncontiguous matrices into independent runtime copies."""
    registry = KnmTemplateSet()
    coupling = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=dtype).T
    lag = np.array([[0.0, -np.pi / 6], [np.pi / 6, 0.0]], dtype=dtype).T
    expected_coupling = coupling.astype(np.float64)
    expected_lag = lag.astype(np.float64)
    assert not coupling.flags.c_contiguous
    assert not lag.flags.c_contiguous

    registry.add(KnmTemplate(" lagged ", coupling, lag, "representable"))
    coupling.fill(0.0)
    lag.fill(0.0)
    stored = registry.get("lagged")
    assert stored.knm.dtype == np.float64
    assert stored.alpha.dtype == np.float64
    assert stored.knm.flags.c_contiguous or stored.knm.flags.f_contiguous
    assert stored.alpha.flags.c_contiguous or stored.alpha.flags.f_contiguous
    np.testing.assert_array_equal(stored.knm, expected_coupling)
    np.testing.assert_array_equal(stored.alpha, expected_lag)

    phase = UPDEEngine(2, dt=0.01, method="euler").step(
        np.array([0.0, np.pi / 2]),
        np.zeros(2),
        stored.knm,
        0.0,
        0.0,
        stored.alpha,
    )
    correction = 0.005 * np.sqrt(3) / 2
    np.testing.assert_allclose(
        phase, [correction, np.pi / 2 - correction], rtol=1e-7, atol=1e-9
    )
    stored.knm.fill(0.0)
    stored.alpha.fill(0.0)
    retrieved = registry.get("lagged")
    np.testing.assert_array_equal(retrieved.knm, expected_coupling)
    np.testing.assert_array_equal(retrieved.alpha, expected_lag)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("next_value", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "longdouble"])
def test_float64_limit_admission_tracks_source_representation(
    sign: int, next_value: bool, dtype: str
) -> None:
    """Accept finite boundary rounding or refuse infinity, then recover."""
    source_dtype = np.dtype(dtype)
    limit = source_dtype.type(np.finfo(np.float64).max) * source_dtype.type(sign)
    if next_value:
        with np.errstate(over="ignore"):
            limit = np.nextafter(limit, source_dtype.type(np.inf) * sign)
    matrix = np.array([[0.0, limit], [limit, 0.0]], dtype=dtype)
    original = matrix.copy()
    with np.errstate(over="ignore"):
        canonical = matrix.astype(np.float64)
    expect_finite = (
        not next_value or np.finfo(source_dtype).maxexp > np.finfo(np.float64).maxexp
    )
    assert bool(np.isfinite(canonical).all()) == expect_finite
    registry = KnmTemplateSet()
    baseline = np.array([[0.0, 0.5], [0.5, 0.0]])
    registry.add(KnmTemplate("limit", baseline, np.zeros((2, 2)), "original"))

    if expect_finite:
        expected_limit = np.finfo(np.float64).max * sign
        np.testing.assert_array_equal(
            canonical, [[0.0, expected_limit], [expected_limit, 0.0]]
        )
        registry.add(KnmTemplate("limit", matrix, matrix.copy(), "finite limit"))
        stored = registry.get("limit")
        assert stored.description == "finite limit"
        assert np.isfinite(stored.knm).all()
        assert np.isfinite(stored.alpha).all()
        np.testing.assert_array_equal(stored.knm, canonical)
        np.testing.assert_array_equal(stored.alpha, canonical)
    else:
        with pytest.raises(ValueError, match="contain only finite values"):
            registry.add(KnmTemplate("limit", matrix, matrix.copy(), "non-finite"))
        preserved = registry.get("limit")
        assert preserved.description == "original"
        np.testing.assert_array_equal(preserved.knm, baseline)
        np.testing.assert_array_equal(preserved.alpha, np.zeros((2, 2)))

    np.testing.assert_array_equal(matrix, original)
    assert registry.list_names() == ["limit"]
    registry.add(KnmTemplate("limit", baseline, np.zeros((2, 2)), "recovered"))
    recovered = registry.get("limit")
    assert recovered.description == "recovered"
    phase = UPDEEngine(2, dt=0.01, method="euler").step(
        np.array([0.0, np.pi / 2]),
        np.zeros(2),
        recovered.knm,
        0.0,
        0.0,
        recovered.alpha,
    )
    np.testing.assert_allclose(phase, [0.005, np.pi / 2 - 0.005], atol=1e-12)
