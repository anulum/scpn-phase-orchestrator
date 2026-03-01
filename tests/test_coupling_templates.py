from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet


def test_add_and_get():
    ts = KnmTemplateSet()
    knm = np.eye(3)
    alpha = np.zeros((3, 3))
    tpl = KnmTemplate(name="id", knm=knm, alpha=alpha, description="identity")
    ts.add(tpl)
    retrieved = ts.get("id")
    np.testing.assert_array_equal(retrieved.knm, knm)


def test_list_names():
    ts = KnmTemplateSet()
    ts.add(KnmTemplate("a", np.eye(2), np.zeros((2, 2)), ""))
    ts.add(KnmTemplate("b", np.eye(2), np.zeros((2, 2)), ""))
    assert sorted(ts.list_names()) == ["a", "b"]


def test_get_missing_raises():
    ts = KnmTemplateSet()
    with pytest.raises(KeyError):
        ts.get("nope")
