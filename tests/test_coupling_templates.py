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


def test_duplicate_name_overwrites():
    ts = KnmTemplateSet()
    ts.add(KnmTemplate("x", np.eye(2), np.zeros((2, 2)), "first"))
    ts.add(KnmTemplate("x", np.ones((2, 2)), np.zeros((2, 2)), "second"))
    assert ts.get("x").description == "second"
    np.testing.assert_array_equal(ts.get("x").knm, np.ones((2, 2)))


def test_empty_template_set():
    ts = KnmTemplateSet()
    assert ts.list_names() == []
    with pytest.raises(KeyError):
        ts.get("anything")


def test_template_frozen():
    tpl = KnmTemplate("f", np.eye(2), np.zeros((2, 2)), "frozen")
    with pytest.raises(AttributeError):
        tpl.name = "changed"


def test_get_missing_error_lists_available():
    ts = KnmTemplateSet()
    ts.add(KnmTemplate("alpha", np.eye(2), np.zeros((2, 2)), ""))
    ts.add(KnmTemplate("beta", np.eye(2), np.zeros((2, 2)), ""))
    with pytest.raises(KeyError, match="alpha"):
        ts.get("gamma")
