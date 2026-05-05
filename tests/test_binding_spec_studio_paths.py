# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — binding-spec studio path safety tests

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "binding_spec_studio.py"


def _load_studio_module() -> types.ModuleType:
    streamlit_stub = types.ModuleType("streamlit")
    previous_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = streamlit_stub
    try:
        spec = importlib.util.spec_from_file_location("binding_spec_studio", SCRIPT)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous_streamlit


def test_binding_spec_path_uses_fixed_domainpack_root() -> None:
    module = _load_studio_module()

    resolved = module._binding_spec_path_for_domainpack("studio_domain")

    assert resolved == REPO / "domainpacks" / "studio_domain" / "binding_spec.yaml"


@pytest.mark.parametrize("name", ["../outside", "/outside", "bad/name", ""])
def test_binding_spec_path_rejects_path_like_names(name: str) -> None:
    module = _load_studio_module()

    with pytest.raises(ValueError, match="Domainpack name"):
        module._binding_spec_path_for_domainpack(name)
