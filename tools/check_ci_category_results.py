#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CI Category Result Gate
"""Fail closed unless every reusable CI category completed successfully."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def unsuccessful_categories(raw_results: str) -> dict[str, str]:
    """Return every absent, malformed, or non-success category result."""
    payload: Any = json.loads(raw_results)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("CATEGORY_RESULTS must be a non-empty JSON object")

    failed: dict[str, str] = {}
    for category, outcome in payload.items():
        if not isinstance(category, str) or not isinstance(outcome, Mapping):
            failed[str(category)] = "malformed"
            continue
        result = outcome.get("result")
        failed[category] = result if isinstance(result, str) else "missing"
        if result == "success":
            failed.pop(category)
    return failed


def main() -> int:
    """Check the GitHub ``needs`` projection supplied by the coordinator."""
    raw_results = os.environ.get("CATEGORY_RESULTS")
    if raw_results is None:
        print("CATEGORY_RESULTS is not set")
        return 1
    try:
        failed = unsuccessful_categories(raw_results)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Invalid CATEGORY_RESULTS: {error}")
        return 1
    if failed:
        for category, result in sorted(failed.items()):
            print(f"CI category {category}: {result}")
        return 1
    print("All CI categories succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "unsuccessful_categories"]
