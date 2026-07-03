# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend delay-embedding validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _embedding_validation as embedding_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_validation import (  # noqa: E501
    validate_delay_embed_backend_inputs,
    validate_delay_embed_backend_output,
    validate_mutual_information_backend_inputs,
    validate_mutual_information_backend_output,
    validate_nearest_neighbor_backend_inputs,
    validate_nearest_neighbor_backend_outputs,
)

_SIGNAL = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)


class TestNumericStringAliasHelpers:
    def test_direct_helpers_distinguish_numeric_strings(self) -> None:
        assert embedding_validation._is_numeric_string_alias(0.5) is False
        assert embedding_validation._is_numeric_string_alias("not-a-number") is False
        assert (
            embedding_validation._contains_numeric_string_alias(
                np.array([0.0, "1.0"], dtype=object)
            )
            is True
        )
        assert (
            embedding_validation._contains_numeric_string_alias(
                np.array([0.0, "not-a-number"], dtype=object)
            )
            is False
        )
        assert (
            embedding_validation._contains_numeric_string_alias(
                np.array([0.0, 1.0], dtype=np.float64)
            )
            is False
        )


class TestDelayEmbedInputs:
    def test_valid_round_trips(self) -> None:
        s, delay, dim, t_eff = validate_delay_embed_backend_inputs(_SIGNAL, 1, 2)
        assert (delay, dim, t_eff) == (1, 2, 4)

    @pytest.mark.parametrize(
        ("signal", "delay", "dim", "match"),
        [
            (np.array([True, False]), 1, 2, "signal must not contain boolean"),
            (np.array([0.1 + 1j, 0.2]), 1, 2, "signal must"),
            (np.array(["0.0", "1.0", "2.0"], dtype=object), 1, 2, "numeric-string"),
            (np.array([object()], dtype=object), 1, 2, "finite one-dimensional"),
            (_SIGNAL, 0, 2, "delay must"),
            (_SIGNAL, 1, 0, "dimension must"),
            (_SIGNAL, 1, 10, "too short for the requested delay embedding"),
        ],
    )
    def test_rejects_corrupt_input(
        self, signal: Any, delay: Any, dim: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_delay_embed_backend_inputs(signal, delay, dim)


class TestMutualInformationInputs:
    def test_valid_round_trips(self) -> None:
        s, lag, n_bins = validate_mutual_information_backend_inputs(_SIGNAL, 1, 4)
        assert (lag, n_bins) == (1, 4)

    def test_rejects_numeric_string_signal(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_mutual_information_backend_inputs(
                np.array(["0.0", "1.0", "0.0", "1.0"], dtype=object),
                1,
                4,
            )

    @pytest.mark.parametrize(
        ("lag", "n_bins", "match"),
        [
            (-1, 4, "lag must"),
            (1, 1, "n_bins must"),
        ],
    )
    def test_rejects_invalid_dimensions(
        self, lag: Any, n_bins: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_mutual_information_backend_inputs(_SIGNAL, lag, n_bins)


class TestNearestNeighborInputs:
    def test_valid_round_trips(self) -> None:
        e, t, m = validate_nearest_neighbor_backend_inputs(np.zeros(4), 2, 2)
        assert (t, m) == (2, 2)

    def test_rejects_numeric_string_embedded(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_nearest_neighbor_backend_inputs(
                np.array(["0.0", "1.0", "2.0", "3.0"], dtype=object),
                2,
                2,
            )

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match t.m"):
            validate_nearest_neighbor_backend_inputs(np.zeros(3), 2, 2)


class TestDelayEmbedOutput:
    def test_rejects_numeric_string_output(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_delay_embed_backend_output(
                np.array([["0.0", "1.0"], ["1.0", "2.0"]], dtype=object),
                signal=_SIGNAL,
                delay=1,
                dimension=2,
                t_effective=2,
            )

    def test_rejects_uncoercible_output(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            validate_delay_embed_backend_output(
                np.array([[object()]], dtype=object),
                signal=_SIGNAL,
                delay=1,
                dimension=1,
                t_effective=1,
            )


class TestMutualInformationOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_mutual_information_backend_output(0.5) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array(True), "must not be boolean"),
            (np.array(0.5 + 1j), "must be real"),
            (np.array("0.5", dtype=object), "numeric-string"),
            (np.array(object(), dtype=object), "numeric"),
            (np.array([0.5, 0.6]), "must be scalar"),
            (-0.5, "must be non-negative"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_mutual_information_backend_output(value)


class TestNearestNeighborOutput:
    @pytest.mark.parametrize(
        ("distances", "indices", "match"),
        [
            (
                np.array(["1.0", "2.0"], dtype=object),
                np.array([1.0, 0.0]),
                "numeric-string",
            ),
            (
                np.array([1.0, 2.0]),
                np.array(["1", "0"], dtype=object),
                "numeric-string",
            ),
        ],
    )
    def test_rejects_numeric_string_payloads(
        self,
        distances: Any,
        indices: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_nearest_neighbor_backend_outputs(distances, indices, t=2)

    def test_rejects_uncoercible_payloads(self) -> None:
        with pytest.raises(ValueError, match="distances must be numeric"):
            validate_nearest_neighbor_backend_outputs(
                np.array([object()], dtype=object),
                np.array([0.0]),
                t=1,
            )
        with pytest.raises(ValueError, match="indices must be numeric"):
            validate_nearest_neighbor_backend_outputs(
                np.array([1.0]),
                np.array([object()], dtype=object),
                t=1,
            )
