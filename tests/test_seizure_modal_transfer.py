# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EEG modal-growth transfer benchmark tests

"""Tests for the grid-modal-growth-to-EEG transfer benchmark's pure core.

The band-power modal-growth score, the shared-moat detector record, and the payload
sealing are exercised on synthetic multichannel fixtures with a planted beta-band rise:
a smooth exponential rise the fit-quality gate keeps, and a late step the gate rejects
— the behaviour the honest sealed result reports on the real corpus. The EDF I/O and
corpus orchestration are pragma-excluded, so only the tested pure functions are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.seizure_modal_transfer import (
    detector_record,
    modal_rise_score,
    modal_transfer_payload,
    modal_transfer_verdict,
)

_RATE = 256.0
_SAMPLES = int(60 * _RATE)  # a 60 s segment
_TIME = np.arange(_SAMPLES) / _RATE


def _signal(beta_envelope: np.ndarray, *, channels: int = 4) -> np.ndarray:
    """Return a multichannel signal: a constant delta wave plus an enveloped beta wave.

    As the beta (20 Hz) envelope grows, the beta-to-delta power ratio rises, so the
    modal-growth score reads a positive growth rate.
    """
    delta = 0.5 * np.sin(2.0 * np.pi * 2.0 * _TIME)
    beta = beta_envelope * np.sin(2.0 * np.pi * 20.0 * _TIME)
    return np.stack([delta + beta * (0.8 + 0.1 * i) for i in range(channels)])


def _growing() -> np.ndarray:
    """A smooth exponential beta rise — a well-fit growth the gate keeps."""
    return _signal(np.exp(0.02 * _TIME))


def _flat() -> np.ndarray:
    """A steady beta envelope — no rise."""
    return _signal(np.full(_SAMPLES, np.exp(0.6)))


def _step() -> np.ndarray:
    """A late beta jump — a step-like transient the fit-quality gate rejects."""
    envelope = np.full(_SAMPLES, 0.2)
    envelope[-int(10 * _RATE) :] = 3.0
    return _signal(envelope)


# --------------------------------------------------------------------------- #
# modal_rise_score                                                            #
# --------------------------------------------------------------------------- #


def test_modal_rise_score_reads_a_growing_ratio() -> None:
    for aggregation in ("mean", "focal"):
        assert modal_rise_score(_growing(), rate=_RATE, aggregation=aggregation) > 0.0


def test_modal_rise_score_is_near_zero_for_a_flat_ratio() -> None:
    assert modal_rise_score(_flat(), rate=_RATE, aggregation="focal") == pytest.approx(
        0.0, abs=1e-3
    )


def test_modal_rise_score_gate_keeps_a_smooth_rise() -> None:
    ungated = modal_rise_score(_growing(), rate=_RATE, aggregation="focal", r2_gate=0.0)
    gated = modal_rise_score(_growing(), rate=_RATE, aggregation="focal", r2_gate=0.5)
    assert gated == pytest.approx(ungated, abs=1e-9)
    assert gated > 0.0


def test_modal_rise_score_gate_rejects_a_step_transient() -> None:
    ungated = modal_rise_score(_step(), rate=_RATE, aggregation="focal", r2_gate=0.0)
    gated = modal_rise_score(_step(), rate=_RATE, aggregation="focal", r2_gate=0.5)
    assert ungated > 0.0  # the step's late jump trends upward
    assert gated == 0.0  # but the fit-quality gate rejects it as non-exponential


def test_modal_rise_score_rejects_an_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation must be 'mean' or 'focal'"):
        modal_rise_score(_flat(), rate=_RATE, aggregation="median")


# --------------------------------------------------------------------------- #
# detector_record                                                             #
# --------------------------------------------------------------------------- #


def test_detector_record_leads_a_separated_corpus() -> None:
    record = detector_record(
        [1.0, 1.1, 0.9],  # transition scores, clearly above the null
        [0.0, 0.05, -0.1, 0.02, 0.0, -0.05],
        detector="modal_growth_rise",
        aggregation="focal",
        n_permutations=2000,
        seed=0,
    )
    assert record["detector"] == "modal_growth_rise"
    assert record["aggregation"] == "focal"
    assert record["led"] == 3
    assert record["n_transitions"] == 3
    assert record["achieved_false_alarm"] <= 0.1 + 1e-9
    assert record["significance"]["p_value"] < 0.05


def test_detector_record_rejects_empty_transitions() -> None:
    with pytest.raises(ValueError, match="transition_scores must not be empty"):
        detector_record([], [0.0], detector="d", aggregation="focal")


def test_detector_record_rejects_empty_nulls() -> None:
    with pytest.raises(ValueError, match="null_scores must not be empty"):
        detector_record([1.0], [], detector="d", aggregation="focal")


# --------------------------------------------------------------------------- #
# verdict + payload                                                           #
# --------------------------------------------------------------------------- #


def _focal_record(led: int) -> dict[str, object]:
    return {"led": led, "n_transitions": 6, "aggregation": "focal"}


def test_modal_transfer_verdict_states_the_non_transfer() -> None:
    verdict = modal_transfer_verdict(
        _focal_record(1), _focal_record(0), _focal_record(0)
    )
    assert "does not transfer" in verdict
    assert "0/6" in verdict  # the modal-growth lead
    assert "1/6" in verdict  # the spectral lead


def test_modal_transfer_payload_seals_a_reproducible_hash() -> None:
    records = {"mean": _focal_record(0), "focal": _focal_record(0)}
    spectral = {"mean": _focal_record(1), "focal": _focal_record(1)}
    payload = modal_transfer_payload(
        spectral_records=spectral,
        modal_records=records,
        gated_records=records,
        robustness=[{"window": 1024, "led": 0}],
        exploratory={"segment_seconds": 600.0, "claim": "not claimed"},
        corpus={"n_transitions": 6, "n_nulls": 20},
        operating_point={"window_samples": 1024, "r2_gate": 0.5},
        target_fa=0.1,
        n_permutations=10000,
        seed=0,
    )
    assert payload["benchmark"] == "seizure_modal_transfer"
    assert "does not transfer" in payload["verdict"]
    assert "content_hash" in payload

    from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert stored == canonical_record_hash(sealed)
