# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for multiverse counterfactual rollouts

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.multiverse import (
    MultiverseBranchSpec,
    simulate_multiverse_counterfactual_branches,
)


def _base_inputs():
    phases = np.array([0.10, 1.20, 2.40], dtype=np.float64)
    omegas = np.array([0.05, -0.02, 0.01], dtype=np.float64)
    baseline_k = np.array(
        [[0.0, 0.15, 0.15], [0.15, 0.0, 0.15], [0.15, 0.15, 0.0]],
        dtype=np.float64,
    )
    baseline_alpha = np.zeros((3, 3), dtype=np.float64)
    return phases, omegas, baseline_k, baseline_alpha


class TestMultiverseCounterfactualRollouts:
    def test_simulate_multiverse_counterfactual_branches_is_deterministic(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        branches = (
            (),
            (ControlAction("K", "global", 0.25, 1.0, "baseline uplift"),),
        )

        first = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_action_sets=branches,
            horizon=8,
            dt=0.02,
        )
        second = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_action_sets=branches,
            horizon=8,
            dt=0.02,
        )

        assert first.manifest_hash == second.manifest_hash
        assert first.branch_count == 2
        assert second.branch_count == 2
        assert json.loads(
            json.dumps(first.to_audit_record(), allow_nan=False)
        ) == json.loads(json.dumps(second.to_audit_record(), allow_nan=False))

    def test_manifest_is_json_safe_with_non_actuation_boundary(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_action_sets=((ControlAction("zeta", "global", 0.10, 1.0, "bias"),),),
            horizon=4,
            dt=0.015,
        )
        record = manifest.to_audit_record()

        assert record["backend"] == "numpy_vectorized"
        assert record["branch_count"] == 1
        assert record["non_actuating"] is True
        assert record["execution_disabled"] is True
        assert (
            record["claim_boundary"]
            == "counterfactual_branch_rollout_not_live_actuation"
        )
        assert isinstance(record["manifest_hash"], str)
        assert len(record["manifest_hash"]) == 64
        assert json.loads(json.dumps(record, allow_nan=False))

        branch_record = record["branch_records"][0]
        for key in ("final_R", "mean_R", "min_R", "max_R", "final_psi"):
            assert np.isfinite(branch_record[key])
        assert isinstance(branch_record["action_labels"], list)
        assert branch_record["action_count"] == 1
        assert branch_record["topology_edge_count"] >= 0

    def test_simulate_multiverse_counterfactual_does_not_mutate_inputs(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        snapshot = {
            "phases": phases.copy(),
            "omegas": omegas.copy(),
            "baseline_k": baseline_k.copy(),
            "baseline_alpha": baseline_alpha.copy(),
        }

        simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_specs=(
                MultiverseBranchSpec(
                    branch_id="a",
                    actions=(
                        ControlAction("alpha", "oscillator_0", 0.02, 1.0, "phase lag"),
                    ),
                ),
            ),
            horizon=5,
            dt=0.01,
        )

        assert np.array_equal(phases, snapshot["phases"])
        assert np.array_equal(omegas, snapshot["omegas"])
        assert np.array_equal(baseline_k, snapshot["baseline_k"])
        assert np.array_equal(baseline_alpha, snapshot["baseline_alpha"])

    @pytest.mark.parametrize(
        ("bad_input", "message"),
        [
            (
                {
                    "branch_action_sets": (),
                },
                "at least one branch specification is required",
            ),
            (
                {
                    "branch_action_sets": (
                        (ControlAction("K", "global", 0.25, 1.0, "bad"),),
                    ),
                    "baseline_k": np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float64),
                },
                "baseline_k.shape",
            ),
            (
                {
                    "branch_action_sets": (
                        (ControlAction("bad", "global", 0.1, 1.0, "bad"),),
                    ),
                },
                "unsupported knob",
            ),
            (
                {
                    "branch_action_sets": (
                        (ControlAction("K", "global", 0.1, 1.0, "bad"),),
                    ),
                    "horizon": 0,
                },
                "horizon",
            ),
            (
                {
                    "branch_action_sets": (
                        (ControlAction("K", "global", 0.1, 1.0, "bad"),),
                    ),
                    "dt": 0.0,
                },
                "dt",
            ),
            (
                {
                    "branch_action_sets": (
                        (ControlAction("K", "global", 0.1, 1.0, "bad"),),
                    ),
                    "backend": "invalid",
                },
                "backend",
            ),
        ],
    )
    def test_fail_closed_on_invalid_inputs(self, bad_input, message: str) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        valid_branch_actions = (
            (ControlAction("zeta", "global", 0.0, 1.0, "default"),),
        )
        options = {
            "phases": phases,
            "omegas": omegas,
            "baseline_k": baseline_k,
            "baseline_alpha": baseline_alpha,
            "branch_action_sets": valid_branch_actions,
        }
        options.update(bad_input)
        options["branch_action_sets"] = bad_input.get(
            "branch_action_sets",
            options["branch_action_sets"],
        )

        with pytest.raises(ValueError, match=message):
            simulate_multiverse_counterfactual_branches(**options)

    def test_k_increase_branch_changes_final_order_parameter(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_action_sets=(
                (),
                (ControlAction("K", "global", 0.30, 1.0, "increase coupling"),),
            ),
            horizon=10,
            dt=0.02,
        )
        baseline_final = manifest.branch_records[0].final_R
        boosted_final = manifest.branch_records[1].final_R

        assert not np.isclose(baseline_final, boosted_final)

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("baseline_k", "baseline_k diagonal"),
            ("baseline_alpha", "baseline_alpha diagonal"),
        ],
    )
    def test_rejects_self_coupling_baseline_matrices(
        self, field: str, message: str
    ) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        if field == "baseline_k":
            baseline_k = baseline_k.copy()
            baseline_k[1, 1] = 0.01
        else:
            baseline_alpha = baseline_alpha.copy()
            baseline_alpha[2, 2] = 0.01

        with pytest.raises(ValueError, match=message):
            simulate_multiverse_counterfactual_branches(
                phases=phases,
                omegas=omegas,
                baseline_k=baseline_k,
                baseline_alpha=baseline_alpha,
                branch_action_sets=((ControlAction("K", "global", 0.1, 1.0, "bad"),),),
            )

    def test_rejects_self_coupling_topology_masks(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        topology_mask = np.ones_like(baseline_k)
        topology_mask[0, 0] = 1.0

        with pytest.raises(ValueError, match="topology_mask diagonal"):
            simulate_multiverse_counterfactual_branches(
                phases=phases,
                omegas=omegas,
                baseline_k=baseline_k,
                baseline_alpha=baseline_alpha,
                branch_action_sets=(
                    (ControlAction("K", "global", 0.1, 1.0, "bad mask"),),
                ),
                topology_masks=(topology_mask,),
            )

    def test_matrix_branch_actions_preserve_off_diagonal_kuramoto_graph(self) -> None:
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        zero_k = np.zeros_like(baseline_k)
        zero_alpha = np.zeros_like(baseline_alpha)
        off_diagonal = np.ones_like(baseline_k)
        np.fill_diagonal(off_diagonal, 0.0)

        action_manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=zero_k,
            baseline_alpha=zero_alpha,
            branch_action_sets=(
                (
                    ControlAction("K", "global", 0.3, 1.0, "global coupling"),
                    ControlAction("alpha", "global", 0.5, 1.0, "global lag"),
                ),
            ),
            horizon=6,
            dt=0.02,
        )
        explicit_manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=0.3 * off_diagonal,
            baseline_alpha=0.5 * off_diagonal,
            branch_action_sets=((),),
            horizon=6,
            dt=0.02,
        )

        action_record = action_manifest.branch_records[0]
        explicit_record = explicit_manifest.branch_records[0]
        assert action_record.final_R == pytest.approx(
            explicit_record.final_R, abs=1e-12
        )
        assert action_record.final_psi == pytest.approx(
            explicit_record.final_psi, abs=1e-12
        )
        assert action_record.topology_edge_count == explicit_record.topology_edge_count
        assert action_record.topology_scale == pytest.approx(
            explicit_record.topology_scale, abs=1e-12
        )

    def test_jax_backend_matches_numpy_branch_invariants(self) -> None:
        pytest.importorskip("jax")
        pytest.importorskip("jax.numpy")
        phases, omegas, baseline_k, baseline_alpha = _base_inputs()
        branches = (
            MultiverseBranchSpec("baseline", ()),
            MultiverseBranchSpec(
                "stronger_coupling",
                (
                    ControlAction("K", "global", 0.14, 1.0, "increase coupling"),
                    ControlAction("alpha", "global", 0.04, 1.0, "phase lag"),
                    ControlAction("zeta", "global", 0.015, 1.0, "forcing"),
                    ControlAction("Psi", "global", 0.2, 1.0, "forcing phase"),
                ),
            ),
        )

        numpy_manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_specs=branches,
            horizon=12,
            dt=0.015,
            backend="numpy",
        )
        jax_manifest = simulate_multiverse_counterfactual_branches(
            phases=phases,
            omegas=omegas,
            baseline_k=baseline_k,
            baseline_alpha=baseline_alpha,
            branch_specs=branches,
            horizon=12,
            dt=0.015,
            backend="jax",
        )

        assert numpy_manifest.backend == "numpy_vectorized"
        assert jax_manifest.backend == "jax_vectorized"
        assert jax_manifest.non_actuating is True
        assert jax_manifest.execution_disabled is True
        assert jax_manifest.branch_count == numpy_manifest.branch_count
        for numpy_record, jax_record in zip(
            numpy_manifest.branch_records,
            jax_manifest.branch_records,
            strict=True,
        ):
            assert jax_record.branch_id == numpy_record.branch_id
            assert jax_record.branch_hash == numpy_record.branch_hash
            assert jax_record.action_count == numpy_record.action_count
            assert jax_record.action_labels == numpy_record.action_labels
            assert jax_record.topology_edge_count == numpy_record.topology_edge_count
            assert jax_record.topology_scale == pytest.approx(
                numpy_record.topology_scale,
                abs=1e-12,
            )
            assert jax_record.final_R == pytest.approx(numpy_record.final_R, abs=1e-10)
            assert jax_record.mean_R == pytest.approx(numpy_record.mean_R, abs=1e-10)
            assert jax_record.min_R == pytest.approx(numpy_record.min_R, abs=1e-10)
            assert jax_record.max_R == pytest.approx(numpy_record.max_R, abs=1e-10)
            assert jax_record.final_psi == pytest.approx(
                numpy_record.final_psi,
                abs=1e-10,
            )
