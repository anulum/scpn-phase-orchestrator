# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — neural supervisor safety and persistence contracts

"""Prove the neural supervisor's audit boundary and checkpoints fail closed.

``jnp.clip`` passes NaN through, so the audit projection handed on a NaN control
marked as projected and not rejected; a negative configured bound inverted the
clip and produced a control from a zero proposal. A checkpoint is two files
replaced one after the other, so a crash between them left a new payload beside
old metadata, which loaded silently with the wrong update count. Every case
uses real policies, real checkpoint files and the public entry points.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scpn_phase_orchestrator.autotune.reward import KnobPolicyCandidate
from scpn_phase_orchestrator.nn.supervisor import (
    DifferentiableSupervisorConfig,
    DifferentiableSupervisorPolicy,
    load_supervisor_ppo_checkpoint,
    project_supervisor_action_for_audit,
    save_supervisor_ppo_checkpoint,
)
from scpn_phase_orchestrator.nn.supervisor._types import SupervisorAction
from scpn_phase_orchestrator.nn.supervisor.candidate_bridge import (
    supervisor_action_to_candidate,
)

CONFIG = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8, hidden_depth=1)
POLICY = DifferentiableSupervisorPolicy(CONFIG, key=jax.random.PRNGKey(0))
OPTIMIZER = optax.adam(1e-3)
OPT_STATE = OPTIMIZER.init(eqx.filter(POLICY, eqx.is_array))


def _action(k: float, zeta: float, layers: list[float]) -> SupervisorAction:
    return SupervisorAction(
        jnp.asarray(k), jnp.asarray(zeta), jnp.asarray(layers), jnp.asarray(0.0)
    )


def _save(
    directory: Path,
    *,
    n_updates: int = 1,
    key: jax.Array | None = None,
    policy: DifferentiableSupervisorPolicy = POLICY,
) -> Path:
    return save_supervisor_ppo_checkpoint(
        directory,
        policy=policy,
        opt_state=OPT_STATE,
        key=jax.random.PRNGKey(1) if key is None else key,
        n_updates=n_updates,
        loss_history=jnp.zeros(3),
    )


def _load(directory: Path) -> object:
    return load_supervisor_ppo_checkpoint(
        directory, template_policy=POLICY, template_opt_state=OPT_STATE
    )


def _rewrite_metadata(directory: Path, **changes: object) -> None:
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for name, value in changes.items():
        if value is _DROP:
            metadata.pop(name, None)
        else:
            metadata[name] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")


_DROP = object()


class TestCheckpointIntegrity:
    def test_payload_beside_older_metadata_is_refused(self, tmp_path: Path) -> None:
        """The on-disk state a crash between the two replacements leaves."""
        _save(tmp_path / "a", n_updates=1)
        shifted = eqx.tree_at(lambda policy: policy.log_std, POLICY, POLICY.log_std + 1)
        _save(tmp_path / "b", n_updates=7, policy=shifted)
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        shutil.copy(tmp_path / "b" / "state.eqx", mixed / "state.eqx")
        shutil.copy(tmp_path / "a" / "metadata.json", mixed / "metadata.json")

        with pytest.raises(ValueError, match="does not match its metadata"):
            _load(mixed)

    def test_a_changed_payload_byte_is_refused(self, tmp_path: Path) -> None:
        directory = _save(tmp_path / "c")
        payload = bytearray((directory / "state.eqx").read_bytes())
        payload[-5] ^= 0xFF
        (directory / "state.eqx").write_bytes(bytes(payload))

        with pytest.raises(ValueError, match="does not match its metadata"):
            _load(directory)

    @pytest.mark.parametrize("digest", [_DROP, "AB" * 32, "ab" * 31, 5])
    def test_digest_must_be_a_lowercase_sha256(
        self, tmp_path: Path, digest: object
    ) -> None:
        directory = _save(tmp_path / "c")
        _rewrite_metadata(directory, state_sha256=digest)
        with pytest.raises(ValueError, match="state_sha256 must be"):
            _load(directory)

    def test_schema_version_one_still_loads_unverified(self, tmp_path: Path) -> None:
        directory = _save(tmp_path / "legacy", n_updates=3)
        _rewrite_metadata(
            directory, schema_version=1, state_sha256=_DROP, key_impl=_DROP
        )
        assert _load(directory).n_updates == 3

    def test_boolean_schema_version_is_unsupported(self, tmp_path: Path) -> None:
        directory = _save(tmp_path / "c")
        _rewrite_metadata(directory, schema_version=True)
        with pytest.raises(ValueError, match="schema is not supported"):
            _load(directory)

    def test_missing_update_count_is_a_value_error(self, tmp_path: Path) -> None:
        directory = _save(tmp_path / "c")
        _rewrite_metadata(directory, n_updates=_DROP)
        with pytest.raises(ValueError, match="n_updates must be a non-negative"):
            _load(directory)


class TestCheckpointTypedKeys:
    @pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
    def test_typed_key_round_trips(self, tmp_path: Path, impl: str) -> None:
        key = jax.random.key(3, impl=impl)
        directory = _save(tmp_path / impl, key=key)

        restored = _load(directory).key

        assert jnp.issubdtype(restored.dtype, jax.dtypes.prng_key)
        assert str(jax.random.key_impl(restored)) == impl
        assert jnp.array_equal(jax.random.key_data(restored), jax.random.key_data(key))

    def test_raw_key_stays_raw(self, tmp_path: Path) -> None:
        directory = _save(tmp_path / "raw")
        restored = _load(directory).key
        assert not jnp.issubdtype(restored.dtype, jax.dtypes.prng_key)
        assert jnp.array_equal(restored, jax.random.PRNGKey(1))

    @pytest.mark.parametrize(
        ("impl", "message"),
        [(7, "key_impl must be"), ("no-such-prng", "cannot wrap the key")],
    )
    def test_key_implementation_is_validated(
        self, tmp_path: Path, impl: object, message: str
    ) -> None:
        directory = _save(tmp_path / "typed", key=jax.random.key(3))
        _rewrite_metadata(directory, key_impl=impl)
        with pytest.raises(ValueError, match=message):
            _load(directory)


class TestConfig:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_oscillators", 0),
            ("n_oscillators", True),
            ("hidden_width", "32"),
            ("hidden_width", 0),
            ("hidden_depth", -1),
            ("n_layer_controls", 0),
            ("n_layer_controls", 3),
            ("n_layer_controls", 2.0),
        ],
    )
    def test_counts_are_validated(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=f"{field} must be an integer"):
            DifferentiableSupervisorConfig(**{"n_oscillators": 4, field: value})

    @pytest.mark.parametrize(
        "field", ["max_global_delta_K", "max_global_delta_zeta", "max_layer_delta_K"]
    )
    @pytest.mark.parametrize(
        "value", [0, -0.05, float("nan"), float("inf"), "0.1", True]
    )
    def test_action_bounds_are_finite_positive(self, field: str, value: object) -> None:
        """A zero bound makes the action log-probability 0/0 during PPO."""
        with pytest.raises(ValueError, match=f"{field} must be a finite positive"):
            DifferentiableSupervisorConfig(n_oscillators=4, **{field: value})

    @pytest.mark.parametrize(
        "field", ["control_energy_weight", "bad_sync_weight", "smoothness_weight"]
    )
    @pytest.mark.parametrize("value", [-0.05, float("nan"), float("inf"), "0.1", True])
    def test_loss_weights_are_finite_non_negative(
        self, field: str, value: object
    ) -> None:
        with pytest.raises(ValueError, match=f"{field} must be a finite non-negative"):
            DifferentiableSupervisorConfig(n_oscillators=4, **{field: value})

    def test_a_zero_loss_weight_switches_the_term_off(self) -> None:
        config = DifferentiableSupervisorConfig(n_oscillators=4, smoothness_weight=0)
        assert config.smoothness_weight == 0.0

    def test_numpy_values_are_normalised(self) -> None:
        config = DifferentiableSupervisorConfig(
            n_oscillators=np.int64(4), max_global_delta_K=np.float32(0.25)
        )
        assert type(config.n_oscillators) is int
        assert type(config.max_global_delta_K) is float


class TestAuditProjection:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_proposal_is_rejected_to_zero(self, bad: float) -> None:
        projection = project_supervisor_action_for_audit(
            _action(0.01, 0.0, [0.0, bad]), CONFIG
        )

        assert projection.audit_record["rejected"] is True
        assert projection.audit_record["rejection_reasons"] == ["non_finite_proposal"]
        values = [
            float(projection.action.delta_K_global),
            float(projection.action.delta_zeta_global),
            *(float(value) for value in projection.action.delta_K_layers),
        ]
        assert values == [0.0, 0.0, 0.0, 0.0]

    def test_non_finite_previous_action_is_rejected_to_zero(self) -> None:
        projection = project_supervisor_action_for_audit(
            _action(0.01, 0.0, [0.0, 0.0]),
            CONFIG,
            previous_action=_action(float("nan"), 0.0, [0.0, 0.0]),
        )

        assert projection.audit_record["rejection_reasons"] == [
            "non_finite_previous_action"
        ]
        assert float(projection.action.delta_K_global) == 0.0

    def test_every_reason_is_reported(self) -> None:
        projection = project_supervisor_action_for_audit(
            _action(float("nan"), 0.0, [0.0, 0.0]),
            CONFIG,
            regime_churn_score=0.9,
            max_regime_churn=0.5,
        )
        assert projection.audit_record["rejection_reasons"] == [
            "non_finite_proposal",
            "regime_churn",
        ]

    @pytest.mark.parametrize("previous", [False, True])
    def test_component_count_must_match_the_config(self, previous: bool) -> None:
        wrong = _action(0.0, 0.0, [0.0, 0.0, 0.0])
        right = _action(0.0, 0.0, [0.0, 0.0])
        field = "previous_action" if previous else "action"
        with pytest.raises(ValueError, match=f"{field} has 5 components"):
            project_supervisor_action_for_audit(
                right if previous else wrong,
                CONFIG,
                previous_action=wrong if previous else None,
            )


class TestCandidateBridge:
    def test_base_channels_beyond_the_layer_deltas_are_kept(self) -> None:
        base = KnobPolicyCandidate(channel_weights=(1.0, 1.0, 1.0, 1.0))
        candidate = supervisor_action_to_candidate(
            _action(0.0, 0.0, [0.25, 0.5]), base=base
        )
        assert candidate.channel_weights == (1.25, 1.5, 1.0, 1.0)

    def test_layer_deltas_beyond_the_base_start_from_zero(self) -> None:
        base = KnobPolicyCandidate(channel_weights=(1.0,))
        candidate = supervisor_action_to_candidate(
            _action(0.0, 0.0, [0.25, 0.5]), base=base
        )
        assert candidate.channel_weights == (1.25, 0.5)

    @pytest.mark.parametrize(
        "action",
        [
            _action(float("nan"), 0.0, [0.0]),
            _action(0.0, float("inf"), [0.0]),
            _action(0.0, 0.0, [float("nan")]),
        ],
    )
    def test_non_finite_action_is_refused(self, action: SupervisorAction) -> None:
        with pytest.raises(ValueError, match="NaN or infinite component"):
            supervisor_action_to_candidate(action)
