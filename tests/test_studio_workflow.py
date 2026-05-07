# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio workflow serialisation tests

from __future__ import annotations

import json
import math
from contextlib import suppress

import pytest

from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ExportManifest,
    ImportedSourceSummary,
    RuntimeSnapshot,
    StudioProjectState,
)


def test_project_state_serialises_with_stable_hashes() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a,b\n0,0.0,1.0\n1,0.2,0.8\n",
        channel_count=2,
        sample_count=2,
    )
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        validation_errors=(),
        inferred_channels=("P", "I"),
        confidence_factors={"phase_quality": 0.75},
        provenance={"source_kind": source.source_kind},
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
        layer_metrics=(("grid", 0.8),),
        hierarchy_watermarks={"edge-a": 3},
        replay_status="proposal_only",
    )
    manifest = ExportManifest.review_artifact(
        target_kind="binding_spec",
        file_name="binding_spec.yaml",
        payload="version: 1\nname: demo\n",
        command="spo run binding_spec.yaml --audit audit.jsonl",
    )
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        exports=(manifest,),
    )

    record = state.to_audit_record()
    restored = json.loads(json.dumps(record))

    assert restored["project_name"] == "demo"
    assert restored["source"]["sha256"] == source.sha256
    assert restored["exports"][0]["payload_sha256"] == manifest.payload_sha256
    assert restored["exports"][0]["safety_posture"] == "review_artifact"
    assert restored["runtime"]["hierarchy_watermarks"] == {"edge-a": 3}


def test_export_manifest_rejects_deployable_without_warning() -> None:
    with pytest.raises(ValueError, match="deployable exports require warnings"):
        ExportManifest(
            target_kind="docker",
            file_name="deploy.json",
            payload="{}",
            command="docker compose up",
            safety_posture="deployable",
            warnings=(),
        )


def test_binding_proposal_rejects_non_json_safe_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            provenance={"unsafe": object()},
        )


def test_binding_proposal_rejects_non_mapping_audit_mappings() -> None:
    with pytest.raises(ValueError, match="confidence_factors"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            confidence_factors=object(),
        )
    with pytest.raises(ValueError, match="provenance"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            provenance=object(),
        )


def test_binding_proposal_rejects_non_string_sequence_fields() -> None:
    with pytest.raises(ValueError, match="validation_errors"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            validation_errors=(object(),),
        )
    with pytest.raises(ValueError, match="inferred_channels"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            inferred_channels=(object(),),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_binding_proposal_rejects_non_finite_confidence_factors(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_factors"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            confidence_factors={"phase_quality": value},
        )


def test_binding_proposal_rejects_non_string_confidence_factor_keys() -> None:
    with pytest.raises(ValueError, match="confidence_factors"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            confidence_factors={1: 0.5},
        )


def test_binding_proposal_rejects_bool_confidence_factors() -> None:
    with pytest.raises(ValueError, match="confidence_factors"):
        BindingProposal(
            yaml_text="version: 1\nname: demo\n",
            confidence_factors={"phase_quality": True},
        )


@pytest.mark.parametrize("field_name", ["R", "Psi", "K", "alpha", "zeta"])
def test_runtime_snapshot_rejects_non_finite_telemetry(field_name: str) -> None:
    values = {
        "R": 0.81,
        "Psi": 0.12,
        "K": 1.4,
        "alpha": 0.0,
        "zeta": 0.2,
    }
    values[field_name] = math.nan

    with pytest.raises(ValueError, match=field_name):
        RuntimeSnapshot(
            **values,
            regime="nominal",
        )


@pytest.mark.parametrize("field_name", ["R", "Psi", "K", "alpha", "zeta"])
def test_runtime_snapshot_rejects_bool_telemetry(field_name: str) -> None:
    values = {
        "R": 0.81,
        "Psi": 0.12,
        "K": 1.4,
        "alpha": 0.0,
        "zeta": 0.2,
    }
    values[field_name] = True

    with pytest.raises(ValueError, match=field_name):
        RuntimeSnapshot(
            **values,
            regime="nominal",
        )


def test_runtime_snapshot_rejects_non_finite_layer_metric() -> None:
    with pytest.raises(ValueError, match="layer_metrics"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            layer_metrics=(("grid", math.inf),),
        )


def test_runtime_snapshot_rejects_bool_layer_metric() -> None:
    with pytest.raises(ValueError, match="layer_metrics"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            layer_metrics=(("grid", True),),
        )


def test_runtime_snapshot_rejects_non_string_status_fields() -> None:
    with pytest.raises(ValueError, match="regime"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime=object(),
        )
    with pytest.raises(ValueError, match="replay_status"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            replay_status=object(),
        )


@pytest.mark.parametrize("layer", ["", object()])
def test_runtime_snapshot_rejects_invalid_layer_metric_names(layer: object) -> None:
    with pytest.raises(ValueError, match="layer_metrics"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            layer_metrics=((layer, 0.8),),
        )


@pytest.mark.parametrize(
    "layer_metrics",
    [object(), "bad", (("ok", 1.0, 2.0),)],
)
def test_runtime_snapshot_rejects_malformed_layer_metrics(
    layer_metrics: object,
) -> None:
    with pytest.raises(ValueError, match="layer_metrics"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            layer_metrics=layer_metrics,
        )


@pytest.mark.parametrize("value", [True, 1.0, math.nan, -1, object()])
def test_runtime_snapshot_rejects_invalid_hierarchy_watermarks(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="hierarchy_watermarks"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            hierarchy_watermarks={"edge-a": value},
        )


def test_runtime_snapshot_rejects_non_mapping_hierarchy_watermarks() -> None:
    with pytest.raises(ValueError, match="hierarchy_watermarks"):
        RuntimeSnapshot(
            R=0.81,
            Psi=0.12,
            K=1.4,
            alpha=0.0,
            zeta=0.2,
            regime="nominal",
            hierarchy_watermarks=object(),
        )


def test_finite_confidence_and_runtime_floats_serialise() -> None:
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        confidence_factors={"phase_quality": 0.75},
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
        layer_metrics=(("grid", 0.8),),
        hierarchy_watermarks={"edge-a": 3},
    )

    record = {
        "binding": proposal.to_audit_record(),
        "runtime": snapshot.to_audit_record(),
    }
    restored = json.loads(json.dumps(record, allow_nan=False))

    assert restored["binding"]["confidence_factors"] == {"phase_quality": 0.75}
    assert restored["runtime"]["R"] == 0.81
    assert restored["runtime"]["layer_metrics"] == [{"layer": "grid", "value": 0.8}]
    assert restored["runtime"]["hierarchy_watermarks"] == {"edge-a": 3}


def test_project_state_rejects_non_json_safe_metadata() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(yaml_text="version: 1\nname: demo\n")
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )

    with pytest.raises(ValueError, match="metadata"):
        StudioProjectState(
            project_name="demo",
            source=source,
            binding=proposal,
            runtime=snapshot,
            metadata={"unsafe": object()},
        )


def test_project_state_rejects_non_mapping_metadata() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(yaml_text="version: 1\nname: demo\n")
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )

    with pytest.raises(ValueError, match="metadata"):
        StudioProjectState(
            project_name="demo",
            source=source,
            binding=proposal,
            runtime=snapshot,
            metadata=object(),
        )


def test_source_export_and_project_reject_non_string_scalars() -> None:
    with pytest.raises(ValueError, match="source_kind"):
        ImportedSourceSummary.from_payload(
            source_kind=object(),
            payload=b"t,a\n0,0.0\n",
            channel_count=1,
            sample_count=1,
        )
    with pytest.raises(ValueError, match="target_kind"):
        ExportManifest.review_artifact(
            target_kind=object(),
            file_name="binding_spec.yaml",
            payload="version: 1\n",
            command="spo run binding_spec.yaml",
        )
    with pytest.raises(ValueError, match="file_name"):
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name=object(),
            payload="version: 1\n",
            command="spo run binding_spec.yaml",
        )
    with pytest.raises(ValueError, match="payload"):
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload=object(),
            command="spo run binding_spec.yaml",
        )
    with pytest.raises(ValueError, match="command"):
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload="version: 1\n",
            command=object(),
        )

    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(yaml_text="version: 1\nname: demo\n")
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )
    with pytest.raises(ValueError, match="project_name"):
        StudioProjectState(
            project_name=object(),
            source=source,
            binding=proposal,
            runtime=snapshot,
        )


@pytest.mark.parametrize("sha256", ["0" * 63, "0" * 65, "g" * 64])
def test_imported_source_rejects_malformed_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        ImportedSourceSummary(
            source_kind="time_series_csv",
            sha256=sha256,
            byte_count=10,
            channel_count=2,
            sample_count=3,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("byte_count", True),
        ("byte_count", 1.0),
        ("byte_count", object()),
        ("byte_count", -1),
        ("channel_count", True),
        ("channel_count", 1.0),
        ("channel_count", object()),
        ("channel_count", -1),
        ("sample_count", True),
        ("sample_count", 1.0),
        ("sample_count", object()),
        ("sample_count", -1),
    ],
)
def test_imported_source_rejects_invalid_count_fields(
    field_name: str,
    value: object,
) -> None:
    counts = {
        "byte_count": 10,
        "channel_count": 2,
        "sample_count": 3,
    }
    counts[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        ImportedSourceSummary(
            source_kind="time_series_csv",
            sha256="0" * 64,
            **counts,
        )


def test_export_manifest_rejects_non_string_warnings() -> None:
    with pytest.raises(ValueError, match="warnings"):
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload="version: 1\n",
            command="spo run binding_spec.yaml",
            warnings=(object(),),
        )


def test_project_state_rejects_non_manifest_exports() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(yaml_text="version: 1\nname: demo\n")
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )

    with pytest.raises(ValueError, match="exports"):
        StudioProjectState(
            project_name="demo",
            source=source,
            binding=proposal,
            runtime=snapshot,
            exports=(object(),),
        )


@pytest.mark.parametrize(
    "field_name",
    ["source", "binding", "runtime"],
)
def test_project_state_rejects_invalid_core_components(field_name: str) -> None:
    source: object = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    binding: object = BindingProposal(yaml_text="version: 1\nname: demo\n")
    runtime: object = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )
    values = {
        "source": source,
        "binding": binding,
        "runtime": runtime,
    }
    values[field_name] = object()

    with pytest.raises(ValueError, match=field_name):
        StudioProjectState(
            project_name="demo",
            source=values["source"],
            binding=values["binding"],
            runtime=values["runtime"],
        )


def test_project_state_copies_mutable_exports_sequence() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(yaml_text="version: 1\nname: demo\n")
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )
    manifest = ExportManifest.review_artifact(
        target_kind="binding_spec",
        file_name="binding_spec.yaml",
        payload="version: 1\n",
        command="spo run binding_spec.yaml",
    )
    exports = [manifest]
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        exports=exports,
    )

    before = state.to_audit_record()
    exports.append(
        ExportManifest.review_artifact(
            target_kind="audit",
            file_name="audit.json",
            payload="{}",
            command="spo audit audit.json",
        ),
    )

    assert state.to_audit_record() == before


def test_mapping_mutation_after_construction_does_not_change_audit_record() -> None:
    confidence_factors = {"phase_quality": 0.75}
    provenance = {"import": {"columns": ["t", "a"]}}
    hierarchy_watermarks = {"edge-a": 3}
    metadata = {"review": {"warnings": []}}

    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        confidence_factors=confidence_factors,
        provenance=provenance,
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
        hierarchy_watermarks=hierarchy_watermarks,
    )
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        metadata=metadata,
    )

    before = state.to_audit_record()
    confidence_factors["phase_quality"] = 0.1
    provenance["import"]["columns"].append("mutated")
    hierarchy_watermarks["edge-a"] = 99
    metadata["review"]["warnings"].append("mutated")
    with suppress(AttributeError, TypeError):
        proposal.confidence_factors["phase_quality"] = 0.2
    with suppress(AttributeError, TypeError):
        proposal.provenance["import"]["columns"].append("attribute-mutated")
    with suppress(AttributeError, TypeError):
        snapshot.hierarchy_watermarks["edge-a"] = 42
    with suppress(AttributeError, TypeError):
        state.metadata["review"]["warnings"].append("attribute-mutated")

    assert state.to_audit_record() == before


def test_audit_records_are_fresh_copies() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        confidence_factors={"phase_quality": 0.75},
        provenance={"import": {"columns": ["t", "a"]}},
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
        hierarchy_watermarks={"edge-a": 3},
    )
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        metadata={"review": {"warnings": []}},
    )

    record = state.to_audit_record()
    record["binding"]["confidence_factors"]["phase_quality"] = 0.1
    record["binding"]["provenance"]["import"]["columns"].append("mutated")
    record["runtime"]["hierarchy_watermarks"]["edge-a"] = 99
    record["metadata"]["review"]["warnings"].append("mutated")

    restored = json.loads(json.dumps(state.to_audit_record(), allow_nan=False))

    assert restored["binding"]["confidence_factors"] == {"phase_quality": 0.75}
    assert restored["binding"]["provenance"]["import"]["columns"] == ["t", "a"]
    assert restored["runtime"]["hierarchy_watermarks"] == {"edge-a": 3}
    assert restored["metadata"]["review"]["warnings"] == []


def test_nested_json_safe_provenance_and_metadata_serialise() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a\n0,0.0\n",
        channel_count=1,
        sample_count=1,
    )
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        provenance={
            "import": {
                "source_kind": source.source_kind,
                "columns": ["t", "a"],
                "accepted": True,
                "notes": None,
            },
        },
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
    )
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        metadata={
            "review": {
                "operator": "studio",
                "attempts": 1,
                "confidence": 0.75,
                "warnings": [],
            },
        },
    )

    restored = json.loads(json.dumps(state.to_audit_record()))

    assert restored["binding"]["provenance"]["import"]["columns"] == ["t", "a"]
    assert restored["metadata"]["review"]["warnings"] == []


def test_workflow_module_does_not_import_streamlit() -> None:
    import sys

    assert "streamlit" not in sys.modules
