from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from coregraph.evidence.archive_store import ArchiveIntegrityError, ArchiveStore
from coregraph.experiments.v5_pilot_executor import (
    MethodInference,
    TargetLabelVault,
    _availability,
    _boolean,
    _evaluate,
    _method_inference,
    assemble_source_environments,
    assemble_target_unlabeled,
    compute_gate,
    execute_coordinate,
)
from coregraph.experiments.v5_pilot_outputs import (
    OUTPUT_SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_npz,
    atomic_write_text,
    coordinate_identity_hash,
    load_checkpoint,
    mark_complete,
    reusable_complete,
    write_checkpoint,
    write_failure,
)
from coregraph.experiments.v5_pilot_types import (
    EXPERT_ORDER,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METHODS,
    PilotCheckpoint,
    PilotCoordinate,
    PilotStage,
    TargetEvaluationBundle,
    TargetUnlabeledBundle,
)
from coregraph.experiments.v5_scenario_loader import (
    _integer,
    build_pilot_coordinates,
    load_v5_config,
    load_v5_surface,
    validate_archive_surface,
)
from coregraph.experiments.v5_synthetic import build_synthetic_fixture
from coregraph.utils.io import atomic_write_json, sha256_path


ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = ROOT / "configs/coregraph/pilot/saved_output_v5.yaml"
CODE_SHA = "c879c979cb5964b55d8da56919ae90d46ac8e9e1"
EFFECTIVE_HASH = "e" * 64


def _coordinates(scenarios, config, effective_hash: str = EFFECTIVE_HASH):
    return build_pilot_coordinates(
        scenarios,
        config,
        effective_execution_config_sha256=effective_hash,
    )


@pytest.fixture(scope="module")
def v5_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5_fixture")
    base_config = load_v5_config(REAL_CONFIG)
    config_path, evidence = build_synthetic_fixture(root, base_config)
    config = load_v5_config(config_path)
    artifacts, scenarios = load_v5_surface(
        config, code_sha=CODE_SHA, evidence_cache=evidence
    )
    store = ArchiveStore(evidence, dict(config.payload["archive_hashes"]))
    return root, evidence, config, artifacts, scenarios, store


def test_authoritative_config_is_strict_and_preregistered(tmp_path: Path) -> None:
    config = load_v5_config(REAL_CONFIG)
    assert config.methods == PRIMARY_METHODS
    assert config.experts == EXPERT_ORDER
    payload = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    payload["unknown_scientific_knob"] = True
    payload["preregistration_path"] = str(config.resolve("preregistration_path"))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="field mismatch"):
        load_v5_config(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("nonmapping", "decode to a mapping"),
        ("schema", "schema version is invalid"),
        ("metric", "metric schema version is invalid"),
        ("methods", "primary method set/order"),
        ("experts", "expert set/order"),
        ("preregistration", "preregistration hash mismatch"),
    ),
)
def test_authoritative_config_rejects_superseded_or_mutated_identity(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config = load_v5_config(REAL_CONFIG)
    path = tmp_path / f"{mutation}.yaml"
    if mutation == "nonmapping":
        path.write_text("- invalid\n", encoding="utf-8")
    else:
        payload = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        payload["preregistration_path"] = str(config.resolve("preregistration_path"))
        if mutation == "schema":
            payload["schema_version"] = "old"
        elif mutation == "metric":
            payload["metric_schema_version"] = "old"
        elif mutation == "methods":
            payload["primary_methods"] = list(reversed(payload["primary_methods"]))
        elif mutation == "experts":
            payload["required_experts"] = list(reversed(payload["required_experts"]))
        elif mutation == "preregistration":
            payload["preregistration_sha256"] = "0" * 64
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_v5_config(path)


def test_registry_integer_parser_is_strict() -> None:
    assert _integer({"rows": "7"}, "rows") == 7
    with pytest.raises(ValueError, match="field 'rows' is empty"):
        _integer({}, "rows")


def test_v5_surface_is_exact_and_role_neutral(v5_fixture) -> None:
    _, _, config, artifacts, scenarios, _ = v5_fixture
    assert len(artifacts) == 180
    assert len(scenarios) == 60
    assert sum(len(item.source_bindings) for item in scenarios) == 360
    assert sum(len(item.target_bindings) for item in scenarios) == 180
    coordinates = _coordinates(scenarios, config)
    assert len(coordinates) == 240
    assert len({item.key for item in coordinates}) == 240
    reused = artifacts[0].base_coordinate_id
    roles = {
        binding.role
        for scenario in scenarios
        for binding in (*scenario.source_bindings, *scenario.target_bindings)
        if binding.base_coordinate_id == reused
    }
    assert roles == {"source", "target"}
    assert all(
        not ({item.base_coordinate_id for item in scenario.source_bindings}
             & {item.base_coordinate_id for item in scenario.target_bindings})
        for scenario in scenarios
    )


def test_synthetic_archive_surface_verifies_all_members(v5_fixture) -> None:
    _, evidence, _, artifacts, _, _ = v5_fixture
    result = validate_archive_surface(
        artifacts, evidence_cache=evidence, verify_members=True
    )
    assert result["archive_count"] == 6
    assert result["member_checksum_verified"] == 180
    assert result["permanent_extractions"] == 0


def test_source_assembly_is_deterministic_and_target_has_no_labels(v5_fixture) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    first = assemble_source_environments(store, scenario, config)
    second = assemble_source_environments(store, scenario, config)
    assert len(first) == 2
    assert [item.row_keys for item in first] == [item.row_keys for item in second]
    assert all(set(item.splits) == {"train", "validation"} for item in first)
    target = assemble_target_unlabeled(store, scenario, config)
    assert len(target.row_keys) == 8
    assert not hasattr(target, "labels")
    assert target.to_serializable()["target_labels_present"] is False
    with pytest.raises(TypeError):
        TargetUnlabeledBundle(
            dataset="d",
            protocol="p",
            provider_seed=1,
            row_keys=(),
            scores=np.empty((0, 3)),
            availability=np.empty((0, 3), dtype=bool),
            labels=np.empty(0),  # type: ignore[call-arg]
        )


def test_label_vault_fails_before_freeze_and_cannot_serialize(
    v5_fixture, tmp_path: Path
) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    target = assemble_target_unlabeled(store, scenario, config)
    vault = TargetLabelVault(store, scenario)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(vault)
    score_path = tmp_path / "scores.bin"
    score_path.write_bytes(b"scores")
    freeze = tmp_path / "POLICY_FREEZE_MANIFEST.json"
    atomic_write_json(freeze, {"schema": "wrong", "target_labels_loaded": False})
    with pytest.raises(RuntimeError, match="requires a V5 policy freeze"):
        vault.open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )


def test_label_vault_rejects_unblinded_changed_and_misaligned_inputs(
    v5_fixture, tmp_path: Path
) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    target = assemble_target_unlabeled(store, scenario, config)
    score_path = tmp_path / "scores.bin"
    score_path.write_bytes(b"scores")
    freeze = tmp_path / "POLICY_FREEZE_MANIFEST.json"
    base = {
        "schema": "coregraph_v5_policy_freeze_manifest_v2",
        "target_labels_loaded": False,
        "target_score_sha256": sha256_path(score_path),
        "effective_execution_config_sha256": EFFECTIVE_HASH,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
    }
    atomic_write_json(freeze, {**base, "target_labels_loaded": True})
    with pytest.raises(RuntimeError, match="does not attest"):
        TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )
    atomic_write_json(freeze, {**base, "target_score_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="changed after policy freeze"):
        TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )
    atomic_write_json(freeze, base)
    with pytest.raises(RuntimeError, match="effective execution identity mismatch"):
        TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256="f" * 64,
        )
    atomic_write_json(freeze, {**base, "metric_schema_version": "old"})
    with pytest.raises(RuntimeError, match="metric schema is superseded"):
        TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )
    atomic_write_json(freeze, base)
    with pytest.raises(RuntimeError, match="do not align"):
        TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=tuple(reversed(target.row_keys)),
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )
    vault = TargetLabelVault(store, scenario)
    evaluation = vault.open_after_freeze(
        freeze_manifest_path=freeze,
        target_scores_path=score_path,
        expected_row_keys=target.row_keys,
        expected_effective_execution_config_sha256=EFFECTIVE_HASH,
    )
    assert evaluation.row_keys == target.row_keys
    with pytest.raises(RuntimeError, match="single-use"):
        vault.open_after_freeze(
            freeze_manifest_path=freeze,
            target_scores_path=score_path,
            expected_row_keys=target.row_keys,
            expected_effective_execution_config_sha256=EFFECTIVE_HASH,
        )


@pytest.mark.parametrize("method", PRIMARY_METHODS)
def test_all_four_methods_fit_and_score_without_target_labels(
    v5_fixture, method: str
) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    source = assemble_source_environments(store, scenario, config)
    target = assemble_target_unlabeled(store, scenario, config)
    inference = _method_inference(method, source, target, scenario, config)
    assert inference.scores.shape == (len(target.row_keys),)
    assert np.isfinite(inference.scores).all()
    assert inference.policy.fit_report["target_labels_available_to_fitter"] is False
    assert inference.policy.policy_state_hash


@pytest.mark.parametrize("method", PRIMARY_METHODS)
def test_target_inference_honors_chunk_boundaries(
    v5_fixture, method: str
) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    source = assemble_source_environments(store, scenario, config)
    target = assemble_target_unlabeled(store, scenario, config)
    payload = dict(config.payload)
    payload["streaming"] = {**dict(config.payload["streaming"]), "chunk_rows": 3}
    chunked_config = replace(config, payload=payload)
    whole = _method_inference(method, source, target, scenario, config)
    chunked = _method_inference(method, source, target, scenario, chunked_config)
    np.testing.assert_allclose(chunked.scores, whole.scores, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        chunked.routing_weights, whole.routing_weights, rtol=0, atol=1e-7
    )
    np.testing.assert_array_equal(chunked.selected_experts, whole.selected_experts)


def test_executor_fail_closed_branches(v5_fixture, tmp_path: Path) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    source = assemble_source_environments(store, scenario, config)
    target = assemble_target_unlabeled(store, scenario, config)
    with pytest.raises(ArchiveIntegrityError, match="invalid label_known"):
        _boolean("not-a-boolean")
    missing_profile = replace(
        scenario,
        definition=replace(scenario.definition, resource_profile="missing"),
    )
    with pytest.raises(ValueError, match="unknown resource profile"):
        _availability(missing_profile, config, 1)
    payload = dict(config.payload)
    payload["streaming"] = {
        **dict(config.payload["streaming"]),
        "source_rows_per_split_per_environment": 0,
    }
    with pytest.raises(ValueError, match="source row cap"):
        assemble_source_environments(store, scenario, replace(config, payload=payload))
    payload["streaming"] = {
        **dict(config.payload["streaming"]),
        "chunk_rows": 0,
    }
    with pytest.raises(ValueError, match="chunk size"):
        _method_inference(
            "uniform_average", source, target, scenario, replace(config, payload=payload)
        )
    with pytest.raises(ValueError, match="unknown primary method"):
        _method_inference("not-a-method", source, target, scenario, config)
    target_with_labels = type(
        "LeakyTarget",
        (),
        {"labels": np.asarray([0]), "to_serializable": lambda self: {}},
    )()
    with pytest.raises(RuntimeError, match="target label firewall failed"):
        _method_inference(
            "uniform_average", source, target_with_labels, scenario, config  # type: ignore[arg-type]
        )

    inference = _method_inference("uniform_average", source, target, scenario, config)
    misaligned = TargetEvaluationBundle(
        dataset=target.dataset,
        protocol=target.protocol,
        provider_seed=target.provider_seed,
        row_keys=tuple(reversed(target.row_keys)),
        labels=np.zeros(len(target.row_keys), dtype=np.int16),
    )
    with pytest.raises(RuntimeError, match="rows do not match"):
        _evaluate(inference, target, misaligned, scenario, config)
    all_abstain = replace(inference, abstain=np.ones(len(target.row_keys), dtype=bool))
    evaluation = TargetEvaluationBundle(
        dataset=target.dataset,
        protocol=target.protocol,
        provider_seed=target.provider_seed,
        row_keys=target.row_keys,
        labels=np.zeros(len(target.row_keys), dtype=np.int16),
    )
    metrics = _evaluate(all_abstain, target, evaluation, scenario, config)
    assert "global_target_auprc" in metrics
    assert metrics["coverage"] == 0

    coordinate = next(
        item
        for item in _coordinates(scenarios, config)
        if item.scenario_id == scenario.definition.scenario_id
        and item.method == "uniform_average"
    )
    broken_payload = dict(config.payload)
    broken_payload["expert_relative_costs"] = {"feature_mlp": 1.0}
    with pytest.raises(KeyError):
        execute_coordinate(
            coordinate=coordinate,
            scenario=scenario,
            source=source,
            target=target,
            store=store,
            config=replace(config, payload=broken_payload),
            output_root=tmp_path,
            code_sha=CODE_SHA,
            dependency_lock_sha256=sha256_path(
                ROOT / "requirements-coregraph-lock.txt"
            ),
            effective_execution_config_sha256=EFFECTIVE_HASH,
            resume=False,
        )
    with pytest.raises(ValueError, match="effective execution identities disagree"):
        execute_coordinate(
            coordinate=coordinate,
            scenario=scenario,
            source=source,
            target=target,
            store=store,
            config=config,
            output_root=tmp_path,
            code_sha=CODE_SHA,
            dependency_lock_sha256=sha256_path(
                ROOT / "requirements-coregraph-lock.txt"
            ),
            effective_execution_config_sha256="f" * 64,
            resume=False,
        )
    failure = tmp_path / "scenarios" / coordinate.scenario_id / "failures" / "uniform_average.json"
    assert failure.is_file()
    assert json.loads(failure.read_text(encoding="utf-8"))["stage"] == "SOURCE_ASSEMBLED"


def test_output_resume_primitives_cover_corruption_and_all_identity_checks(
    v5_fixture, tmp_path: Path
) -> None:
    _, _, config, _, scenarios, _ = v5_fixture
    coordinate = _coordinates(scenarios, config)[0]
    method_root = tmp_path / "scenarios" / coordinate.scenario_id / "methods" / coordinate.method
    method_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="empty required CSV"):
        atomic_write_csv(tmp_path / "empty.csv", [])
    atomic_write_csv(tmp_path / "one.csv", [{"a": 1, "b": 2}])
    atomic_write_text(tmp_path / "value.txt", "value\n")
    atomic_write_npz(tmp_path / "value.npz", values=np.asarray([1], dtype=np.int8))
    assert load_checkpoint(method_root) is None
    (method_root / "checkpoint.json").write_text("{bad", encoding="utf-8")
    assert load_checkpoint(method_root).stage is PilotStage.FAILED  # type: ignore[union-attr]

    output = method_root / "result.json"
    atomic_write_json(output, {"ok": True})
    checkpoint = PilotCheckpoint(
        coordinate_key="wrong",
        identity_hash="wrong",
        stage=PilotStage.EVALUATED,
        output_schema_version="wrong",
        metric_schema_version="wrong",
        effective_execution_config_sha256="wrong",
        checksums={"result.json": "0" * 64, "missing.json": "0" * 64},
    )
    write_checkpoint(method_root, checkpoint)
    reusable, reasons = reusable_complete(
        method_root, coordinate=coordinate, identity_hash="expected"
    )
    assert not reusable
    assert {
        "coordinate_key_mismatch",
        "identity_hash_mismatch",
        "output_schema_mismatch",
        "metric_schema_mismatch",
        "effective_execution_config_mismatch",
        "stage_evaluated",
        "complete_marker_missing",
        "output_checksum_mismatch:result.json",
        "output_missing:missing.json",
    }.issubset(reasons)

    identity = "a" * 64
    mark_complete(
        method_root,
        coordinate=coordinate,
        identity_hash=identity,
        outputs=("result.json",),
        retry_count=2,
    )
    assert reusable_complete(
        method_root, coordinate=coordinate, identity_hash=identity
    )[0]
    failure_path = write_failure(
        method_root,
        coordinate=coordinate,
        identity_hash=identity,
        stage=PilotStage.POLICY_FITTED,
        exception=RuntimeError("synthetic failure"),
        traceback_text="trace",
        retry_count=3,
    )
    assert failure_path.is_file()
    assert load_checkpoint(method_root).stage is PilotStage.FAILED  # type: ignore[union-attr]


def test_policy_freeze_offline_evaluation_and_resume(v5_fixture, tmp_path: Path) -> None:
    _, _, config, _, scenarios, store = v5_fixture
    scenario = scenarios[0]
    coordinate = next(
        item
        for item in _coordinates(scenarios, config)
        if item.scenario_id == scenario.definition.scenario_id
        and item.method == "uniform_average"
    )
    source = assemble_source_environments(store, scenario, config)
    target = assemble_target_unlabeled(store, scenario, config)
    dependency_hash = sha256_path(ROOT / "requirements-coregraph-lock.txt")
    result = execute_coordinate(
        coordinate=coordinate,
        scenario=scenario,
        source=source,
        target=target,
        store=store,
        config=config,
        output_root=tmp_path,
        code_sha=CODE_SHA,
        dependency_lock_sha256=dependency_hash,
        effective_execution_config_sha256=EFFECTIVE_HASH,
        resume=False,
    )
    method_root = (
        tmp_path / "scenarios" / coordinate.scenario_id / "methods" / coordinate.method
    )
    freeze = json.loads(
        (method_root / "POLICY_FREEZE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert freeze["target_labels_loaded"] is False
    assert result["target_labels_loaded_only_by_offline_evaluator"] is True
    identity = coordinate_identity_hash(
        coordinate,
        code_sha=CODE_SHA,
        config_sha256=config.config_sha256,
        preregistration_sha256=config.preregistration_sha256,
        dependency_lock_sha256=dependency_hash,
        effective_execution_config_sha256=EFFECTIVE_HASH,
    )
    assert reusable_complete(method_root, coordinate=coordinate, identity_hash=identity)[0]
    stale = replace(coordinate, scenario_fingerprint="f" * 64)
    assert not reusable_complete(method_root, coordinate=stale, identity_hash=identity)[0]
    resumed = execute_coordinate(
        coordinate=coordinate,
        scenario=scenario,
        source=source,
        target=target,
        store=store,
        config=config,
        output_root=tmp_path,
        code_sha=CODE_SHA,
        dependency_lock_sha256=dependency_hash,
        effective_execution_config_sha256=EFFECTIVE_HASH,
        resume=True,
    )
    assert resumed["coordinate_key"] == coordinate.key


def _gate_rows(
    coordinates: tuple[PilotCoordinate, ...],
    config,
    *,
    core_regret: float,
    baseline_regret: float,
):
    rows = []
    for coordinate in coordinates:
        core = coordinate.method == "coregraph"
        rows.append(
            {
                "schema": "coregraph_v5_pilot_method_result_v2",
                "coordinate_key": coordinate.key,
                "effective_execution_config_sha256": EFFECTIVE_HASH,
                "preregistration_sha256": config.preregistration_sha256,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "coordinate": {
                    "dataset": coordinate.dataset,
                    "target_protocol": coordinate.target_protocol,
                    "provider_seed": coordinate.provider_seed,
                    "method": coordinate.method,
                },
                "metrics": {
                    "metric_schema_version": METRIC_SCHEMA_VERSION,
                    "contract_regret_vs_feasible_row_oracle": (
                        core_regret if core else baseline_regret
                    ),
                    "global_target_auprc": 0.9 if core else 0.8,
                },
            }
        )
    return rows


def test_gate_emits_go_no_go_and_inconclusive(v5_fixture) -> None:
    _, _, config, _, scenarios, _ = v5_fixture
    scenario = scenarios[0]
    coordinates = tuple(
        PilotCoordinate(
            dataset=scenario.definition.dataset,
            target_protocol=scenario.definition.target_protocol,
            provider_seed=scenario.definition.provider_seed,
            method=method,
            pilot_specification_version=config.specification_version,
            scenario_id=scenario.definition.scenario_id,
            scenario_fingerprint=scenario.scenario_fingerprint,
            effective_execution_config_sha256=EFFECTIVE_HASH,
        )
        for method in PRIMARY_METHODS
    )
    go = compute_gate(
        _gate_rows(coordinates, config, core_regret=0.0, baseline_regret=0.02),
        coordinates=coordinates,
        config=config,
    )
    assert go["outcome"] == "GO"
    no_go = compute_gate(
        _gate_rows(coordinates, config, core_regret=0.03, baseline_regret=0.02),
        coordinates=coordinates,
        config=config,
    )
    assert no_go["outcome"] == "NO_GO"
    incomplete = compute_gate(
        _gate_rows(coordinates, config, core_regret=0.0, baseline_regret=0.02)[:-1],
        coordinates=coordinates,
        config=config,
    )
    assert incomplete["outcome"] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("planned_effective", "planned_effective_execution_config_mixed"),
        ("schema", "old_or_unknown_method_result_schema"),
        ("metric", "old_or_mixed_metric_schema"),
        ("effective", "mixed_effective_execution_config"),
        ("preregistration", "mixed_preregistration"),
        ("duplicate", "duplicated_method_within_paired_cell"),
        ("superseded", "superseded_metric_field_present"),
        ("missing_metric", "corrected_primary_metrics_missing_or_invalid"),
        ("negative", "negative_regret_below_tolerance"),
        ("method_family", "paired_cell_method_family_malformed"),
    ),
)
def test_gate_fails_closed_on_mixed_or_superseded_results(
    v5_fixture, mutation: str, reason: str
) -> None:
    _, _, config, _, scenarios, _ = v5_fixture
    scenario = scenarios[0]
    coordinates = tuple(
        PilotCoordinate(
            dataset=scenario.definition.dataset,
            target_protocol=scenario.definition.target_protocol,
            provider_seed=scenario.definition.provider_seed,
            method=method,
            pilot_specification_version=config.specification_version,
            scenario_id=scenario.definition.scenario_id,
            scenario_fingerprint=scenario.scenario_fingerprint,
            effective_execution_config_sha256=EFFECTIVE_HASH,
        )
        for method in PRIMARY_METHODS
    )
    rows = _gate_rows(coordinates, config, core_regret=0.0, baseline_regret=0.02)
    if mutation == "planned_effective":
        coordinates = (*coordinates[:-1], replace(coordinates[-1], effective_execution_config_sha256="f" * 64))
    elif mutation == "schema":
        rows[0]["schema"] = "old"
    elif mutation == "metric":
        rows[0]["metric_schema_version"] = "old"
    elif mutation == "effective":
        rows[0]["effective_execution_config_sha256"] = "f" * 64
    elif mutation == "preregistration":
        rows[0]["preregistration_sha256"] = "f" * 64
    elif mutation == "duplicate":
        rows[1]["coordinate"]["method"] = rows[0]["coordinate"]["method"]
    elif mutation == "superseded":
        rows[0]["metrics"]["auprc"] = 0.9
    elif mutation == "missing_metric":
        rows[0]["metrics"].pop("global_target_auprc")
    elif mutation == "negative":
        rows[0]["metrics"]["contract_regret_vs_feasible_row_oracle"] = -1.0
    elif mutation == "method_family":
        rows[0]["coordinate"]["method"] = "unknown"
    gate = compute_gate(rows, coordinates=coordinates, config=config)
    assert gate["outcome"] == "INCONCLUSIVE"
    assert reason in gate["reasons"]


def test_archive_change_and_member_change_fail_closed(v5_fixture) -> None:
    _, evidence, config, artifacts, _, _ = v5_fixture
    artifact = artifacts[0]
    store = ArchiveStore(evidence, dict(config.payload["archive_hashes"]))
    with pytest.raises(ArchiveIntegrityError, match="member checksum mismatch"):
        store.verify_member(
            artifact.archive_name,
            artifact.member_name,
            expected_sha256="0" * 64,
        )
    changed_hashes = dict(config.payload["archive_hashes"])
    changed_hashes[artifact.archive_name] = "0" * 64
    with pytest.raises(ArchiveIntegrityError, match="archive checksum mismatch"):
        ArchiveStore(evidence, changed_hashes).verify_archive(artifact.archive_name)


def test_output_schema_version_is_frozen() -> None:
    assert OUTPUT_SCHEMA_VERSION == "coregraph_v5_pilot_outputs_v2"
