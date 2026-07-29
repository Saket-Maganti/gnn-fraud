from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from coregraph.evidence import (
    ConstructState,
    DataStatus,
    EvidencePredicate,
    EvidenceUnitV2,
    ImportState,
    IntegrityState,
    ProvenanceLevel,
    ResourceState,
    SupportEngine,
    SupportStatus,
    TypedClaim,
    ValidationState,
)


def unit(contract, **updates) -> EvidenceUnitV2:
    base = EvidenceUnitV2(
        artifact_id="artifact",
        dataset="fixture",
        task_type="node_classification",
        prediction_unit="node",
        dataset_variant="v2",
        deployment_contract=contract,
        model_contract="coregraph",
        seeds=tuple(range(10)),
        result_path="result.json",
        prediction_path="prediction.csv",
        prediction_manifest="manifest.json",
        metric_family=("auprc", "regret"),
        statistical_block_key="fixture-node",
        data_status=DataStatus.COMPLETE,
        provenance_level=ProvenanceLevel.NATIVE,
        validation_state=ValidationState.VALID,
        integrity_state=IntegrityState.VERIFIED,
        construct_state=ConstructState.VALID,
        resource_state=ResourceState.FEASIBLE,
        diagnostic_only=False,
        import_state=ImportState.NATIVE,
        outcomes=(("regret_delta", 0.2),),
    )
    return replace(base, **updates)


def claim(**updates) -> TypedClaim:
    base = TypedClaim(
        claim_id="claim",
        scope="fixture/v2",
        task_type="node_classification",
        prediction_unit="node",
        quantifier="all declared contracts",
        comparison="coregraph > baseline",
        metric="auprc",
        direction="positive",
        uncertainty_requirement="paired",
        pairing_requirement="seed block",
        prediction_requirement=True,
        deployment_interpretation="bounded fixture claim",
        permitted_wording=("in the declared fixture",),
        prohibited_wording=("universally",),
        evidence_predicate=EvidencePredicate(
            datasets=("fixture",),
            variants=("v2",),
            required_metrics=("auprc",),
            minimum_seeds=10,
        ),
    )
    return replace(base, **updates)


def test_22_claim_fixture_is_complete() -> None:
    path = Path(__file__).parent / "fixtures/typed_claims.yaml"
    payload = yaml.safe_load(path.read_text())
    assert len(payload["claim_ids"]) == 22
    assert len(set(payload["claim_ids"])) == 22


def test_support_happy_path(contract_factory) -> None:
    contract = contract_factory()
    report = SupportEngine().evaluate(
        claim(),
        [
            unit(contract),
            unit(contract, artifact_id="baseline", model_contract="baseline"),
        ],
        statistical_requirement_met=True,
    )
    assert report.status is SupportStatus.SUPPORTED
    assert report.predictive_ordering_permitted
    assert report.pairing_diagnostics == ("PAIRING_OK:10_seed_blocks",)


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        ({"integrity_state": IntegrityState.EXCLUDED}, SupportStatus.EXCLUDED_INTEGRITY),
        ({"construct_state": ConstructState.INVALID}, SupportStatus.EXCLUDED_CONSTRUCT_INVALID),
        ({"diagnostic_only": True}, SupportStatus.DIAGNOSTIC_ONLY),
        ({"resource_state": ResourceState.RESOURCE_BLOCKED}, SupportStatus.RESOURCE_BLOCKED),
        ({"validation_state": ValidationState.PENDING}, SupportStatus.BLOCKED_INCOMPLETE_SCOPE),
    ],
)
def test_support_exclusion_statuses(contract_factory, mutation, status) -> None:
    assert SupportEngine().evaluate(claim(), [unit(contract_factory(), **mutation)]).status is status


def test_seed_and_prediction_blocks(contract_factory) -> None:
    short = unit(contract_factory(), seeds=(0, 1, 2, 3, 4))
    assert SupportEngine().evaluate(claim(), [short]).status is SupportStatus.BLOCKED_INCOMPLETE_SEEDS
    missing = unit(contract_factory(), prediction_path="", prediction_manifest="")
    assert SupportEngine().evaluate(claim(), [missing]).status is SupportStatus.BLOCKED_MISSING_PREDICTIONS


def test_contradiction_and_scope_widening(contract_factory) -> None:
    contradictory = claim(
        contradiction_metric="regret_delta",
        contradiction_direction="positive",
    )
    report = SupportEngine().evaluate(
        contradictory,
        [unit(contract_factory())],
        requested_scope="all fraud",
    )
    assert report.status is SupportStatus.REFUTED_IN_SCOPE
    assert report.contradiction_ids == ("artifact",)
    assert report.scope_widening_detected


def test_theory_only_is_not_predictive() -> None:
    theoretical = claim(theoretical=True, prediction_requirement=False)
    report = SupportEngine().evaluate(theoretical, [])
    assert report.status is SupportStatus.SUPPORTED_THEORETICALLY
    assert not report.predictive_ordering_permitted


def test_not_applicable_and_pending_integrity_are_conservative(contract_factory) -> None:
    report = SupportEngine().evaluate(claim(applicable=False), [])
    assert report.status is SupportStatus.NOT_APPLICABLE
    pending = unit(contract_factory(), integrity_state=IntegrityState.PENDING)
    assert SupportEngine().evaluate(claim(), [pending]).status is SupportStatus.BLOCKED_INCOMPLETE_SCOPE
