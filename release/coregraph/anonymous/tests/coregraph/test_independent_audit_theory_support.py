from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from coregraph.evidence import (
    ResourceState,
    ScopeRelation,
    SupportEngine,
    SupportStatus,
)
from tests.coregraph.test_evidence_support import claim, unit


def test_theoretical_boolean_alone_cannot_support_a_claim(tmp_path: Path) -> None:
    theoretical = claim(theoretical=True, prediction_requirement=False)
    blocked = SupportEngine().evaluate(theoretical, [])
    assert blocked.status is SupportStatus.BLOCKED_INCOMPLETE_SCOPE
    assert "verified_theory_artifacts_missing" in blocked.missing_requirements

    proof = tmp_path / "proof.tex"
    status = tmp_path / "status.yaml"
    proof.write_text("proved statement\n", encoding="utf-8")
    status.write_text(
        "results:\n"
        "  fixture_theorem:\n"
        "    status: PROVED\n"
        "    statement: proof.tex\n",
        encoding="utf-8",
    )
    verified = replace(
        theoretical,
        proof_artifact=str(proof),
        proof_artifact_hash=hashlib.sha256(proof.read_bytes()).hexdigest(),
        proof_status_artifact=str(status),
        proof_status_hash=hashlib.sha256(status.read_bytes()).hexdigest(),
        proof_status_key="fixture_theorem",
    )
    report = SupportEngine().evaluate(verified, [])
    assert report.status is SupportStatus.SUPPORTED_THEORETICALLY

    unproved_status = tmp_path / "unproved.yaml"
    unproved_status.write_text(
        "results:\n"
        "  fixture_theorem:\n"
        "    status: CONJECTURE\n"
        "    statement: proof.tex\n",
        encoding="utf-8",
    )
    unproved = replace(
        verified,
        proof_status_artifact=str(unproved_status),
        proof_status_hash=hashlib.sha256(
            unproved_status.read_bytes()
        ).hexdigest(),
    )
    assert (
        SupportEngine().evaluate(unproved, []).status
        is SupportStatus.BLOCKED_INCOMPLETE_SCOPE
    )


def test_unrelated_resource_block_does_not_change_claim_status(
    contract_factory,
) -> None:
    evidence = [
        unit(contract_factory()),
        unit(
            contract_factory(),
            artifact_id="baseline",
            model_contract="baseline",
        ),
        unit(
            contract_factory(),
            artifact_id="unrelated",
            dataset="other",
            resource_state=ResourceState.RESOURCE_BLOCKED,
        ),
    ]
    report = SupportEngine().evaluate(
        claim(),
        evidence,
        statistical_requirement_met=True,
    )
    assert report.status is SupportStatus.SUPPORTED


def test_only_claim_relevant_blocked_cell_creates_resource_boundary(
    contract_factory,
) -> None:
    evidence = [
        unit(contract_factory()),
        unit(
            contract_factory(),
            artifact_id="baseline",
            model_contract="baseline",
        ),
        unit(
            contract_factory(),
            artifact_id="relevant_block",
            model_contract="baseline",
            resource_state=ResourceState.RESOURCE_BLOCKED,
        ),
    ]
    report = SupportEngine().evaluate(
        claim(),
        evidence,
        statistical_requirement_met=True,
    )
    assert report.status is SupportStatus.SUPPORTED_WITH_RESOURCE_BOUNDARY


def test_scope_subset_widening_and_incompatibility_are_distinct(
    contract_factory,
) -> None:
    evidence = [
        unit(contract_factory()),
        unit(
            contract_factory(),
            artifact_id="baseline",
            model_contract="baseline",
        ),
    ]
    engine = SupportEngine()
    assert (
        engine.scope_relation("fixture/v2", "fixture/v2/elliptic")
        is ScopeRelation.NARROWER
    )
    assert (
        engine.evaluate(
            claim(),
            evidence,
            requested_scope="fixture/v2/elliptic",
            statistical_requirement_met=True,
        ).status
        is SupportStatus.SUPPORTED
    )
    widened = engine.evaluate(
        claim(),
        evidence,
        requested_scope="fixture",
        statistical_requirement_met=True,
    )
    assert widened.scope_widening_detected
    assert "requested_scope_widens_typed_claim" in widened.missing_requirements
    incompatible = engine.evaluate(
        claim(),
        evidence,
        requested_scope="other/v2",
        statistical_requirement_met=True,
    )
    assert not incompatible.scope_widening_detected
    assert "requested_scope_incompatible" in incompatible.missing_requirements
