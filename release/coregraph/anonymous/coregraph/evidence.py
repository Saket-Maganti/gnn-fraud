"""Typed evidence, claims, and conservative support evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

from coregraph.contracts.contract import DeploymentContract
from coregraph.contracts.serialization import to_primitive


class _Value(str, Enum):
    def __str__(self) -> str:
        return self.value


class DataStatus(_Value):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


class ProvenanceLevel(_Value):
    NATIVE = "NATIVE"
    VERIFIED_IMPORT = "VERIFIED_IMPORT"
    LEGACY_IMPORT = "LEGACY_IMPORT"
    UNVERIFIED = "UNVERIFIED"


class ValidationState(_Value):
    VALID = "VALID"
    PENDING = "PENDING"
    INVALID = "INVALID"


class IntegrityState(_Value):
    VERIFIED = "VERIFIED"
    PENDING = "PENDING"
    EXCLUDED = "EXCLUDED"


class ConstructState(_Value):
    VALID = "VALID"
    CURATOR_REVIEW_REQUIRED = "CURATOR_REVIEW_REQUIRED"
    INVALID = "INVALID"


class ResourceState(_Value):
    FEASIBLE = "FEASIBLE"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    UNKNOWN = "UNKNOWN"


class ImportState(_Value):
    NATIVE = "NATIVE"
    IMPORTED = "IMPORTED"
    PENDING = "PENDING"


class SupportStatus(_Value):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_RESOURCE_BOUNDARY = "SUPPORTED_WITH_RESOURCE_BOUNDARY"
    SUPPORTED_THEORETICALLY = "SUPPORTED_THEORETICALLY"
    BLOCKED_INCOMPLETE_SCOPE = "BLOCKED_INCOMPLETE_SCOPE"
    BLOCKED_INCOMPLETE_SEEDS = "BLOCKED_INCOMPLETE_SEEDS"
    BLOCKED_MISSING_PREDICTIONS = "BLOCKED_MISSING_PREDICTIONS"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    EXCLUDED_INTEGRITY = "EXCLUDED_INTEGRITY"
    EXCLUDED_CONSTRUCT_INVALID = "EXCLUDED_CONSTRUCT_INVALID"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    REFUTED_IN_SCOPE = "REFUTED_IN_SCOPE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class EvidenceUnitV2:
    artifact_id: str
    dataset: str
    task_type: str
    prediction_unit: str
    dataset_variant: str
    deployment_contract: DeploymentContract
    model_contract: str
    seeds: Tuple[int, ...]
    result_path: str
    prediction_path: str
    prediction_manifest: str
    metric_family: Tuple[str, ...]
    statistical_block_key: str
    data_status: DataStatus
    provenance_level: ProvenanceLevel
    validation_state: ValidationState
    integrity_state: IntegrityState
    construct_state: ConstructState
    resource_state: ResourceState
    diagnostic_only: bool
    import_state: ImportState
    checksums: Tuple[Tuple[str, str], ...] = ()
    code_hash: str = ""
    config_hash: str = ""
    notes: str = ""
    aliases: Tuple[str, ...] = ()
    independent_replication_id: str = ""
    outcomes: Tuple[Tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @property
    def outcome_map(self) -> dict[str, float]:
        return dict(self.outcomes)


@dataclass(frozen=True)
class EvidencePredicate:
    datasets: Tuple[str, ...] = ()
    task_types: Tuple[str, ...] = ()
    prediction_units: Tuple[str, ...] = ()
    variants: Tuple[str, ...] = ()
    model_contracts: Tuple[str, ...] = ()
    contract_axes: Tuple[str, ...] = ()
    contract_projection: Tuple[Tuple[str, Any], ...] = ()
    required_metrics: Tuple[str, ...] = ()
    minimum_seeds: int = 0
    minimum_independent_replications: int = 0
    allow_diagnostic: bool = False
    allow_resource_blocked: bool = False


@dataclass(frozen=True)
class TypedClaim:
    claim_id: str
    scope: str
    task_type: str
    prediction_unit: str
    quantifier: str
    comparison: str
    metric: str
    direction: str
    uncertainty_requirement: str
    pairing_requirement: str
    prediction_requirement: bool
    deployment_interpretation: str
    permitted_wording: Tuple[str, ...]
    prohibited_wording: Tuple[str, ...]
    evidence_predicate: EvidencePredicate
    contradiction_metric: str = ""
    contradiction_direction: str = ""
    theoretical: bool = False
    applicable: bool = True


@dataclass(frozen=True)
class SupportReport:
    claim_id: str
    status: SupportStatus
    matched_evidence_ids: Tuple[str, ...] = ()
    missing_requirements: Tuple[str, ...] = ()
    excluded_evidence: Tuple[Tuple[str, str], ...] = ()
    pairing_diagnostics: Tuple[str, ...] = ()
    statistical_requirement_result: str = "NOT_EVALUATED"
    contradiction_ids: Tuple[str, ...] = ()
    scope_widening_detected: bool = False
    predictive_ordering_permitted: bool = False
    curator_judgment_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


class SupportEngine:
    """Conservatively evaluate whether evidence satisfies a typed claim.

    The engine checks declared metadata and numeric direction only. It does not
    establish scientific truth or replace construct-validity review.
    """

    def evaluate(
        self,
        claim: TypedClaim,
        evidence: Sequence[EvidenceUnitV2],
        *,
        requested_scope: Optional[str] = None,
        statistical_requirement_met: Optional[bool] = None,
    ) -> SupportReport:
        if not claim.applicable:
            return SupportReport(
                claim_id=claim.claim_id,
                status=SupportStatus.NOT_APPLICABLE,
                statistical_requirement_result="NOT_APPLICABLE",
                predictive_ordering_permitted=False,
                curator_judgment_required=False,
            )
        if claim.theoretical and not evidence:
            return SupportReport(
                claim_id=claim.claim_id,
                status=SupportStatus.SUPPORTED_THEORETICALLY,
                statistical_requirement_result="NOT_APPLICABLE",
                predictive_ordering_permitted=False,
            )

        scope_widening = requested_scope is not None and requested_scope != claim.scope
        candidates: list[EvidenceUnitV2] = []
        excluded: list[tuple[str, str]] = []
        missing: list[str] = []
        predicate = claim.evidence_predicate

        for unit in evidence:
            reason = self._exclusion_reason(claim, predicate, unit)
            if reason:
                excluded.append((unit.artifact_id, reason))
            else:
                candidates.append(unit)

        incomplete_exclusions = {
            "data",
            "data_partial",
            "validation",
            "integrity_pending",
            "construct_review",
            "resource_unknown",
            "provenance",
            "import_pending",
        }
        missing.extend(
            f"excluded_incomplete:{artifact_id}:{reason}"
            for artifact_id, reason in excluded
            if reason in incomplete_exclusions
        )

        if not candidates:
            statuses = {reason for _, reason in excluded}
            if "integrity" in statuses:
                status = SupportStatus.EXCLUDED_INTEGRITY
            elif "construct" in statuses:
                status = SupportStatus.EXCLUDED_CONSTRUCT_INVALID
            elif "diagnostic" in statuses:
                status = SupportStatus.DIAGNOSTIC_ONLY
            elif "resource" in statuses:
                status = SupportStatus.RESOURCE_BLOCKED
            else:
                status = SupportStatus.BLOCKED_INCOMPLETE_SCOPE
            return SupportReport(
                claim_id=claim.claim_id,
                status=status,
                missing_requirements=("no_eligible_evidence",),
                excluded_evidence=tuple(sorted(excluded)),
                scope_widening_detected=scope_widening,
            )

        if scope_widening:
            missing.append("requested_scope_widens_typed_claim")

        datasets = {unit.dataset for unit in candidates}
        for dataset in predicate.datasets:
            if dataset not in datasets:
                missing.append(f"missing_dataset:{dataset}")
        variants = {unit.dataset_variant for unit in candidates}
        for variant in predicate.variants:
            if variant not in variants:
                missing.append(f"missing_variant:{variant}")

        seed_union = sorted({seed for unit in candidates for seed in unit.seeds})
        if len(seed_union) < predicate.minimum_seeds:
            return SupportReport(
                claim_id=claim.claim_id,
                status=SupportStatus.BLOCKED_INCOMPLETE_SEEDS,
                matched_evidence_ids=tuple(sorted(unit.artifact_id for unit in candidates)),
                missing_requirements=(
                    f"seed_count:{len(seed_union)}<{predicate.minimum_seeds}",
                ),
                excluded_evidence=tuple(sorted(excluded)),
                scope_widening_detected=scope_widening,
            )

        if claim.prediction_requirement:
            no_predictions = [
                unit.artifact_id
                for unit in candidates
                if not unit.prediction_path or not unit.prediction_manifest
            ]
            if no_predictions:
                return SupportReport(
                    claim_id=claim.claim_id,
                    status=SupportStatus.BLOCKED_MISSING_PREDICTIONS,
                    matched_evidence_ids=tuple(sorted(unit.artifact_id for unit in candidates)),
                    missing_requirements=tuple(
                        f"missing_prediction:{artifact}" for artifact in sorted(no_predictions)
                    ),
                    excluded_evidence=tuple(sorted(excluded)),
                    scope_widening_detected=scope_widening,
                )

        replications = {
            unit.independent_replication_id
            for unit in candidates
            if unit.independent_replication_id
        }
        if len(replications) < predicate.minimum_independent_replications:
            missing.append(
                "independent_replications:"
                f"{len(replications)}<{predicate.minimum_independent_replications}"
            )

        pairing = self._pairing_diagnostics(claim, candidates)
        if any(item.startswith("PAIRING_FAILED") for item in pairing):
            missing.extend(pairing)

        contradictions = self._contradictions(claim, candidates)
        if contradictions:
            status = SupportStatus.REFUTED_IN_SCOPE
        elif missing:
            status = SupportStatus.BLOCKED_INCOMPLETE_SCOPE
        elif statistical_requirement_met is False:
            status = SupportStatus.BLOCKED_INCOMPLETE_SCOPE
            missing.append("statistical_requirement_failed")
        elif any(unit.resource_state is ResourceState.RESOURCE_BLOCKED for unit in evidence):
            status = SupportStatus.SUPPORTED_WITH_RESOURCE_BOUNDARY
        else:
            status = SupportStatus.SUPPORTED

        stat_result = (
            "MET"
            if statistical_requirement_met is True
            else "FAILED"
            if statistical_requirement_met is False
            else "NOT_DECLARED"
        )
        return SupportReport(
            claim_id=claim.claim_id,
            status=status,
            matched_evidence_ids=tuple(sorted(unit.artifact_id for unit in candidates)),
            missing_requirements=tuple(sorted(set(missing))),
            excluded_evidence=tuple(sorted(excluded)),
            pairing_diagnostics=tuple(pairing),
            statistical_requirement_result=stat_result,
            contradiction_ids=tuple(sorted(contradictions)),
            scope_widening_detected=scope_widening,
            predictive_ordering_permitted=status
            in {SupportStatus.SUPPORTED, SupportStatus.SUPPORTED_WITH_RESOURCE_BOUNDARY},
        )

    @staticmethod
    def _exclusion_reason(
        claim: TypedClaim,
        predicate: EvidencePredicate,
        unit: EvidenceUnitV2,
    ) -> str:
        if unit.data_status is DataStatus.RESOURCE_BLOCKED:
            return "resource"
        if unit.data_status is DataStatus.MISSING:
            return "data"
        if unit.data_status is DataStatus.PARTIAL:
            return "data_partial"
        if unit.provenance_level is ProvenanceLevel.UNVERIFIED:
            return "provenance"
        if unit.import_state is ImportState.PENDING:
            return "import_pending"
        if unit.integrity_state is IntegrityState.EXCLUDED:
            return "integrity"
        if unit.integrity_state is IntegrityState.PENDING:
            return "integrity_pending"
        if unit.construct_state is ConstructState.INVALID:
            return "construct"
        if unit.construct_state is ConstructState.CURATOR_REVIEW_REQUIRED:
            return "construct_review"
        if unit.validation_state is not ValidationState.VALID:
            return "validation"
        if unit.diagnostic_only and not predicate.allow_diagnostic:
            return "diagnostic"
        if (
            unit.resource_state is ResourceState.RESOURCE_BLOCKED
            and not predicate.allow_resource_blocked
        ):
            return "resource"
        if unit.resource_state is ResourceState.UNKNOWN:
            return "resource_unknown"
        if unit.task_type != claim.task_type or unit.prediction_unit != claim.prediction_unit:
            return "task_unit"
        checks = (
            (predicate.datasets, unit.dataset),
            (predicate.task_types, unit.task_type),
            (predicate.prediction_units, unit.prediction_unit),
            (predicate.variants, unit.dataset_variant),
            (predicate.model_contracts, unit.model_contract),
        )
        if any(allowed and value not in allowed for allowed, value in checks):
            return "scope"
        if predicate.required_metrics and not set(predicate.required_metrics).issubset(unit.metric_family):
            return "metric"
        if predicate.contract_projection:
            actual = unit.deployment_contract.claim_projection(predicate.contract_axes)
            expected = dict(predicate.contract_projection)
            if actual != expected:
                return "contract"
        return ""

    @staticmethod
    def _pairing_diagnostics(
        claim: TypedClaim,
        evidence: Sequence[EvidenceUnitV2],
    ) -> list[str]:
        if not claim.pairing_requirement or claim.pairing_requirement.lower() in {"none", "not required"}:
            return ["PAIRING_NOT_REQUIRED"]
        by_block_and_model: dict[tuple[str, str], set[int]] = {}
        for unit in evidence:
            by_block_and_model.setdefault(
                (unit.statistical_block_key, unit.model_contract), set()
            ).update(unit.seeds)
        if not by_block_and_model or any(
            not seeds for seeds in by_block_and_model.values()
        ):
            return ["PAIRING_FAILED:missing_seed_blocks"]
        models = {model for _, model in by_block_and_model}
        if len(models) < 2:
            return ["PAIRING_FAILED:missing_comparator_model"]
        seed_sets = list(by_block_and_model.values())
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            return ["PAIRING_FAILED:unaligned_seed_blocks"]
        return [f"PAIRING_OK:{len(seed_sets[0])}_seed_blocks"]

    @staticmethod
    def _contradictions(
        claim: TypedClaim,
        evidence: Sequence[EvidenceUnitV2],
    ) -> list[str]:
        metric = claim.contradiction_metric or claim.metric
        direction = claim.contradiction_direction
        if not direction:
            return []
        out: list[str] = []
        for unit in evidence:
            value = unit.outcome_map.get(metric)
            if value is None:
                continue
            if direction == "positive" and value > 0:
                out.append(unit.artifact_id)
            elif direction == "negative" and value < 0:
                out.append(unit.artifact_id)
            elif direction == "nonpositive" and value <= 0:
                out.append(unit.artifact_id)
            elif direction == "nonnegative" and value >= 0:
                out.append(unit.artifact_id)
        return out


def evidence_report(
    claims: Iterable[TypedClaim],
    evidence: Sequence[EvidenceUnitV2],
) -> list[dict[str, Any]]:
    engine = SupportEngine()
    return [engine.evaluate(claim, evidence).to_dict() for claim in claims]
