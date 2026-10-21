from fraudshiftbench.claims import ClaimGate, evaluate_claim_gate
from fraudshiftbench.evidence import EvidenceUnit


def test_claim_gate_requires_seed_count_and_predictions() -> None:
    unit = EvidenceUnit(
        artifact_id="x",
        dataset="ibm_aml_hi_small",
        variant="hi-small",
        protocol="aggregate",
        model="aggregate",
        seed=None,
        result_json_path="lock.json",
        prediction_path="lock.json",
        dry_run=False,
        diagnostic_only=False,
        resource_blocked=False,
        imported=True,
        validation_status="FULL10_PASS",
        sha256="",
        seed_coverage=list(range(1, 11)),
    )
    gate = ClaimGate(
        claim_id="hi_small_full10",
        claim_text="HI-Small full10 is claim eligible.",
        required_datasets=["ibm_aml_hi_small"],
        required_variants=["hi-small"],
        required_seed_count=10,
        required_prediction_exports=True,
    )
    assert evaluate_claim_gate(gate, [unit]).status == "PASS"
