from fraudshiftbench.evidence import EvidenceUnit, validate_evidence_unit


def test_evidence_unit_rejects_dry_run_and_missing_prediction() -> None:
    unit = EvidenceUnit(
        artifact_id="x",
        dataset="ibm_aml_hi_small",
        variant="hi-small",
        protocol="late_window_holdout",
        model="graphsage",
        seed=1,
        result_json_path="result.json",
        prediction_path="",
        dry_run=True,
        diagnostic_only=False,
        resource_blocked=False,
        imported=True,
        validation_status="FULL10_PASS",
        sha256="abc",
    )
    reasons = validate_evidence_unit(unit, require_prediction=True)
    assert "dry_run_evidence_rejected" in reasons
    assert "missing_prediction_export" in reasons
