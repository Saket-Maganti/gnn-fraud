from fraudshiftbench.evidence import EvidenceUnit, validate_evidence_unit


def make_unit(**overrides):
    payload = dict(
        artifact_id="x",
        dataset="ibm_aml",
        variant="hi-small",
        protocol="late_window_holdout",
        model="m",
        seed=1,
        result_json_path="result.json",
        prediction_path="pred.csv",
        dry_run=False,
        diagnostic_only=False,
        resource_blocked=False,
        imported=True,
        validation_status="FULL10_PASS",
        sha256="abc",
        seed_coverage=[1],
    )
    payload.update(overrides)
    return EvidenceUnit(**payload)


def test_rejects_dry_run_artifact():
    assert "dry_run_evidence_rejected" in validate_evidence_unit(make_unit(dry_run=True))


def test_rejects_missing_prediction_for_review_budget():
    assert "missing_prediction_export" in validate_evidence_unit(make_unit(prediction_path=""), require_prediction=True)


def test_rejects_resource_boundary_as_empirical():
    assert "resource_blocked_evidence_rejected" in validate_evidence_unit(make_unit(resource_blocked=True, validation_status="SAFE_RESOURCE_BLOCKED"))


def test_rejects_pending_or_placeholder_status():
    assert "pending_or_placeholder_status" in validate_evidence_unit(make_unit(validation_status="PENDING_GPU_EXECUTION"))
