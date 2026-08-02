from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from coregraph.contracts.axes import AccessRegime
from coregraph.experiments.config import HardwareConfig, OutputConfig, RunConfig
from coregraph.experiments.manifests import RunStatus
from coregraph.experiments.prediction_export import export_predictions
from coregraph.experiments.resume import ResumeDecision, audit_resume
from coregraph.experiments.runner import ExperimentRunner, deterministic_early_stopping


def config(tmp_path: Path, **updates) -> RunConfig:
    base = RunConfig(
        dataset="fixture",
        task="node_classification",
        source_contracts=("source",),
        target_contracts=("target",),
        access_regime=AccessRegime.DG_NO_TARGET,
        experts=("logistic",),
        router="factorised",
        objective="bce",
        metrics=("auprc",),
        seed=0,
        hardware=HardwareConfig(),
        data_checksum="a" * 64,
        code_commit="2dec25eac1d7a8951f9d4639f49e889c4c9ca486",
        dependency_lock="b" * 64,
        dataset_manifest="c" * 64,
        output=OutputConfig(
            prediction_export=False,
            output_root=str(tmp_path),
        ),
        smoke=True,
    )
    return replace(base, **updates)


def test_runner_smoke_and_resume(tmp_path: Path) -> None:
    runner = ExperimentRunner(config(tmp_path))
    manifest = runner.run(lambda _: {"metric": 0.5})
    assert manifest.status is RunStatus.SMOKE_PASS
    assert audit_resume(runner.config, runner.manifest_path).decision is ResumeDecision.SKIP_COMPLETE


def test_stale_hash_and_interrupted_manifest(tmp_path: Path) -> None:
    runner = ExperimentRunner(config(tmp_path))
    runner.plan()
    assert audit_resume(runner.config, runner.manifest_path).decision is ResumeDecision.RERUN_INCOMPLETE
    payload = json.loads(runner.manifest_path.read_text())
    payload["config_hash"] = "stale"
    runner.manifest_path.write_text(json.dumps(payload))
    assert audit_resume(runner.config, runner.manifest_path).decision is ResumeDecision.RERUN_STALE_HASH


def test_result_checksum_mutation_invalidates_resume(tmp_path: Path) -> None:
    runner = ExperimentRunner(config(tmp_path))
    runner.run(lambda _: {"metric": 0.5})
    runner.result_path.write_text("{}\n")
    assert (
        audit_resume(runner.config, runner.manifest_path).decision
        is ResumeDecision.RERUN_INVALID_OUTPUT
    )


def test_prediction_export_checksum_is_bound_to_manifest(tmp_path: Path) -> None:
    configured = config(
        tmp_path,
        output=OutputConfig(prediction_export=True, output_root=str(tmp_path)),
    )
    runner = ExperimentRunner(configured)

    def execute(_: RunConfig) -> dict[str, str]:
        path = runner.output_dir / "predictions.csv"
        checksum = export_predictions(
            path,
            [
                {
                    "node_id": "node:1",
                    "score": 0.7,
                    "y_true": 1,
                }
            ],
            required_columns=("node_id", "score", "y_true"),
        )
        return {"prediction_path": str(path), "prediction_checksum": checksum}

    manifest = runner.run(execute)
    assert manifest.prediction_checksum
    assert audit_resume(configured, runner.manifest_path).decision is ResumeDecision.SKIP_COMPLETE


def test_deterministic_early_stopping() -> None:
    assert deterministic_early_stopping([1, 2, 2, 2, 3], patience_checks=2) == 4
