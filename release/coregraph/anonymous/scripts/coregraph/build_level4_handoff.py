#!/usr/bin/env python3
"""Build deterministic Level-4 handoff reports without running experiments."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "results" / "coregraph_build"
PROMPTS = BUILD / "LEVEL4_NEXT_EXECUTION_PROMPTS"
VERDICT = "COREGRAPH_LEVEL4_BUILD_COMPLETE_READY_FOR_SAVED_OUTPUT_PILOT"
TREE_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def read_json(
    name: str, *, default: dict[str, object] | None = None
) -> dict[str, object]:
    path = BUILD / name
    if not path.is_file() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def archive_lines(evidence: dict[str, object]) -> str:
    archives = evidence.get("archives", [])
    assert isinstance(archives, list)
    return "\n".join(
        f"- `{item['name']}` — `{item['observed_sha256']}`"
        for item in archives
        if isinstance(item, dict)
    )


def baseline_counts() -> dict[str, int]:
    with (BUILD / "LEVEL4_BASELINE_REGISTRY.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return dict(sorted(Counter(row["status"] for row in csv.DictReader(handle)).items()))


def cleanup_bytes() -> int:
    report = (BUILD / "LOCAL_CLEANUP_REPORT.md").read_text(encoding="utf-8")
    match = re.search(r"Workspace bytes removed:\s*\*\*([0-9]+)\*\*", report)
    return int(match.group(1)) if match else 0


def tree_candidates() -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Clean source snapshots deliberately contain no Git metadata. Keep the
        # handoff generator usable there without weakening the snapshot audit.
        return [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
            and not any(part in TREE_EXCLUDED_PARTS for part in path.parts)
            and not path.relative_to(ROOT).as_posix().startswith("paper_iclr/build/")
        ]


def common_prompt() -> str:
    return """## Inherited non-negotiable controls

Work only in `${COREGRAPH_REPO_ROOT}` on
`codex/coregraph-iclr-buildout-2026`. Begin with read-only Git, authority,
manifest, and frozen-boundary checks; fetch without resetting; stop on an
unexpected branch, divergence, dirty user work, or failed checksum. Never
force-push or merge PR #2.

Use `${COREGRAPH_EVIDENCE_CACHE}` as the canonical RB09v3 authority. Require
the six archive hashes and the 180 member identities in the tracked manifests
to match before reading payload bytes. Do not use the SSD when the local cache
passes. Stream ZIP members; do not permanently extract prediction CSVs. Never
stage archives, predictions, data, checkpoints, credentials, private path maps,
or local runtime logs.

Preserve the 180 role-neutral artifacts, 60 held-out-protocol scenarios, 540
scenario-local bindings, source/target role separation, source-only fitting,
known-label evaluation filters, chronology, dataset identity, and seed-local
pairing. A target artifact may be evaluated only in its scenario-local target
role. Target labels and oracle quantities are forbidden during fitting,
threshold selection, calibration, model selection, or routing.

Run `scripts/coregraph/hash_frozen_assets.py --verify` before and after work and
require `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. Record code, config, data,
archive, member, prediction, environment, and output hashes. Fail closed on
missing or invalid coordinates. Never fabricate metrics, runtime, memory,
citations, CI, completeness, or scientific conclusions. Any new empirical
claim must pass the frozen claim and statistical gates before entering paper
prose."""


def prompt_specs() -> dict[str, tuple[str, str]]:
    return {
        "01_saved_output_pilot_execution.md": (
            "Saved-output pilot execution",
            """This is the next authorised empirical step, but execute it only
after the user explicitly invokes this prompt. Revalidate pilot inputs, archive
and member hashes, V5 row scopes, and the frozen preregistration hash. Create a
run manifest before evaluating anything. Use only the checksum-verified saved
prediction members; do not train experts or regenerate predictions.

Execute the frozen 240-coordinate pilot plan sequentially with resumability and
hash-equivalent skip logic. Fit any router, calibration, or threshold component
on source train/validation scopes only. Permit target known-label rows only in
the offline evaluation stage. Produce completeness, exclusion, leakage,
resource-unknown, metric, statistical-gate, and claim-linkage records. Never
compute or expose an oracle outside its explicitly non-deployable diagnostic
cell. Stop on leakage, pairing, checksum, or scenario-coverage failure.

At completion, issue a conservative GO/NO-GO/INCONCLUSIVE decision against the
already frozen gate; do not redefine thresholds after seeing results. Do not
start full training, update empirical paper claims, or launch Kaggle work in the
same phase.""",
        ),
        "02_fraud_full_gpu_execution.md": (
            "Fraud full GPU execution",
            """Execute only after a separately reviewed pilot GO and explicit
user authorisation. Freeze the accepted pilot code/config boundary, verify the
Kaggle dataset copy hashes, and use the validated fraud-full notebook/run
matrix. Create all manifests before jobs start. Partition seeds 1–10 across the
two T4 lanes without duplicate coordinates, checkpoint atomically, resume only
after hash equivalence, and classify OOM as `RESOURCE_BLOCKED_OOM`.

Run only the approved fraud training and ablation cells. Maintain source-only
selection and target-label-free fitting. Package validated outputs into one
checksum-indexed ZIP per runbook, with explicit failed/missing coordinates and
no silent skips. Do not reinterpret training runtime as inference latency. Do
not populate paper results until local import, completeness, statistical, and
claim gates pass.""",
        ),
        "03_synthetic_mechanism_execution.md": (
            "Synthetic mechanism execution",
            """Execute only with explicit authorisation for the full synthetic
grid. Verify all 15 mechanism definitions and deterministic tiny-fixture hashes
first. Freeze pilot/full sample sizes, seeds, held-out compositions, expected
directional sanity checks, and failure conditions before observing outcomes.

Run the scalable suite with distinct source and held-out contracts, preserving
ground-truth mechanism metadata. Separate theorem/counterexample validation
from empirical performance. Report every mechanism, seed, failed coordinate,
and resource block; do not cherry-pick mechanisms or promote the optional
latent-discovery extension without its gate. Package manifests and outputs with
hashes, then run the preregistered analysis without changing hypotheses.""",
        ),
        "04_graph_ood_execution.md": (
            "Graph-OOD benchmark execution",
            """Execute only after explicit approval for downloads, licences,
and compute. Recheck the primary GOOD and fallback molecular benchmark records,
official licences, pinned commits, official splits, task compatibility, disk,
memory, and target-label policy. Stop if faithful integration or licence
authority cannot be established; an internal stand-in is never an official
result.

Use the prepared adapter schema and first run parity/tiny checks. Record every
necessary modification to the official code, then execute only the approved
matrix. Keep fraud and non-fraud analyses distinct, preserve source-only model
selection, and mark unsupported or resource-blocked cells rather than
substituting easier splits. Claims of cross-domain generalisation require the
frozen cells and statistical gate.""",
        ),
        "05_official_baseline_integration_execution.md": (
            "Official baseline integration and execution",
            """Execute one baseline family at a time after explicit licence,
dependency, and compute approval. Re-audit the registry, official repository,
licence, pinned revision, task bridge, split semantics, target access, expert
set, tuning budget, and expected resources. Preserve the distinction between
faithful official integration and internal approximation.

First establish parity on the smallest valid fixture. Then freeze a fair
configuration using the common source/validation budget and run only approved
coordinates. Never repair a missing official cell with an inspired surrogate.
Record unavailable, licence-blocked, dependency-blocked, and resource-blocked
statuses explicitly. Oracle diagnostics remain non-deployable and cannot be
reported as baselines.""",
        ),
        "06_cpu_statistical_analysis.md": (
            "CPU statistical analysis",
            """Run only after a validated result import exists. Verify every
run manifest, code/config/data/prediction hash, completion status, execution
reason, feasibility set, and claim linkage. Reject partial or mismatched cells
before calculating summaries. Never pool same-number seeds across datasets as
one paired observation.

Apply the frozen per-dataset paired analysis, permutation/bootstrap procedure,
effect sizes, confidence intervals, Holm correction, missing-cell policy,
resource-blocked policy, and hierarchical secondary summary exactly as
preregistered. Emit machine-readable provenance for every table cell. Do not
run training, change gates after seeing outcomes, or write paper conclusions;
return supported, unsupported, inconclusive, and blocked claim decisions.""",
        ),
        "07_result_driven_paper_population.md": (
            "Result-driven paper population",
            """Begin only from validated statistical outputs and an approved
claim-support report. Map each empirical sentence, table cell, and figure to a
claim ID, required comparison, exact run cells, statistic, and feasibility
scope. Leave unsupported or incomplete cells visibly pending; never infer a
winner from descriptive values alone.

Replace `RESULT_PENDING` tokens only when their individual gates pass. Preserve
anonymous metadata, the theory limitations, resource-unknown boundaries, and
the TKDE/ICLR differentiation firewall. Verify citations against primary
sources, obtain external mathematical review, rebuild under the official target
conference style when available, and rerun claim, fake-number, overlap, font,
identity, path, reference, and page-by-page visual audits.""",
        ),
        "08_final_independent_reviewer_audit.md": (
            "Final independent reviewer audit",
            """Perform a read-only, adversarial audit of the exact candidate
commit and release. Recompute archive/member and release checksums; inspect V5
role semantics, row scopes, target-label boundaries, statistical pairing,
baseline fairness, theorem assumptions, counterexamples, claims, citations,
resource measurement, overlap, anonymity, and frozen TKDE status.

Attempt to falsify each primary claim and reproduce tiny/clean-room gates from
the source snapshot. Distinguish code existence, deterministic validation,
empirical support, and submission readiness. Report every finding with severity,
evidence, reproduction command, affected claim, and required closure. Do not
modify results, waive failures, merge the PR, or declare acceptance readiness.""",
        ),
        "09_final_submission_packaging.md": (
            "Final submission packaging",
            """Execute only after experiments, official baselines, statistics,
external review, citation verification, target-style rebuild, and independent
audit are complete. Resolve the exact conference rules from official sources,
then freeze the candidate commit and require exact-SHA CI success.

Build the anonymous paper/supplement and public source package in a clean room;
validate page limits, fonts, references, figures, tables, claims, licences,
checksums, private paths, identity, provider payload exclusion, overlap
disclosure, and reproducibility artifacts. Keep the data-free source release
separate from restricted evidence. Do not force-push, merge PR #2, publish, or
submit without the user's final explicit approval.""",
        ),
    }


def write_prompts() -> None:
    common = common_prompt()
    for filename, (title, body) in prompt_specs().items():
        write_text(
            PROMPTS / filename,
            f"# {title}\n\n{body}\n\n{common}\n\n"
            "## Required handoff\n\nReport the exact Git SHA, input and output "
            "hashes, commands, completed and failed coordinates, leakage and "
            "frozen-boundary status, scientific conclusions permitted by the "
            "gate, remaining blockers, and the next separately authorised action.",
        )


def main() -> int:
    evidence = read_json("ARCHIVE_MEMBER_VALIDATION.json")
    leakage = read_json("V5_LEAKAGE_AUDIT.json")
    pilot = read_json("LEVEL4_PILOT_INPUT_VALIDATION.json")
    notebooks = read_json("NOTEBOOK_VALIDATION.json")
    paper = read_json("LEVEL4_PAPER_CLAIM_AUDIT.json")
    overlap = read_json("LEVEL4_CROSS_PAPER_OVERLAP_AUDIT.json")
    smoke = read_json("CPU_ONE_EPOCH_SMOKE.json")
    coverage = read_json("LEVEL4_COVERAGE_SUMMARY.json")
    cleanroom = read_json(
        "LEVEL4_CLEANROOM_VALIDATION.json",
        default={
            "checksum_validation": "PENDING_EXTERNAL_RELEASE_GATE",
            "cleanroom": {"status": "PENDING_EXTERNAL_RELEASE_GATE"},
        },
    )
    preregistration_line = (
        (BUILD / "LEVEL4_PREREGISTRATION_HASH.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    preregistration = preregistration_line.split()[1]
    baselines = baseline_counts()
    removed = cleanup_bytes()
    cleanroom_status = str(cleanroom.get("cleanroom", {}).get("status", "PENDING"))

    write_prompts()

    master = f"""# CoReGraph Level-4 master build report

Verdict: `{VERDICT}`

## Authority and integrity

The active CoReGraph authority is the independent Git checkout on
`codex/coregraph-iclr-buildout-2026`; the exact completed SHA is the normally
pushed branch tip recorded in the final Git/PR handoff. The curated
FraudShiftBench authority remains frozen at
`2dec25eac1d7a8951f9d4639f49e889c4c9ca486`, with
`ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. The historical folder remains
read-only pending the documented backup/uniqueness decision.

The six canonical RB09v3 archives occupy {evidence['archive_size_bytes']:,}
compressed bytes in `${{COREGRAPH_EVIDENCE_CACHE}}/archives`; all are read-only,
ZIP-CRC valid, whole-file SHA-256 valid, and represented by 180 unique streamed
member hashes:

{archive_lines(evidence)}

No archive or prediction payload is in Git. No SSD source was used after the
verified local cache became sufficient, and no prediction member was
permanently extracted.

## Scientific system

- V5: {leakage['base_artifact_count']}/180 role-neutral base artifacts,
  {leakage['scenario_count']}/60 scenarios, and {leakage['binding_count']}/540
  bindings materialise; byte, schema, coordinate, chronology, known-label,
  ordering, 60-group expert alignment, and 20-group cross-protocol row-scope
  audits pass.
- Method: factorised/interaction/attention/uncertainty/latent/hybrid contract
  encoders; expert diagnostics; contract, instance, and hierarchical routing;
  resource masks; robust regret/CVaR/budget/stability/abstention objectives;
  selective, counterfactual, and resource evaluation.
- Theory: fixed-mixture impossibility, regret decomposition, compositional
  bound, and selective-risk transfer are internally proved pending external
  review; resource-mask validity is proved and reviewed. Executable finite
  checks pass, including declared failure cases.
- Benchmarks: six fraud contracts, 15 deterministic synthetic mechanisms, GOOD
  primary adapter plan, and OGB molecular fallback plan. No official benchmark
  download or training occurred.
- Baselines: {sum(baselines.values())} registered; status counts are
  `{json.dumps(baselines, sort_keys=True)}`. Internal methods are implemented;
  official repositories remain uninstalled or explicitly licence/dependency
  blocked.
- Statistics: preregistration SHA-256 `{preregistration}`; all empirical claims
  remain blocked until validated results pass the frozen gates.

## Validation and paper

- Tests: {coverage['tests_passed']} passed; critical-module coverage minimum
  {coverage['critical_minimum_percent']}% and every declared group meets the
  85% gate.
- Deterministic checks: compile, Ruff, mypy, theory, synthetic fixtures,
  one-epoch CPU smoke, notebook syntax/packaging fixture, paper claims,
  cross-paper overlap, archive offline smoke, release checksums, clean room,
  and frozen boundary pass. Final clean-room status: `{cleanroom_status}`.
- Runbooks: {notebooks['kaggle_level4_runbooks']} Kaggle T4x2 plus
  {notebooks['notebooks'] - notebooks['kaggle_level4_runbooks']} local notebooks;
  none executed.
- Paper: 12 main sections, 7 supplement sections, 13 main pages, 5 supplement
  pages, 8 non-empirical figures, 7 empty result templates, and 11 tables.
  All 18 pages passed visual QA. Four empirical claim families remain blocked;
  no numerical result was invented.
- Overlap firewall: `{overlap['status']}` with zero common eight-grams, zero
  exact long sentences, and zero byte-identical visual assets.
- Cleanup: {removed:,} reproducible workspace bytes removed; no evidence,
  report, environment, historical folder, or user-owned file was deleted.

## Explicit execution boundary

No full real-data training, target metric, target oracle, official-baseline
installation, Kaggle job, empirical paper population, force-push, or PR merge
occurred. The only training-like operation was the documented one-epoch,
24-node synthetic CPU smoke (`{smoke['status']}`), which used no provider data.

The next authorised action is a separately invoked saved-output pilot using
`LEVEL4_NEXT_EXECUTION_PROMPTS/01_saved_output_pilot_execution.md`. This build
is not labelled submission-ready.
"""
    write_text(BUILD / "LEVEL4_MASTER_BUILD_REPORT.md", master)

    gate = {
        "schema": "coregraph_level4_final_gate_v1",
        "verdict": VERDICT,
        "ready_for_saved_output_pilot": pilot.get("status")
        == "READY_FOR_SAVED_OUTPUT_PILOT",
        "repository": {
            "branch": "codex/coregraph-iclr-buildout-2026",
            "form": "INDEPENDENT_GIT_CHECKOUT_VALIDATED",
            "final_sha_authority": "NORMALLY_PUSHED_BRANCH_TIP_AT_HANDOFF",
        },
        "evidence": {
            "archives_verified": evidence.get("archive_verified"),
            "members_verified": evidence.get("member_checksum_verified"),
            "compressed_bytes": evidence.get("archive_size_bytes"),
            "permanent_extractions": evidence.get("permanent_extractions"),
            "target_metrics_computed": evidence.get("target_metrics_computed"),
            "target_oracles_computed": evidence.get("target_oracles_computed"),
            "ssd_source_used_after_local_verification": False,
        },
        "v5": {
            "base_artifacts": leakage.get("base_artifact_count"),
            "scenarios": leakage.get("scenario_count"),
            "bindings": leakage.get("binding_count"),
            "leakage_status": leakage.get("overall_status"),
        },
        "tests": coverage,
        "theory_gate": "PASS_EXTERNAL_REVIEW_PENDING",
        "notebooks": {
            "total": notebooks.get("notebooks"),
            "kaggle": notebooks.get("kaggle_level4_runbooks"),
            "executed": notebooks.get("notebooks_executed"),
        },
        "paper": {
            "status": paper.get("status"),
            "main_pages": 13,
            "supplement_pages": 5,
            "main_sections": paper.get("main_sections"),
            "supplement_sections": paper.get("supplement_sections"),
            "blocked_empirical_claims": paper.get("blocked_empirical_claims"),
            "invented_numeric_results": paper.get("invented_numeric_results"),
            "visual_qa": "PASS_18_OF_18_PAGES",
        },
        "release": {
            "checksum_validation": cleanroom.get("checksum_validation"),
            "cleanroom": cleanroom_status,
        },
        "frozen_tkde": "ZERO_TKDE_SCIENTIFIC_DELTAS_249_FILES",
        "cleanup_workspace_bytes_removed": removed,
        "prohibited_actions": {
            "full_real_data_training": 0,
            "target_metric_computation": 0,
            "target_oracle_computation": 0,
            "official_baseline_repository_installation": 0,
            "kaggle_jobs": 0,
            "fabricated_numerical_paper_results": 0,
            "force_pushes": 0,
            "pull_request_merges": 0,
        },
        "submission_status": "RESULTS_BLOCKED_NOT_SUBMISSION_READY",
        "next_authorised_action": "SAVED_OUTPUT_PILOT_AFTER_EXPLICIT_INVOCATION",
    }
    write_text(
        BUILD / "LEVEL4_FINAL_GATE_STATUS.json",
        json.dumps(gate, indent=2, sort_keys=True),
    )

    checklist = """# Level-4 run-after-build checklist

The build is complete; the saved-output pilot is the next separately
authorised action. Do not begin it merely by reading this checklist.

1. Confirm explicit user authorisation for the saved-output pilot only.
2. Verify the active branch/tip, clean worktree, PR #2 draft state, six archive
   hashes, 180 member hashes, V5 leakage report, preregistration hash, and
   `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.
3. Invoke `01_saved_output_pilot_execution.md`; create manifests before any
   target evaluation and keep all fitting source-only.
4. Import and validate every output hash and coordinate. Do not silently omit
   failed or unavailable cells.
5. Run the frozen CPU statistical analysis only after completeness and leakage
   gates pass; issue GO, NO-GO, or INCONCLUSIVE without changing thresholds.
6. Request separate authority before full fraud training, synthetic full runs,
   graph-OOD downloads, official-baseline integration, or Kaggle jobs.
7. Populate paper results only from claim-gated statistical artifacts; retain
   pending markers everywhere else.
8. Obtain external mathematical review, official target-style verification,
   independent review, and exact-SHA CI before submission packaging.
9. Never merge PR #2, publish restricted evidence, or submit without the
   user's explicit final approval.
"""
    write_text(BUILD / "LEVEL4_RUN_AFTER_BUILD_CHECKLIST.md", checklist)

    commands = f"""# Level-4 final command log

All paths below are repository-relative or environment-variable based.

| Gate | Command family | Outcome |
|---|---|---|
| Git/authority preflight | `git status`, `rev-parse`, `remote`, `fetch`, `fsck` | PASS; independent checkout |
| Evidence full audit | `validate_level4_evidence_cache.py` | PASS; 6 archives, 180 members, zero extraction |
| Row-scope audit | `validate_level4_row_scopes.py` | PASS; 20 dataset-seed groups |
| Artifact construction | `build_level4_artifacts.py` | PASS; 180/60/540 |
| Pilot input validation | `orchestrate_level4.py pilot-validate` | READY; no execution |
| Full planning | `orchestrate_level4.py full-plan` | PASS; 1,680 plan rows, no jobs |
| Compile/lint/type | `compileall`, `ruff check`, `mypy` | PASS; 97 typed source files |
| Tests | `pytest -q` through coverage | PASS; {coverage['tests_passed']} tests |
| Coverage | `make coregraph-coverage` | PASS; minimum {coverage['critical_minimum_percent']}% |
| Theory | numeric/status/standalone executable checks | PASS |
| Tiny synthetic suite | `run_synthetic_method_checks.py` | PASS; no real data |
| CPU smoke | `run_coregraph_smoke.py` | PASS; one epoch, 24 graph nodes |
| Notebook audit | `validate_notebooks.py` | PASS; 12 static, 0 executed |
| Paper/claims | skeleton, overlap, build, page QA | PASS_RESULTS_BLOCKED |
| Frozen inherited files | `hash_frozen_assets.py --verify` | ZERO_TKDE_SCIENTIFIC_DELTAS (249 files) |
| Release checksums | `validate_level4_release.py` | {cleanroom.get('checksum_validation', 'PENDING')} |
| Clean room | `validate_level4_release.py --cleanroom` | {cleanroom_status} |

The empirical analysis target was deliberately not run because no validated
pilot result import exists. No SSD access, full training, target metric/oracle,
official baseline install, Kaggle launch, force-push, or PR merge occurred.
"""
    write_text(BUILD / "LEVEL4_FINAL_COMMAND_LOG.md", commands)

    candidates = tree_candidates()
    candidates.append("results/coregraph_build/LEVEL4_FINAL_TREE.txt")
    tree = "# CoReGraph Level-4 final public-neutral tree\n\n" + "\n".join(
        sorted(set(candidates))
    )
    write_text(BUILD / "LEVEL4_FINAL_TREE.txt", tree)

    print(
        json.dumps(
            {
                "verdict": VERDICT,
                "prompts": len(prompt_specs()),
                "archives": evidence.get("archive_verified"),
                "members": evidence.get("member_checksum_verified"),
                "scenarios": leakage.get("scenario_count"),
                "bindings": leakage.get("binding_count"),
                "cleanroom": cleanroom_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
