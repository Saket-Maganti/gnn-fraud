#!/usr/bin/env python3
"""Create the typed claim-to-evidence ledger for the TKDE rebuild."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tkde_rebuild"

FIELDS = [
    "claim_id",
    "claim_type",
    "paper_facing_claim",
    "scope",
    "quantifier",
    "comparison",
    "metric",
    "direction",
    "uncertainty_requirement",
    "required_evidence",
    "matched_evidence_ids",
    "prediction_requirement",
    "pairing_or_statistical_requirement",
    "support_status",
    "permitted_wording",
    "prohibited_wording",
    "rationale",
    "paper_location",
]


def load_inventory() -> list[dict[str, str]]:
    with (OUT / "EVIDENCE_INVENTORY.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def match(rows: list[dict[str, str]], *prefixes: str) -> str:
    ids = sorted(
        row["evidence_id"]
        for row in rows
        if any(row["evidence_id"].startswith(prefix) for prefix in prefixes)
    )
    return ";".join(ids)


def claim(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(values)
    return row


def build_claims(inventory: list[dict[str, str]]) -> list[dict[str, Any]]:
    validation_path = OUT / "FRAMEWORK_VALIDATION_CASES.csv"
    framework_validated = False
    if validation_path.exists():
        with validation_path.open("r", encoding="utf-8", newline="") as handle:
            validation_rows = list(csv.DictReader(handle))
        framework_validated = bool(validation_rows) and all(row.get("pass", "").lower() == "true" for row in validation_rows)
    rb09 = match(inventory, "RB09::")
    v24_main = match(inventory, "V24::RB41::")
    v26 = match(inventory, "V26::ibm_aml::")
    v27 = match(inventory, "V27::ibm_aml::")
    v28 = match(inventory, "V28::ibm_aml::")
    graphsafe = match(inventory, "RB15::", "RB15b::", "RB16::", "RB17::")
    blocked = match(inventory, "RB18::", "V22::RB30::", "V24::DGRAPHFIN_FIXED", "V26::ibm_aml::hi-large", "V26::ibm_aml::li-large", "V28::ibm_aml::hi-medium::gine", "V28::ibm_aml::li-medium::gine")
    claims = [
        claim(
            claim_id="C01",
            claim_type="empirical ranking",
            paper_facing_claim="On Elliptic, the leading AUPRC model among MLP, GCN, and GraphSAGE changes between strict-inductive and isolated-inductive graph visibility.",
            scope="Elliptic; MLP/GCN/GraphSAGE; strict versus isolated; seeds 1-10",
            quantifier="argmax over three models within each protocol",
            comparison="strict-inductive versus isolated-inductive",
            metric="AUPRC",
            direction="GraphSAGE leads under strict; GCN leads under isolated",
            uncertainty_requirement="paired seed deltas and bootstrap/t interval for each model; no unpaired aggregate claim",
            required_evidence="complete 3 model x 2 protocol x 10 seed grid",
            matched_evidence_ids=rb09,
            prediction_requirement="saved prediction manifest available; aggregate result rows sufficient for mean ranking",
            pairing_or_statistical_requirement="pair by dataset, model, seed",
            support_status="SUPPORTED",
            permitted_wording="the evaluated Elliptic leaderboard reverses at the top under the two visibility contracts",
            prohibited_wording="evaluation protocols always reverse rankings",
            rationale="All 60 required Elliptic strict/isolated rows are present; MLP is an invariant control.",
            paper_location="Abstract; RQ1",
        ),
        claim(
            claim_id="C02",
            claim_type="negative control",
            paper_facing_claim="The feature-only MLP is numerically invariant to the strict-versus-isolated graph-visibility switch on both Elliptic and DGraphFin.",
            scope="Elliptic and DGraphFin; MLP; strict versus isolated; seeds 1-10",
            quantifier="all paired seeds in the locked grid",
            comparison="same MLP scores under two graph-visibility contracts",
            metric="AUPRC, AUROC, F1",
            direction="zero recorded delta",
            uncertainty_requirement="report exact paired zero and note shared feature-only computation",
            required_evidence="four complete MLP cells and source implementation contract",
            matched_evidence_ids=match(inventory, "RB09::elliptic::inductive_isolated::mlp", "RB09::elliptic::strict_inductive::mlp", "RB09::dgraphfin::inductive_isolated::mlp", "RB09::dgraphfin::strict_inductive::mlp"),
            prediction_requirement="prediction references available",
            pairing_or_statistical_requirement="exact seed pairing; no significance test needed for copied invariant branch",
            support_status="SUPPORTED",
            permitted_wording="the MLP negative control is unchanged by the graph-only visibility intervention",
            prohibited_wording="the two protocols are identical",
            rationale="The intervention changes graph access, which the MLP does not consume; all recorded paired deltas are zero.",
            paper_location="RQ1",
        ),
        claim(
            claim_id="C03",
            claim_type="heterogeneous treatment effect",
            paper_facing_claim="Strict-to-isolated effects are architecture- and dataset-dependent rather than a uniform penalty or benefit.",
            scope="Elliptic and DGraphFin; GCN and GraphSAGE; strict versus isolated; seeds 1-10",
            quantifier="heterogeneity across four matched dataset-model cells",
            comparison="isolated minus strict",
            metric="AUPRC with AUROC/F1 as secondary metrics",
            direction="mixed; Elliptic GCN improves while GraphSAGE declines, and DGraphFin effects are smaller/mixed",
            uncertainty_requirement="paired confidence intervals, effect sizes, and multiplicity-aware tests",
            required_evidence="complete matched cells",
            matched_evidence_ids=rb09,
            prediction_requirement="manifest-backed predictions available",
            pairing_or_statistical_requirement="pair by seed; Holm correction over declared comparison family",
            support_status="SUPPORTED",
            permitted_wording="effects depend on architecture and dataset in the evaluated grid",
            prohibited_wording="graph isolation is always harmful or always beneficial",
            rationale="Observed delta directions and magnitudes differ across GCN and GraphSAGE cells.",
            paper_location="RQ1; RQ2",
        ),
        claim(
            claim_id="C04",
            claim_type="universal claim",
            paper_facing_claim="Graph structure is universally harmful under deployment-realistic evaluation.",
            scope="all datasets, constructions, protocols, and graph models",
            quantifier="universal",
            comparison="graph versus feature-only",
            metric="all",
            direction="harm",
            uncertainty_requirement="complete cross-dataset/model/construction evidence",
            required_evidence="not satisfiable from current finite grid; observed cells include benefits, harms, and nulls",
            matched_evidence_ids=f"{rb09};{v26};{v27};{v28}",
            prediction_requirement="would require complete prediction-backed scope",
            pairing_or_statistical_requirement="would require prespecified complete-cell family",
            support_status="REFUTED_IN_SCOPE",
            permitted_wording="the evaluated evidence rejects a uniform graph-harm narrative",
            prohibited_wording="GNNs fail universally; graph methods are always harmful",
            rationale="Some graph cells improve, others decline, and strong non-graph baselines also win; the universal direction is false on observed cells.",
            paper_location="Introduction; implications",
        ),
        claim(
            claim_id="C05",
            claim_type="empirical baseline",
            paper_facing_claim="Histogram gradient boosting has the highest mean AUPRC in every IBM AML baseline-grid variant/protocol cell.",
            scope="IBM AML HI/LI Small/Medium; two temporal protocols; HistGB/LogReg/GraphSAGE-h32; seeds 1-10",
            quantifier="all eight complete baseline-grid cells",
            comparison="three model families",
            metric="mean AUPRC",
            direction="HistGB highest",
            uncertainty_requirement="mean with 95% interval; paired seed contrasts to alternatives",
            required_evidence="24 complete cells (4 variants x 2 protocols x 3 models)",
            matched_evidence_ids=v26,
            prediction_requirement="60 prediction exports per variant; all available",
            pairing_or_statistical_requirement="pair by variant, protocol, seed",
            support_status="SUPPORTED",
            permitted_wording="HistGB leads mean AUPRC in all eight evaluated baseline-grid contexts",
            prohibited_wording="trees are universally best for AML",
            rationale="The claim is restricted to the locked synthetic baseline grid and AUPRC.",
            paper_location="Abstract; RQ3",
        ),
        claim(
            claim_id="C06",
            claim_type="metric disagreement",
            paper_facing_claim="IBM AML model/construction conclusions change between ranking and thresholded decision metrics.",
            scope="IBM AML Small/Medium locked V26-V28 cells",
            quantifier="cell-level winner/rank comparisons",
            comparison="AUPRC/AUROC ranks versus F1@0.5 and review-budget behavior",
            metric="AUPRC, AUROC, F1@0.5, Precision/Recall@K",
            direction="non-identical winners and ranks",
            uncertainty_requirement="seed-aggregate ranks plus Spearman/Kendall statistics; disclose fixed threshold",
            required_evidence="complete metric rows and prediction-backed review-budget subset",
            matched_evidence_ids=f"{v26};{v27};{v28}",
            prediction_requirement="required for review-budget metrics; V27/V28 manifests available",
            pairing_or_statistical_requirement="rank within identical variant/protocol/feasible configuration set",
            support_status="SUPPORTED",
            permitted_wording="rank and decision metrics answer different questions and disagree in the evaluated cells",
            prohibited_wording="higher AUROC implies better operational decisions",
            rationale="The same cells expose all metrics, permitting matched winner and rank comparisons.",
            paper_location="RQ3; RQ4",
        ),
        claim(
            claim_id="C07",
            claim_type="matched ablation",
            paper_facing_claim="Original transaction edge features carry AUPRC signal relative to zeroed or shuffled edge-feature controls across the complete IBM AML GraphSAGE grid.",
            scope="HI/LI Small/Medium; both protocols; edge-aware GraphSAGE h64 versus NoEdge/ShuffledEdge; seeds 1-10",
            quantifier="eight matched variant-protocol contexts",
            comparison="original minus zeroed/shuffled edge features",
            metric="AUPRC",
            direction="positive in each evaluated matched context",
            uncertainty_requirement="paired seed deltas and confidence intervals; correction over 16 contrasts",
            required_evidence="V27 reference and V28 controls on identical cells",
            matched_evidence_ids=f"{v27};{v28}",
            prediction_requirement="available for every compared run",
            pairing_or_statistical_requirement="pair by variant, protocol, seed; Holm correction",
            support_status="SUPPORTED",
            permitted_wording="the original edge-feature representation outperforms the two matched controls on mean AUPRC in this grid",
            prohibited_wording="edge features causally improve all fraud models",
            rationale="Graph structure and feasible cells are held fixed; interpretation remains representation-specific.",
            paper_location="RQ3",
        ),
        claim(
            claim_id="C08",
            claim_type="heterogeneous ablation",
            paper_facing_claim="Degree capping is not uniformly beneficial: its effect changes with scale, class-prior regime, protocol, and metric.",
            scope="IBM AML HI/LI Small/Medium; both protocols; DegreeCap versus edge-aware GraphSAGE reference",
            quantifier="matched complete-cell heterogeneity",
            comparison="DegreeCap minus reference",
            metric="AUPRC, AUROC, F1@0.5, runtime",
            direction="mixed",
            uncertainty_requirement="matched intervals; do not pool unequal regimes",
            required_evidence="complete V27 reference and DegreeCap cells",
            matched_evidence_ids=f"{v27};{v28}",
            prediction_requirement="available",
            pairing_or_statistical_requirement="pair by variant/protocol/seed",
            support_status="SUPPORTED",
            permitted_wording="degree capping exposes metric- and regime-dependent tradeoffs",
            prohibited_wording="degree capping improves the benchmark",
            rationale="AUPRC and F1 directions differ across matched contexts.",
            paper_location="RQ3",
        ),
        claim(
            claim_id="C09",
            claim_type="performance-resource tradeoff",
            paper_facing_claim="The recent-window construction is faster and achieves the highest mean AUROC among complete graph configurations on Medium, while its AUPRC remains below the full edge-aware reference.",
            scope="IBM AML HI/LI Medium; both protocols; complete V27/V28 graph configurations",
            quantifier="four Medium variant-protocol contexts",
            comparison="RecentWindow versus reference and other complete constructions",
            metric="runtime, AUROC, AUPRC",
            direction="runtime lower; AUROC higher; AUPRC lower",
            uncertainty_requirement="mean/range runtime and seed intervals; matched contexts only",
            required_evidence="complete Medium cells; GINE excluded as resource-blocked",
            matched_evidence_ids=f"{v27};{v28}",
            prediction_requirement="available for performance values",
            pairing_or_statistical_requirement="pair by variant/protocol/seed; feasibility set declared",
            support_status="SUPPORTED",
            permitted_wording="RecentWindow lies on a metric-dependent speed/performance tradeoff in the Medium grid",
            prohibited_wording="RecentWindow is Pareto-optimal for every objective",
            rationale="Configuration-specific performance and runtime are both recorded; the comparison excludes unavailable GINE.",
            paper_location="RQ4",
        ),
        claim(
            claim_id="C10",
            claim_type="scoped architecture result",
            paper_facing_claim="GINE h64 has the highest mean AUPRC in all four feasible IBM AML Small variant/protocol cells, but Medium GINE is unmeasured because it exhausted T4 memory.",
            scope="IBM AML HI/LI Small for performance; HI/LI Medium for resource status",
            quantifier="all four feasible Small contexts; no Medium performance quantifier",
            comparison="GINE versus complete Small graph configurations",
            metric="AUPRC and runtime; resource status on Medium",
            direction="Small AUPRC higher; Medium unordered",
            uncertainty_requirement="Small seed intervals and matched ranks; explicit unequal feasibility set",
            required_evidence="40 Small GINE outputs plus two blocked Medium records",
            matched_evidence_ids=v28,
            prediction_requirement="Small exports available; Medium absent",
            pairing_or_statistical_requirement="Small-only pairing; no cross-scale aggregate",
            support_status="SUPPORTED_WITH_RESOURCE_BOUNDARY",
            permitted_wording="GINE leads mean AUPRC on the feasible Small cells; Medium remains resource-blocked",
            prohibited_wording="GINE is best across Small and Medium; Medium GINE performs poorly",
            rationale="Performance and feasibility are separate outcome types.",
            paper_location="Abstract; RQ3; resource feasibility",
        ),
        claim(
            claim_id="C11",
            claim_type="resource boundary",
            paper_facing_claim="IBM AML Large variants have no imported empirical performance evidence under the safe resource contract.",
            scope="IBM AML HI-Large and LI-Large",
            quantifier="both Large variants",
            comparison="none",
            metric="resource status only",
            direction="resource-blocked",
            uncertainty_requirement="not applicable",
            required_evidence="canonical V26 lock and zero output counts",
            matched_evidence_ids=match(inventory, "V26::ibm_aml::hi-large::BLOCKED", "V26::ibm_aml::li-large::BLOCKED"),
            prediction_requirement="absence must be explicit",
            pairing_or_statistical_requirement="no predictive comparison permitted",
            support_status="RESOURCE_BLOCKED",
            permitted_wording="Large lies outside the admitted resource envelope and is unmeasured",
            prohibited_wording="Large performs worse; Large full10 evidence",
            rationale="The lock records zero results and predictions for both Large variants.",
            paper_location="Resource feasibility; limitations",
        ),
        claim(
            claim_id="C12",
            claim_type="resource boundary",
            paper_facing_claim="Medium GINE has no performance evidence because both HI and LI lanes encountered a T4 CUDA OOM.",
            scope="IBM AML HI-Medium and LI-Medium; GINE h64",
            quantifier="both planned Medium lanes",
            comparison="none",
            metric="resource status only",
            direction="resource-blocked",
            uncertainty_requirement="not applicable",
            required_evidence="canonical V28 blocked-run records",
            matched_evidence_ids=match(inventory, "V28::ibm_aml::hi-medium::gine", "V28::ibm_aml::li-medium::gine"),
            prediction_requirement="zero exports",
            pairing_or_statistical_requirement="no predictive ranking permitted",
            support_status="RESOURCE_BLOCKED",
            permitted_wording="Medium GINE is outside the measured performance set",
            prohibited_wording="Medium GINE underperforms",
            rationale="Each blocked row records 20 planned outputs and zero results/predictions.",
            paper_location="RQ3; resource feasibility",
        ),
        claim(
            claim_id="C13",
            claim_type="implementation equivalence",
            paper_facing_claim="The sender-receiver construction row is an explicit restatement of the existing account transaction edge contract, not an independent graph transformation.",
            scope="IBM AML V28 sender-receiver configuration versus V27 reference",
            quantifier="all matched cells",
            comparison="payload and scores",
            metric="all recorded metrics",
            direction="identical",
            uncertainty_requirement="exact equality check",
            required_evidence="source transform note and matched result rows",
            matched_evidence_ids=f"{v27};{match(inventory, 'V28::ibm_aml::hi-medium', 'V28::ibm_aml::hi-small', 'V28::ibm_aml::li-medium', 'V28::ibm_aml::li-small')}",
            prediction_requirement="available",
            pairing_or_statistical_requirement="exact row-wise comparison",
            support_status="DIAGNOSTIC_ONLY",
            permitted_wording="contract restatement/equivalence check",
            prohibited_wording="a distinct sender-receiver method independently replicated the result",
            rationale="The implementation documents that existing arrays already encode sender-to-receiver transaction edges.",
            paper_location="Methods; supplement",
        ),
        claim(
            claim_id="C14",
            claim_type="prevalence-normalized diagnostic",
            paper_facing_claim="Raw AUPRC must be interpreted against each test-window prevalence; normalized lift remains a diagnostic, not a new universal metric.",
            scope="IBM AML V26-V28 cells with recorded test prevalence",
            quantifier="cell-level",
            comparison="AUPRC versus random-ranking prevalence baseline",
            metric="AUPRC/prevalence and AUPRC-prevalence",
            direction="varies by variant/protocol",
            uncertainty_requirement="report prevalence and seed aggregation",
            required_evidence="test split stats and AUPRC",
            matched_evidence_ids=f"{v26};{v27};{v28}",
            prediction_requirement="not required beyond locked metrics",
            pairing_or_statistical_requirement="never pool unlike prevalence cells without stratification",
            support_status="DIAGNOSTIC_ONLY",
            permitted_wording="prevalence-normalized AUPRC contextualizes raw values",
            prohibited_wording="normalized AUPRC is prevalence-invariant performance",
            rationale="The random-ranking baseline equals prevalence, but ratio estimates can become unstable at extreme rarity.",
            paper_location="Metrics; supplement",
        ),
        claim(
            claim_id="C15",
            claim_type="bounded decision case study",
            paper_facing_claim="On the aggregate DGraphFin saved-output surface, conservative GraphSafe has lower mean cost-sensitive risk than simple averaging, while corrected cell-level evidence does not establish universal dominance.",
            scope="DGraphFin saved RB15/RB15b policy rows and declared cost setting",
            quantifier="aggregate descriptive contrast with cell-level correction caveat",
            comparison="GraphSafe conservative versus simple average",
            metric="cost-sensitive risk",
            direction="lower aggregate mean",
            uncertainty_requirement="absolute means, paired deltas, confidence intervals, corrected p-values by cell",
            required_evidence="validated saved-output policy tables",
            matched_evidence_ids=graphsafe,
            prediction_requirement="inherits saved prediction imports",
            pairing_or_statistical_requirement="pair by source cell and seed; Holm correction",
            support_status="DIAGNOSTIC_ONLY",
            permitted_wording="bounded DGraphFin decision-risk case study",
            prohibited_wording="GraphSafe universally dominates; GraphSafe repairs ranking",
            rationale="Aggregate means are favorable, but broad corrected best-branch claims are not supported.",
            paper_location="Bounded case study",
        ),
        claim(
            claim_id="C16",
            claim_type="negative method result",
            paper_facing_claim="On Elliptic aggregate policy summaries, simple averaging has better mean F1 and lower mean cost-sensitive risk than GraphSafe; several strict-SAGE GraphSafe contrasts are corrected negative results.",
            scope="Elliptic saved RB15/RB15b policy rows",
            quantifier="aggregate descriptive plus exact corrected cells",
            comparison="simple average and best-validation branch versus GraphSafe variants",
            metric="F1; cost-sensitive risk; worst-block regret",
            direction="mixed/negative for GraphSafe",
            uncertainty_requirement="report exact corrected family and comparator",
            required_evidence="validated policy and statistical-test tables",
            matched_evidence_ids=graphsafe,
            prediction_requirement="inherits saved predictions",
            pairing_or_statistical_requirement="pair by source cell/seed; Holm correction",
            support_status="SUPPORTED",
            permitted_wording="GraphSafe does not dominate simple averaging on Elliptic",
            prohibited_wording="GraphSafe improves all datasets",
            rationale="The negative result is part of the method audit and prevents selective reporting.",
            paper_location="Bounded case study; limitations",
        ),
        claim(
            claim_id="C17",
            claim_type="blocked method claim",
            paper_facing_claim="GraphSafe-TTA universally improves rank and decision metrics across datasets.",
            scope="all datasets/protocols/models",
            quantifier="universal",
            comparison="GraphSafe versus all comparators",
            metric="rank and decision metrics",
            direction="improvement",
            uncertainty_requirement="complete corrected evidence",
            required_evidence="not present; rank metrics are not generally changed by monotone calibration and observed decision effects are mixed",
            matched_evidence_ids=graphsafe,
            prediction_requirement="source predictions available only for bounded families",
            pairing_or_statistical_requirement="would require broad prespecified corrected family",
            support_status="REFUTED_IN_SCOPE",
            permitted_wording="GraphSafe is a bounded decision-analysis case study with mixed outcomes",
            prohibited_wording="universal dominance; ranking repair",
            rationale="Elliptic negative results and non-significant broad comparisons contradict the universal statement.",
            paper_location="Bounded case study",
        ),
        claim(
            claim_id="C18",
            claim_type="formal property",
            paper_facing_claim="A strictly monotone score calibration preserves AUROC and AUPRC rankings but can change fixed-threshold decisions.",
            scope="binary scoring with no tie-order changes",
            quantifier="all strictly increasing score transforms",
            comparison="before versus after transform",
            metric="AUROC, AUPRC, thresholded metrics",
            direction="rank invariant; threshold outcomes may change",
            uncertainty_requirement="proof, not empirical interval",
            required_evidence="mathematical order-preservation argument and calibration citations",
            matched_evidence_ids=graphsafe,
            prediction_requirement="not required for theorem",
            pairing_or_statistical_requirement="not applicable",
            support_status="SUPPORTED_THEORETICALLY",
            permitted_wording="rank invariance under strictly monotone transforms",
            prohibited_wording="calibration necessarily improves AUPRC/AUROC",
            rationale="Both metrics depend on score ordering; a strictly increasing transform leaves the ordering unchanged.",
            paper_location="Framework properties",
        ),
        claim(
            claim_id="C19",
            claim_type="formal property",
            paper_facing_claim="Results from two deployment contracts are interchangeable only for claims whose relevant visibility, construction, selection, budget, and resource components coincide.",
            scope="typed deployment contracts",
            quantifier="contract-relative",
            comparison="two protocol contracts",
            metric="claim support",
            direction="non-equivalence when claim-relevant components differ",
            uncertainty_requirement="definition and proof sketch",
            required_evidence="formal contract semantics",
            matched_evidence_ids=f"{rb09};{v24_main};{v26};{v27};{v28}",
            prediction_requirement="claim dependent",
            pairing_or_statistical_requirement="claim dependent",
            support_status="SUPPORTED_THEORETICALLY",
            permitted_wording="protocol results are contract-relative",
            prohibited_wording="any numerical difference is caused solely by graph visibility",
            rationale="Different contracts answer different estimands unless their claim-relevant components agree.",
            paper_location="Framework properties",
        ),
        claim(
            claim_id="C20",
            claim_type="formal/resource property",
            paper_facing_claim="A resource-blocked cell is outside predictive ordering and cannot be labeled worse than a completed cell.",
            scope="all benchmark cells",
            quantifier="all resource-blocked statuses",
            comparison="status versus measured performance",
            metric="feasibility status only",
            direction="unordered predictively",
            uncertainty_requirement="definition and audit validation",
            required_evidence="blocked records with zero outputs",
            matched_evidence_ids=blocked,
            prediction_requirement="absence recorded",
            pairing_or_statistical_requirement="no predictive test permitted",
            support_status="SUPPORTED",
            permitted_wording="unmeasured under the declared resource envelope",
            prohibited_wording="blocked model underperforms",
            rationale="Compute failure and predictive quality are distinct outcome types.",
            paper_location="Framework properties; resource feasibility",
        ),
        claim(
            claim_id="C21",
            claim_type="evidence integrity",
            paper_facing_claim="The primary IBM AML package contains 840 validated result files and 840 prediction exports across V26-V28, excluding blocked cells.",
            scope="V26, V27, V28 canonical imported locks",
            quantifier="exact file counts",
            comparison="expected versus actual",
            metric="artifact counts",
            direction="240+80+520",
            uncertainty_requirement="exact deterministic count",
            required_evidence="three canonical evidence locks and manifests",
            matched_evidence_ids=f"{v26};{v27};{v28}",
            prediction_requirement="required and complete",
            pairing_or_statistical_requirement="not applicable",
            support_status="SUPPORTED",
            permitted_wording="840/840 locked IBM AML result/prediction files",
            prohibited_wording="840 independent configurations; blocked outputs included",
            rationale="This is artifact scope, not a scientific contribution or performance claim.",
            paper_location="Artifact/reproducibility; supplement",
        ),
        claim(
            claim_id="C22",
            claim_type="framework validation",
            paper_facing_claim="The support validator blocks widened, incomplete, prediction-missing, integrity-failed, and resource-blocked claims without changing canonical evidence.",
            scope="sandbox mutations of the generated evidence inventory",
            quantifier="declared validation cases",
            comparison="complete claim versus mutated requirements/evidence",
            metric="support status and false-promotion count",
            direction="status downgrades as specified",
            uncertainty_requirement="deterministic mutation tests and hashes",
            required_evidence="generated validation cases and immutable source hashes",
            matched_evidence_ids="generated by scripts/tkde_rebuild/validate_support_relation.py",
            prediction_requirement="tested explicitly",
            pairing_or_statistical_requirement="tested explicitly",
            support_status="SUPPORTED" if framework_validated else "PENDING_FRAMEWORK_VALIDATION",
            permitted_wording="the validator rejects the tested incomplete/invalid cases",
            prohibited_wording="the gate proves all scientific claims true",
            rationale="This becomes supported only after deterministic mutation tests pass.",
            paper_location="Framework validation",
        ),
    ]
    return claims


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend("| " + " | ".join(clean(row[field]) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    inventory = load_inventory()
    claims = build_claims(inventory)
    with (OUT / "CLAIM_EVIDENCE_LEDGER.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(claims)
    statuses: dict[str, int] = {}
    for row in claims:
        statuses[row["support_status"]] = statuses.get(row["support_status"], 0) + 1
    status_lines = "\n".join(f"- `{key}`: {statuses[key]}" for key in sorted(statuses))
    table = markdown_table(
        claims,
        ["claim_id", "paper_facing_claim", "scope", "support_status", "permitted_wording", "prohibited_wording"],
    )
    (OUT / "CLAIM_EVIDENCE_LEDGER.md").write_text(
        f"""# TKDE Claim-Evidence Ledger

The ledger treats a claim as a typed object with scope, quantifier, comparison, metric, direction, uncertainty requirement, and deployment interpretation. `SUPPORTED` is always relative to the recorded scope. `RESOURCE_BLOCKED` means unmeasured, not poor performance. `REFUTED_IN_SCOPE` means at least one observed cell contradicts the claim's stated universal direction. `DIAGNOSTIC_ONLY` bars promotion to a general conclusion. The full machine-readable fields and matched evidence IDs are in the CSV.

## Status counts

{status_lines}

## Paper-facing view

{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {len(claims)} typed claims")


if __name__ == "__main__":
    main()
