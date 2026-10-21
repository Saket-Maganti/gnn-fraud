#!/usr/bin/env python3
"""Frozen-baseline object registry for the TKDE visual reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    object_type: str
    latex_label: str
    caption_title: str
    document: str
    source_tex: str
    generation_script: str
    source_data_files: str
    provenance_checksum_record: str
    current_pdf_page: str
    float_type: str
    orientation: str
    physical_width_in: str
    physical_height_in: str
    estimated_text_size_pt: str
    row_count: str
    column_count: str
    caption_word_count: int
    scientific_question: str
    claim_ids: str
    metrics: str
    seed_scope: str
    protocol_scope: str
    feasibility_set: str
    uncertainty: str
    redundancy: str
    visual_defects: str
    scientific_communication_defects: str
    page_utilization: str
    print_readability: str
    grayscale_readability: str
    final_disposition: str
    planned_replacement: str
    final_destination: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _spec(
    object_id: str,
    object_type: str,
    label: str,
    title: str,
    document: str,
    source_tex: str,
    *,
    page: str,
    float_type: str = "display",
    orientation: str = "portrait",
    width: str = "3.45",
    height: str = "auto",
    text_size: str = "10",
    rows: str = "n/a",
    cols: str = "n/a",
    question: str = "Formal definition used by the scientific argument.",
    claims: str = "",
    metrics: str = "n/a",
    seeds: str = "n/a",
    protocols: str = "all scoped contracts",
    feasibility: str = "n/a",
    uncertainty: str = "n/a",
    redundancy: str = "none",
    visual_defects: str = "none",
    communication_defects: str = "none",
    utilization: str = "embedded in prose",
    readability: str = "readable",
    grayscale: str = "not color dependent",
    disposition: str = "KEEP_AS_IS",
    replacement: str = "Retain with unchanged scientific notation.",
    destination: str | None = None,
    generator: str = "LaTeX source",
    data: str = "n/a",
    provenance: str = "results/tkde_rebuild/NUMBER_PROVENANCE_MAP.csv",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type=object_type,
        latex_label=label,
        caption_title=title,
        document=document,
        source_tex=source_tex,
        generation_script=generator,
        source_data_files=data,
        provenance_checksum_record=provenance,
        current_pdf_page=page,
        float_type=float_type,
        orientation=orientation,
        physical_width_in=width,
        physical_height_in=height,
        estimated_text_size_pt=text_size,
        row_count=rows,
        column_count=cols,
        caption_word_count=len(title.split()),
        scientific_question=question,
        claim_ids=claims,
        metrics=metrics,
        seed_scope=seeds,
        protocol_scope=protocols,
        feasibility_set=feasibility,
        uncertainty=uncertainty,
        redundancy=redundancy,
        visual_defects=visual_defects,
        scientific_communication_defects=communication_defects,
        page_utilization=utilization,
        print_readability=readability,
        grayscale_readability=grayscale,
        final_disposition=disposition,
        planned_replacement=replacement,
        final_destination=destination or document,
    )


def main_figures() -> list[ObjectSpec]:
    g = "scripts/tkde_rebuild/make_figures.py"
    p = "results/tkde_rebuild/FIGURE_DATA_PROVENANCE.csv"
    return [
        _spec("M-F01", "figure", "fig:deployment-contract", "Deployment-contract framework", "main", "paper_tkde/sections/04_deployment_claim_contracts.tex", page="4", float_type="figure*", width="7.16", height="3.05", text_size="5.7-8.5", question="How does a contract become typed evidence and a scoped claim?", claims="C18-C22", protocols="all", visual_defects="slide-like pastel boxes; dense prose; small status labels", communication_defects="administrative flow dominates quantitative paper", utilization="page 4: high", readability="requires close reading", grayscale="status palette loses distinctions", disposition="REDESIGN", replacement="Compact six-coordinate band plus minimal contract-to-evidence-to-claim flow and labeled status key.", generator=g, data="results/tkde_rebuild/DEPLOYMENT_CONTRACT_AXES.csv", provenance=p),
        _spec("M-F02", "figure", "fig:protocol-effects", "Protocol and architecture sensitivity", "main", "paper_tkde/sections/07_results_protocol_architecture.tex", page="8", float_type="figure*", width="7.16", height="3.00", text_size="6.2-8.2", question="How does strict-to-isolated visibility change AUPRC by model and dataset?", claims="C01-C04", metrics="AUPRC", seeds="1-10 paired", protocols="strict-inductive vs isolated-inductive", feasibility="complete 2 datasets x 3 models", uncertainty="deterministic 95% bootstrap CI", visual_defects="labels below V2 floor", grayscale="model lines remain distinguishable but color is helpful", disposition="KEEP_WITH_MINOR_EDIT", replacement="Retain forest/slope structure with 8 pt typography, redundant markers, and tighter caption.", generator=g, data="results/tkde_rebuild/RB09_AUPRC_MAIN.csv", provenance=p),
        _spec("M-F03", "figure", "fig:ibm-results", "IBM metric, scale, and construction grid", "main", "paper_tkde/sections/08_results_construction_decisions.tex", page="10", float_type="figure*", width="7.16", height="5.10", text_size="5.8-8.2", question="How do IBM configurations vary by metric, scale, regime, and protocol?", claims="C05-C14", metrics="AUPRC; AUROC; F1", seeds="1-10", protocols="early-to-late; late-window", feasibility="V27/V28; GINE Small only", uncertainty="deterministic 95% bootstrap CI", visual_defects="six crowded panels; small labels; color-dependent regimes", communication_defects="baseline and construction questions are conflated", readability="requires zoom", grayscale="regime/configuration distinctions weaken", disposition="REPLACE_WITH_DIFFERENT_VISUAL_FORM", replacement="Readable two-panel IBM baseline-family AUPRC comparison; construction effects move to the matched multi-metric figure/table.", generator=g, data="results/tkde_rebuild/IBM_CELL_SUMMARY.csv", provenance=p),
        _spec("M-F04", "figure", "fig:rank-divergence", "Rank-versus-decision divergence", "main", "paper_tkde/sections/08_results_construction_decisions.tex", page="11", float_type="figure*", width="7.16", height="3.20", text_size="6.3-8.2", question="When do AUPRC and fixed-threshold F1 rank the same feasible set differently?", claims="C11-C14", metrics="AUPRC rank; F1 rank; Spearman rho", seeds="means over seeds 1-10", protocols="IBM early-to-late; late-window", feasibility="exact baseline/graph grid per cell", uncertainty="descriptive ranks", visual_defects="many categorical colors", disposition="KEEP_WITH_MINOR_EDIT", replacement="Retain direct rank comparison with grayscale-first styling and repaired source schema.", generator=g, data="results/tkde_rebuild/IBM_RANK_DIVERGENCE.csv; results/tkde_rebuild/IBM_METRIC_RANKS.csv", provenance=p),
        _spec("M-F05", "figure", "fig:ablation-effects", "Matched IBM construction effects", "main", "paper_tkde/sections/08_results_construction_decisions.tex", page="11", float_type="figure*", width="7.16", height="2.85", text_size="6.8-8.2", question="Which matched constructions change IBM performance relative to h64?", claims="C08-C10; C13", metrics="AUPRC only", seeds="10 seed blocks; 4 contexts averaged within seed", protocols="both IBM protocols", feasibility="Small and Medium separated; GINE Small only", uncertainty="95% CI; Holm status", visual_defects="right title clipped; AUPRC-only view hides stated metric tradeoffs", disposition="REDESIGN", replacement="Two-panel multi-metric forest showing AUPRC, AUROC, and F1 deltas with significance and feasibility encoded redundantly.", generator=g, data="results/tkde_rebuild/IBM_MATCHED_ABLATION_EFFECTS.csv", provenance=p),
        _spec("M-F06", "figure", "fig:runtime-pareto", "Runtime and resource Pareto view", "main", "paper_tkde/sections/08_results_construction_decisions.tex", page="12", float_type="figure*", width="7.16", height="5.00", text_size="6.1-8.2", question="What performance-runtime tradeoffs are measured within compatible IBM cells?", claims="C10; C14; C20", metrics="AUPRC; elapsed seconds; Pareto status", seeds="1-10", protocols="IBM early-to-late; late-window", feasibility="variant-protocol matched; blocked cells nonnumeric", uncertainty="AUPRC SD; runtime min-max", visual_defects="large legend; crowded points; color-only configurations", disposition="REDESIGN", replacement="Grayscale-first four-cell Pareto view with only reference, GINE, Pareto, and blocked states emphasized.", generator=g, data="results/tkde_rebuild/IBM_RUNTIME_FEASIBILITY.csv", provenance=p),
        _spec("M-F07", "figure", "fig:claim-validation", "Claim-support validation", "main", "paper_tkde/sections/09_framework_validation_graphsafe.tex", page="13", float_type="figure*", width="7.16", height="3.00", text_size="6.0-8.2", question="Do controlled claim/evidence mutations produce the specified support statuses?", claims="C18-C22", metrics="14/14 status matches", seeds="case-specific", protocols="validator fixtures", feasibility="all 14 designed cases", uncertainty="none; conformance suite", visual_defects="administrative flowchart; hard-coded text; only subset shown", communication_defects="caption says 14 although diagram depicts selected cases", utilization="page 13: dominant", grayscale="pastel categories weaken", disposition="REPLACE_WITH_DIFFERENT_VISUAL_FORM", replacement="Data-driven expected-versus-observed status matrix plus compact mutation-family counts.", generator=g, data="results/tkde_rebuild/FRAMEWORK_VALIDATION_CASES.csv", provenance=p),
    ]


def main_tables() -> list[ObjectSpec]:
    g = "scripts/tkde_rebuild/build_tables.py"
    p = "results/tkde_rebuild/TABLE_DATA_PROVENANCE.csv"
    common = dict(document="main", float_type="table*", width="7.16", text_size="~7", generator=g, provenance=p, readability="small at print scale", grayscale="not color dependent")
    return [
        _spec("M-T01", "table", "tab:related-comparison", "Positioning against benchmark families", source_tex="paper_tkde/tables/table01_related_work.tex", page="4", rows="6", cols="8", question="Which novelty distinctions remain after related-work comparison?", claims="C18", data="code-authored; results/tkde_rebuild/LITERATURE_MATRIX.csv", visual_defects="eight dense text columns", communication_defects="multiword cells require small type", disposition="REDESIGN", replacement="Compact symbol matrix over temporal, graph-contract, capacity, resource, prediction, and executable-support distinctions.", **common),
        _spec("M-T02", "table", "tab:datasets", "Dataset and task summary", source_tex="paper_tkde/tables/table02_dataset_tasks.tex", page="6", rows="6", cols="9", question="Why are task units and values not pooled across datasets?", claims="C01-C14", metrics="counts; prevalence", seeds="n/a", protocols="all evaluated", feasibility="real Elliptic/DGraphFin; IBM Small/Medium; Large blocked", data="results/tkde_rebuild/DATASET_TASK_STATISTICS.csv", visual_defects="nine columns at scriptsize", disposition="REDESIGN", replacement="Readable task-card summary with unit, temporal extent, prior, graph/features, protocol, and empirical/resource scope.", **common),
        _spec("M-T03", "table", "tab:models", "Model and construction inventory", source_tex="paper_tkde/tables/table03_model_constructions.tex", page="6", rows="8", cols="4", question="What models and graph controls instantiate the benchmark?", claims="C01-C14", data="code-authored; results/tkde_rebuild/MODEL_CONSTRUCTION_INVENTORY.csv", visual_defects="dense prose cells", communication_defects="detail competes with central protocol table", disposition="MOVE_MAIN_TO_SUPPLEMENT", replacement="Role-specific model and construction cards in the supplement; replace main float with a protocol/visibility matrix.", destination="supplement", **common),
        _spec("M-T04", "table", "tab:rb09-effects", "Paired strict-to-isolated protocol effects", source_tex="paper_tkde/tables/table04_rb09_protocol_effects.tex", page="8", rows="6", cols="8", question="What are the paired AUPRC changes and corrected inference?", claims="C01-C04", metrics="AUPRC; absolute/relative delta; dz; Holm p", seeds="1-10 paired", protocols="strict vs isolated", feasibility="complete 2 x 3 grid", uncertainty="95% bootstrap CI; paired effect; Holm", data="results/tkde_rebuild/RB09_AUPRC_MAIN.csv", visual_defects="scriptsize", disposition="REDESIGN", replacement="Footnotesize aligned numeric table with the same six paired effects and concise note.", **common),
        _spec("M-T05", "table", "tab:ibm-results", "IBM baseline and graph-grid results", source_tex="paper_tkde/tables/table05_ibm_results.tex", page="9", rows="16", cols="7", question="Who leads the baseline grid and graph grid?", claims="C05-C14", metrics="AUPRC; AUROC; F1", seeds="1-10", protocols="two IBM protocols", feasibility="exact variant-protocol sets", uncertainty="mean plus SD", data="results/tkde_rebuild/IBM_CELL_SUMMARY.csv; results/tkde_rebuild/IBM_RANK_DIVERGENCE.csv", visual_defects="two dense panels; repeated winner columns", communication_defects="mixes baseline-family and graph-construction questions", disposition="SPLIT", replacement="Separate IBM baseline AUPRC table and graph-construction matched-effect table.", **common),
        _spec("M-T06", "table", "tab:resource-boundaries", "Resource boundaries", source_tex="paper_tkde/tables/table06_resource_boundaries.tex", page="9", rows="6", cols="5", question="Which cells are unmeasured and how are they treated?", claims="C10; C14; C20", metrics="resource status only", seeds="n=0 blocked", protocols="cell-specific", feasibility="guard/T4 OOM/waiting GPU", uncertainty="none", data="results/tkde_rebuild/RESOURCE_BOUNDARIES.csv", visual_defects="wide prose columns", communication_defects="file counts are administrative", disposition="REDESIGN", replacement="Four-column cell, envelope, short status tag, and benchmark-treatment table.", **common),
        _spec("M-T07", "table", "tab:graphsafe", "Bounded GraphSafe comparator", source_tex="paper_tkde/tables/table07_graphsafe_case.tex", page="10", float_type="table", width="3.45", rows="6", cols="5", question="How does bounded GraphSafe compare with strong saved-score baselines?", claims="C15-C17", metrics="F1; Recall@1%; cost risk", seeds="10 seed blocks", protocols="six contexts averaged within seed", feasibility="Elliptic and DGraphFin saved predictions", uncertainty="paired tests in prose/supplement", data="results/tkde_rebuild/GRAPHSAFE_BOUNDED_SUMMARY.csv", visual_defects="caption longer than table", disposition="KEEP_WITH_MINOR_EDIT", replacement="Footnotesize compact table with shorter caption and explicit comparator boundary.", document="main", generator=g, provenance=p),
    ]


def main_equations() -> list[ObjectSpec]:
    rows = [
        ("M-E01", "eq:eligibility", "Eligibility mask and binary target", "paper_tkde/sections/04_deployment_claim_contracts.tex", "3", "C01-C04"),
        ("M-E02", "eq:deployment-contract", "Six-coordinate deployment contract", "paper_tkde/sections/04_deployment_claim_contracts.tex", "3", "C18-C22"),
        ("M-E03", "eq:evidence-unit", "Typed evidence unit", "paper_tkde/sections/04_deployment_claim_contracts.tex", "3", "C18-C22"),
        ("M-E04", "eq:typed-claim", "Typed empirical claim", "paper_tkde/sections/04_deployment_claim_contracts.tex", "3", "C18-C22"),
        ("M-E05", "eq:sage", "GraphSAGE visibility-dependent update", "paper_tkde/sections/05_benchmark_instantiation.tex", "5", "C01-C04"),
        ("M-E06", "eq:cost-risk", "Illustrative 1:5 cost risk", "paper_tkde/sections/06_experimental_design.tex", "6", "C15-C17"),
    ]
    return [
        _spec(
            object_id=oid,
            object_type="equation",
            label=label,
            title=title,
            document="main",
            source_tex=source,
            page=page,
            claims=claims,
        )
        for oid, label, title, source, page, claims in rows
    ]


def supplement_figures() -> list[ObjectSpec]:
    base = [
        ("S-F01", "fig:s-contract", "Repeated contract framework", "paper_tkde/supplement/sections/02_contract_support.tex", "4", "fig01_deployment_contract.pdf", "REMOVE_REDUNDANT", "Use the redesigned main figure once or replace with a text cross-reference."),
        ("S-F02", "fig:s-protocol-effects", "Repeated protocol effects", "paper_tkde/supplement/sections/06_protocol_grid_results.tex", "11", "fig02_protocol_architecture_effects.pdf", "REMOVE_REDUNDANT", "Use readable aggregate effect tables and refer to the main figure."),
        ("S-F03", "fig:s-ibm-ablation", "Repeated matched IBM ablation", "paper_tkde/supplement/sections/07_ibm_results.tex", "20", "fig05_matched_ablation_effects.pdf", "REPLACE_WITH_DIFFERENT_VISUAL_FORM", "Add a compact context-direction/heterogeneity visual if it improves the aggregate tables."),
        ("S-F04", "fig:s-review-budget", "Review-budget curves", "paper_tkde/supplement/sections/08_graphsafe_case.tex", "42", "fig07_review_budget_analysis.pdf", "KEEP_WITH_MINOR_EDIT", "Retain with 8 pt typography and grayscale-redundant line styles/markers."),
        ("S-F05", "fig:s-framework-validation", "Repeated validation flowchart", "paper_tkde/supplement/sections/09_integrity_resources.tex", "43", "fig08_claim_support_validation.pdf", "REMOVE_REDUNDANT", "Use the complete controlled-case table; main contains the quantitative status matrix."),
    ]
    return [_spec(i, "figure", l, t, "supplement", s, page=p, float_type="figure", width="6.6", height="3-5", text_size="6-8", redundancy="duplicates main object" if i != "S-F04" else "unique diagnostic", visual_defects="small labels and full-width scaling", communication_defects="duplicate of main" if i != "S-F04" else "none", disposition=d, replacement=r, destination="artifact" if d == "REMOVE_REDUNDANT" else "supplement", generator="scripts/tkde_rebuild/make_figures.py", data=f"paper_tkde/figures/{f}", provenance="results/tkde_rebuild/FIGURE_DATA_PROVENANCE.csv") for i,l,t,s,p,f,d,r in base]


SUPPLEMENT_TABLES = [
    ("S-T01","tab:s-datasets","Dataset cards",7,"10","10","DATASET_TASK_STATISTICS.csv","SPLIT","Readable Elliptic, DGraphFin, and IBM dataset cards."),
    ("S-T02","tab:s-protocols","Protocol contracts",7,"5","6","PROTOCOL_DEFINITIONS.csv","REDESIGN","Explicit label/feature/graph visibility matrix including IBM 50%/60%."),
    ("S-T03","tab:s-models","Model and construction inventory",9,"13","4","MODEL_CONSTRUCTION_INVENTORY.csv","SPLIT","Role-specific baseline, node-GNN, edge-model, and construction cards."),
    ("S-T04","tab:s-training","Optimization and selection settings",9,"6","11","TRAINING_CONFIGURATION.csv","REDESIGN","Readable family-specific training and selection cards."),
    ("S-T05","tab:s-rb09-seed","Complete RB09 seed rows",12,"180","10","results/runs_rb09v3/runs.csv","MOVE_SUPPLEMENT_TO_ARTIFACT","Completeness count and schema excerpt; exhaustive 180 rows in artifact."),
    ("S-T06","tab:s-rb09-effects","Paired RB09 effects",14,"18","11","RB09_PROTOCOL_EFFECTS.csv","REDESIGN","Readable aggregate effects split by metric/dataset."),
    ("S-T07","tab:s-v24-duplicate","V24 duplicate-label audit",14,"12","6","V24_DUPLICATE_STRESS_AUDIT.csv","KEEP_WITH_MINOR_EDIT","Compact construct-audit table with 120/240 counts."),
    ("S-T08","tab:s-v22-lanes","V22 lane completeness",15,"6","7","manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv","KEEP_WITH_MINOR_EDIT","Readable lane completeness and blocked fixed-GAT row."),
    ("S-T09","tab:s-v22-stats","Complete V22 tests",16,"198","13","manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv","MOVE_SUPPLEMENT_TO_ARTIFACT","Hypothesis-family summary; all 198 rows in artifact."),
    ("S-T10","tab:s-ibm-seed","Complete IBM seed rows",21,"840","9","IBM_IMPORTED_SEED_ROWS.csv","MOVE_SUPPLEMENT_TO_ARTIFACT","Completeness/schema summary; all 840 rows in artifact."),
    ("S-T11","tab:s-ibm-cells","IBM cell aggregates",32,"84","10","IBM_CELL_SUMMARY.csv","SPLIT","Baseline, graph-grid, and scale-focused aggregate tables."),
    ("S-T12","tab:s-ranks","Rank/decision disagreement",33,"16","10","IBM_RANK_DIVERGENCE.csv","KEEP_WITH_MINOR_EDIT","Readable exact-feasibility rank summary."),
    ("S-T13","tab:s-ablation","Matched IBM construction effects",33,"52","12","IBM_MATCHED_ABLATION_EFFECTS.csv","REDESIGN","Multi-metric aggregate table with ten seed blocks and correction status."),
    ("S-T14","tab:s-ablation-contexts","Context-specific IBM effects",34,"208","12","IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv","MOVE_SUPPLEMENT_TO_ARTIFACT","Directional heterogeneity counts plus representative contexts; all 208 rows in artifact."),
    ("S-T15","tab:s-runtime","Runtime and feasibility",38,"58","10","IBM_RUNTIME_FEASIBILITY.csv","REDESIGN","Reference/Pareto/blocked resource summary by scale and protocol."),
    ("S-T16","tab:s-graphsafe","GraphSafe aggregate summary",40,"10","8","GRAPHSAFE_BOUNDED_SUMMARY.csv","KEEP_WITH_MINOR_EDIT","Comparator-focused aggregate table."),
    ("S-T17","tab:s-graphsafe-tests","GraphSafe paired tests",40,"48","9","GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE.csv","REDESIGN","Selected decision metrics plus 48-test family summary."),
    ("S-T18","tab:s-review-budget","Review-budget results",40,"30","7","REVIEW_BUDGET_CURVES.csv","REDESIGN","Focused 1% table plus curves for 0.5/1/2%."),
    ("S-T19","tab:s-resources","Resource cases",42,"6","5","RESOURCE_BOUNDARIES.csv","SPLIT","Readable case-study cards with evidence/no-evidence and safe wording."),
    ("S-T20","tab:s-claims","Typed claim ledger",5,"22","5","CLAIM_EVIDENCE_LEDGER.csv","SPLIT","Five thematic claim tables with concise permitted/prohibited quantifiers."),
    ("S-T21","tab:s-families","Evidence family and lock map",45,"15","5","EVIDENCE_INVENTORY.csv","MOVE_SUPPLEMENT_TO_ARTIFACT","Concise provenance map; exhaustive paths/checksums in artifact."),
    ("S-T22","tab:s-framework","Controlled validation cases",44,"14","7","FRAMEWORK_VALIDATION_CASES.csv","REDESIGN","Readable mutation family, violated rule, expected/observed status, pass table."),
    ("S-T23","tab:s-false-promotion","False-promotion audit",44,"4","7","FALSE_PROMOTION_AUDIT.csv","KEEP_WITH_MINOR_EDIT","Compact category/count/correct-treatment table."),
]


def supplement_tables() -> list[ObjectSpec]:
    out: list[ObjectSpec] = []
    source_by_label = {
        "tab:s-datasets": "table_s01_dataset_cards.tex",
        "tab:s-protocols": "table_s02_protocols.tex",
        "tab:s-models": "table_s03_models.tex",
        "tab:s-training": "table_s04_training.tex",
        "tab:s-rb09-seed": "table_s05_rb09_seed.tex",
        "tab:s-rb09-effects": "table_s06_rb09_effects.tex",
        "tab:s-v24-duplicate": "table_s07_v24_duplicate_audit.tex",
        "tab:s-v22-lanes": "table_s08_v22_lanes.tex",
        "tab:s-v22-stats": "table_s09_v22_stats.tex",
        "tab:s-ibm-seed": "table_s10_ibm_seed.tex",
        "tab:s-ibm-cells": "table_s11_ibm_cells.tex",
        "tab:s-ranks": "table_s12_ranks.tex",
        "tab:s-ablation": "table_s13_ablation.tex",
        "tab:s-ablation-contexts": "table_s13b_ablation_contexts.tex",
        "tab:s-runtime": "table_s14_runtime.tex",
        "tab:s-graphsafe": "table_s15_graphsafe.tex",
        "tab:s-graphsafe-tests": "table_s16_graphsafe_tests.tex",
        "tab:s-review-budget": "table_s17_review_budget.tex",
        "tab:s-resources": "table_s18_resources.tex",
        "tab:s-claims": "table_s19_claim_ledger.tex",
        "tab:s-families": "table_s20_evidence_families.tex",
        "tab:s-framework": "table_s21_framework_validation.tex",
        "tab:s-false-promotion": "table_s22_false_promotion.tex",
    }
    for oid,label,title,page,rows,cols,data,disp,repl in SUPPLEMENT_TABLES:
        destination = "artifact" if disp == "MOVE_SUPPLEMENT_TO_ARTIFACT" else "supplement"
        out.append(_spec(oid,"longtable",label,title,"supplement",f"paper_tkde/supplement/tables/{source_by_label[label]}",page=str(page),float_type="longtable",orientation="landscape" if page in {9,12,14,15,16,21,32,33,34,38,40,44,45} else "portrait",width="9.0 landscape" if page in {9,12,14,15,16,21,32,33,34,38,40,44,45} else "6.6",height="multi-page" if int(rows) > 60 else "auto",text_size="~5-7",rows=rows,cols=cols,question=title,claims="see CLAIM_EVIDENCE_LEDGER.csv",metrics="table-specific",seeds="1-10 where empirical",protocols="table-specific",feasibility="explicit in source",uncertainty="table-specific",redundancy="raw CSV duplication" if disp == "MOVE_SUPPLEMENT_TO_ARTIFACT" else "none",visual_defects="tiny type; landscape/raw-row density",communication_defects="machine-readable detail dominates" if disp == "MOVE_SUPPLEMENT_TO_ARTIFACT" else "insufficient narrative grouping",utilization=f"baseline page {page}; often sparse landscape",readability="microscopic or zoom-dependent",grayscale="not color dependent",disposition=disp,replacement=repl,destination=destination,generator="scripts/tkde_rebuild/build_tables.py",data=f"results/tkde_rebuild/{data}" if "/" not in data else data,provenance="results/tkde_rebuild/TABLE_DATA_PROVENANCE.csv"))
    return out


def supplement_equations() -> list[ObjectSpec]:
    rows = [
        ("S-E01","Raw label semantics","01_scope_notation.tex","3"),("S-E02","Eligibility and binary target","01_scope_notation.tex","3"),("S-E03","Scientific cell key","01_scope_notation.tex","3"),
        ("S-E04","Deployment contract","02_contract_support.tex","3"),("S-E05","Evidence unit","02_contract_support.tex","4"),("S-E06","Typed claim","02_contract_support.tex","4"),("S-E07","Requirement monotonicity","02_contract_support.tex","4"),
        ("S-E08","GCN update","04_models_constructions.tex","7"),("S-E09","GraphSAGE update","04_models_constructions.tex","7"),("S-E10","IBM neighborhood summary","04_models_constructions.tex","7"),("S-E11","IBM edge score","04_models_constructions.tex","7"),("S-E12","GINE update","04_models_constructions.tex","7"),
        ("S-E13","Normalized AUPRC diagnostic","05_metrics_statistics.tex","10"),("S-E14","Precision, recall, and F1","05_metrics_statistics.tex","10"),("S-E15","Illustrative cost risk","05_metrics_statistics.tex","10"),("S-E16","Brier score","05_metrics_statistics.tex","10"),("S-E17","Paired Cohen dz","05_metrics_statistics.tex","10"),
        ("S-E18","GraphSafe reliability risk","08_graphsafe_case.tex","39"),("S-E19","GraphSafe switching score","08_graphsafe_case.tex","39"),
    ]
    return [_spec(oid,"equation",f"unlabeled:{oid}",title,"supplement",f"paper_tkde/supplement/sections/{src}",page=page,question=f"Define {title.lower()} without changing the frozen scientific specification.",disposition="KEEP_AS_IS",replacement="Retain in the corresponding curated technical section.") for oid,title,src,page in rows]


def landscape_blocks() -> list[ObjectSpec]:
    rows = [
        ("S-L01","Model and training cards","paper_tkde/supplement/sections/04_models_constructions.tex","9","S-T03; S-T04"),
        ("S-L02","Node protocol raw tables","paper_tkde/supplement/sections/06_protocol_grid_results.tex","12-18","S-T05-S-T09"),
        ("S-L03","IBM raw and aggregate tables","paper_tkde/supplement/sections/07_ibm_results.tex","21-38","S-T10-S-T15"),
        ("S-L04","GraphSafe tables","paper_tkde/supplement/sections/08_graphsafe_case.tex","40-41","S-T16-S-T18"),
        ("S-L05","Validator tables","paper_tkde/supplement/sections/09_integrity_resources.tex","44","S-T22; S-T23"),
    ]
    return [_spec(oid,"landscape_block",f"unlabeled:{oid}",title,"supplement",src,page=page,float_type="landscape",orientation="landscape",width="9.0",height="multi-page",text_size="~5-7",rows="multiple",cols="multiple",question="Does landscape orientation materially improve readable scientific comparison?",redundancy="wrapper only",visual_defects="forced rotation and sparse pages",communication_defects="breaks narrative flow",utilization="often below 50%",readability="tables remain microscopic",grayscale="not color dependent",disposition="REDESIGN",replacement="Remove the wrapper; use portrait footnotesize curated tables and natural pagination.") for oid,title,src,page,_ in rows]


def all_objects() -> list[ObjectSpec]:
    objects = main_figures()+main_tables()+main_equations()+supplement_figures()+supplement_tables()+supplement_equations()+landscape_blocks()
    assert len(objects) == 72, len(objects)
    assert len({obj.object_id for obj in objects}) == len(objects)
    return objects


def rows(objects: Iterable[ObjectSpec] | None = None) -> list[dict[str, object]]:
    return [obj.as_row() for obj in (objects or all_objects())]
