#!/usr/bin/env python3
"""
Cross-check claim gate documents against RUNS results index.

Training-free audit only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.audit_utils import (  # noqa: E402
    second_dataset_results_exist,
    utc_now,
    write_json,
    write_text,
)


DOC_PATHS = {
    "policy": "runs_expansion/CLAIM_GATE_POLICY.md",
    "evidence_map": "gnnpaper/claim_evidence_map.md",
    "runs_paper_status": "runs_paper/CLAIM_STATUS.md",
}
INDEX_PATH = "results/runs/RUNS_RESULTS_INDEX.json"

PHRASE_SCAN_PATHS = (
    "README.md",
    "runs_paper",
    "gnnpaper",
    "aaai_upgrade",
)

UNSUPPORTED_SECOND_DATASET_PATTERNS = (
    re.compile(r"\bDGraphFin\s+results?\s+(show|shows|demonstrate|demonstrates|prove|proves|confirm|confirms|support|supports)\b", re.IGNORECASE),
    re.compile(r"\bT-?Finance\s+results?\s+(show|shows|demonstrate|demonstrates|prove|proves|confirm|confirms|support|supports|improve|improves)\b", re.IGNORECASE),
    re.compile(r"\bDGraphFin\b.{0,80}\b(improve|improves|outperform|outperforms|wins?|beats?)\b", re.IGNORECASE),
    re.compile(r"\bT-?Finance\b.{0,80}\b(improve|improves|outperform|outperforms|wins?|beats?)\b", re.IGNORECASE),
)

UNSUPPORTED_GENERAL_CLAIM_PATTERNS = (
    {
        "pattern": re.compile(r"\bcausal\s+graph\s+harm\b", re.IGNORECASE),
        "message": "Causal graph-harm language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bwe\s+prove\s+graph\s+causes\s+harm\b", re.IGNORECASE),
        "message": "Graph-harm evidence is diagnostic/counterfactual, not causal proof.",
    },
    {
        "pattern": re.compile(r"\bcounterfactual\s+results\s+show\b", re.IGNORECASE),
        "message": "Counterfactual result language is blocked until graph intervention result files exist.",
        "requires_counterfactual_results": True,
    },
    {
        "pattern": re.compile(r"\bDGraphFin\s+confirms\b", re.IGNORECASE),
        "message": "DGraphFin confirmation language is blocked without RB09 evidence.",
        "requires_second_dataset_results": True,
    },
    {
        "pattern": re.compile(r"\bT-?Finance\s+confirms\b", re.IGNORECASE),
        "message": "T-Finance confirmation language is blocked without RB09 evidence.",
        "requires_second_dataset_results": True,
    },
    {
        "pattern": re.compile(r"\bmethod\s+universally\s+improves\b", re.IGNORECASE),
        "message": "Universal method-improvement language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bHolm[- ]significant\s+method\s+improvement\b", re.IGNORECASE),
        "message": "Holm-significant method-improvement language is unsupported without a Holm-surviving row.",
        "requires_tpc_tta_holm": True,
    },
    {
        "pattern": re.compile(r"\bTPC[/+ ]?TTA\s+fix(?:es)?\s+the\s+problem\b", re.IGNORECASE),
        "message": "Blocked unconditionally: 'TPC/TTA fixes the problem' is an over-claim; the corrected wins are narrow (threshold-level F1 on the transductive arm), not a general fix.",
    },
    {
        "pattern": re.compile(r"\bgraph[- ]harm\s+is\s+causal\b", re.IGNORECASE),
        "message": "Graph-harm evidence is diagnostic, not causal.",
    },
    {
        "pattern": re.compile(r"\bcross[- ]dataset\s+results?\s+show\b", re.IGNORECASE),
        "message": "Cross-dataset result language is blocked without RB09 evidence.",
    },
    {
        "pattern": re.compile(r"\bDGraphFin\s+improves\b", re.IGNORECASE),
        "message": "DGraphFin improvement language is blocked without real RB09 outputs.",
    },
    {
        "pattern": re.compile(r"\bT-?Finance\s+confirms\b", re.IGNORECASE),
        "message": "T-Finance confirmation language is blocked without real RB09 outputs.",
    },
    {
        "pattern": re.compile(r"\bGAT\s+completed\s+on\s+DGraphFin\b", re.IGNORECASE),
        "message": "GAT-on-DGraphFin completion language is blocked without real result rows.",
        "requires_dgraphfin_gat_completed": True,
    },
    {
        "pattern": re.compile(r"\bscalable\s+to\s+DGraphFin\s+without\s+qualification\b", re.IGNORECASE),
        "message": "Unqualified DGraphFin scalability language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bgit\s+commit\s+verified\b", re.IGNORECASE),
        "message": "Imported empty git_commit fields must not be treated as verified commits.",
    },
    {
        "pattern": re.compile(r"\ball\s+protocols\s+generalize\b", re.IGNORECASE),
        "message": "Universal protocol-generalization language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bproves\s+causality\b", re.IGNORECASE),
        "message": "Causality-proof language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bsolves\s+temporal\s+fraud\b", re.IGNORECASE),
        "message": "Temporal-fraud solution language is unsupported.",
    },
    {
        "pattern": re.compile(r"\buniversally\s+robust\b", re.IGNORECASE),
        "message": "Universal robustness language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bgeneralizes\s+to\s+DGraphFin/T-?Finance\b", re.IGNORECASE),
        "message": "DGraphFin/T-Finance generalization language is blocked without RB09 evidence.",
        "requires_second_dataset_results": True,
    },
    {
        "pattern": re.compile(r"\bRB09\s+confirms\b", re.IGNORECASE),
        "message": "RB09 confirmation language is blocked without imported RB09 evidence.",
        "requires_second_dataset_results": True,
    },
    # --- RB09v3 method-evidence gates (added Jun 2026) ---
    {
        "pattern": re.compile(r"\bTPC[\s/+-]*TTA\s+(?:fix|repair|solv|address|undo|revers|recover)\w*\s+(?:the\s+)?(?:graph[\s-]*)?structure[\s-]*(?:decay|loss)", re.IGNORECASE),
        "message": "Blocked: rank metrics (AUPRC/AUROC) are invariant to calibration, so TPC+TTA cannot fix structure-decay.",
    },
    {
        "pattern": re.compile(r"\bTPC[\s/+-]*TTA\s+(?:fix|repair|solv|address|undo|revers)\w*\s+(?:the\s+)?representation[\s-]*shift", re.IGNORECASE),
        "message": "Blocked: TPC+TTA does not repair representation shift (Elliptic transductive ECE not improved).",
    },
    {
        "pattern": re.compile(r"\bTPC[\s/+-]*TTA\s+(?:solv\w*|eliminat\w*|fully\s+fix\w*)\s+(?:the\s+)?protocol[\s-]*(?:shift|gap|damage|reversal)", re.IGNORECASE),
        "message": "Blocked: only the threshold/decision component is recovered; the protocol shift is not solved/eliminated.",
    },
    {
        "pattern": re.compile(r"\brank(?:ing)?s?\s+(?:are\s+|is\s+|fully\s+|completely\s+)?restored\b", re.IGNORECASE),
        "message": "Blocked: ranking-restoration is a 3-model Kendall-tau diagnostic, Elliptic-only and weak; use 'partial diagnostic rank recovery'.",
    },
    {
        "pattern": re.compile(r"\b(?:harm[\s-]*aware\s+)?gat(?:e|ing)\s+(?:is\s+|proves?\s+|provides?\s+|establish\w*\s+)?caus\w*", re.IGNORECASE),
        "message": "Blocked: harm-aware gate is a diagnostic, not a causal mechanism.",
    },
    {
        "pattern": re.compile(r"\bgat(?:e|ing)\s+(?:always\s+|universally\s+)?(?:improv|help|win)\w*\s+universal|\buniversal\w*\s+gat(?:e|ing)\s+(?:improv|win)", re.IGNORECASE),
        "message": "Blocked: harm-aware gate does not improve universally (untrained combiners hurt on most arms).",
    },
    {
        "pattern": re.compile(r"\bmethod\s+wins?\s+on\s+DGraphFin\b", re.IGNORECASE),
        "message": "Blocked: DGraphFin method-win language must cite the separate DGraphFin Holm-corrected row, not Elliptic-only.",
    },
    {
        "pattern": re.compile(r"\bstress\s+tests\s+are\s+real\s+training\s+results\b", re.IGNORECASE),
        "message": "Stress-test outputs are simulations, not training results.",
    },
    {
        "pattern": re.compile(r"\bsimulated\s+shift\s+proves\s+deployment\s+performance\b", re.IGNORECASE),
        "message": "Simulated-shift deployment proof language is unsupported.",
    },
    {
        "pattern": re.compile(r"\ball\s+graph\s+harm\s+is\s+predictable\b", re.IGNORECASE),
        "message": "Universal graph-harm predictability language is unsupported.",
    },
    {
        "pattern": re.compile(r"\ball\s+protocols\s+are\s+covered\b", re.IGNORECASE),
        "message": "Universal protocol-coverage language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bdeployable\s+oracle\b|\boracle\s+is\s+deployable\b", re.IGNORECASE),
        "message": "Oracle analyses are upper-bound diagnostics, not deployable methods.",
    },
    {
        "pattern": re.compile(r"\bensemble\s+universally\s+improves\b|\buniversal\s+ensemble\s+improvement\b", re.IGNORECASE),
        "message": "Universal ensemble-improvement language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bconformal\s+guarantees\s+deployment\s+performance\b", re.IGNORECASE),
        "message": "Conformal/selective analyses do not guarantee deployment performance.",
    },
    {
        "pattern": re.compile(r"\bcalibration\s+fixes\s+all\s+drift\b", re.IGNORECASE),
        "message": "Calibration cannot be claimed to fix all drift.",
    },
    {
        "pattern": re.compile(r"\bgraph[- ]harm\s+predictor\s+proves\s+causality\b", re.IGNORECASE),
        "message": "Graph-harm predictor evidence is diagnostic, not causal.",
    },
    {
        "pattern": re.compile(r"\bpartial\s+RB09\b.{0,80}\bfull\s+cross[- ]dataset\s+support\b", re.IGNORECASE),
        "message": "Partial RB09 evidence cannot be described as full cross-dataset support.",
        "requires_second_dataset_results": True,
    },
    {
        "pattern": re.compile(r"\bGAT\s+completed\b.{0,80}\bOOM\s+evidence\b|\bGAT\s+completed\s+if\s+only\s+OOM\s+evidence\s+exists\b", re.IGNORECASE),
        "message": "GAT completion cannot be claimed from OOM/resource evidence.",
        "requires_dgraphfin_gat_completed": True,
    },
    # --- AAAI_GPU_EXPANSION gates (engineering package; no claims before import) ---
    {
        "pattern": re.compile(r"\bcross[- ]dataset\s+SAGE[- ]family\s+support\b", re.IGNORECASE),
        "message": "Cross-dataset SAGE-family support is blocked until RB11b is imported and verified.",
        "requires_aaai_gpu_family_verified": "RB11b",
    },
    {
        "pattern": re.compile(r"\bDGraphFin\s+sage[_ -]?maxpool\s+(?:confirms|validates|establishes|proves)\b", re.IGNORECASE),
        "message": "DGraphFin sage_maxpool confirmation language is blocked until RB11b is verified.",
        "requires_aaai_gpu_family_verified": "RB11b",
    },
    {
        "pattern": re.compile(r"\bRB11b\s+(?:confirms|validates|establishes|proves)\b", re.IGNORECASE),
        "message": "RB11b confirmation language is blocked until RB11b is verified.",
        "requires_aaai_gpu_family_verified": "RB11b",
    },
    {
        "pattern": re.compile(r"\bRB12\s+(?:architecture\s+expansion\s+)?(?:confirms|validates|establishes|proves|(?:is|are)\s+verified)\b", re.IGNORECASE),
        "message": "RB12 architecture-expansion claims are blocked until RB12 is verified.",
        "requires_aaai_gpu_family_verified": "RB12",
    },
    {
        "pattern": re.compile(r"\bRB13\s+(?:protocol\s+stress[- ]tests?\s+)?(?:confirms|validates|establishes|proves|(?:is|are)\s+verified)\b", re.IGNORECASE),
        "message": "RB13 stress-test claims are blocked until RB13 is verified.",
        "requires_aaai_gpu_family_verified": "RB13",
    },
    {
        "pattern": re.compile(r"\bnotebooks?\s+(?:are|is)\s+(?:empirical\s+)?evidence\b|\bgenerated\s+notebooks?\s+(?:show|prove|confirm)\b", re.IGNORECASE),
        "message": "Generated Kaggle notebooks are readiness artifacts, not empirical evidence.",
    },
    {
        "pattern": re.compile(r"\bREADY_TO_RUN\s+(?:is|means|equals)\s+VERIFIED\b", re.IGNORECASE),
        "message": "READY_TO_RUN/SMOKE_PENDING must not be treated as VERIFIED.",
    },
    {
        "pattern": re.compile(r"\bmethod\s+(?:solves|fixes|repairs)\s+protocol\s+shift\b", re.IGNORECASE),
        "message": "Method contribution is bounded decision-level mitigation, not a protocol-shift solution.",
    },
    {
        "pattern": re.compile(r"\bmethod\s+(?:repairs|fixes|solves)\s+(?:graph[- ]?)?structure[- ]decay\b", re.IGNORECASE),
        "message": "Method cannot be claimed to repair rank-level graph-structure decay.",
    },
    {
        "pattern": re.compile(r"\bmethod\s+(?:repairs|fixes|solves)\s+rank[- ]level\s+decay\b", re.IGNORECASE),
        "message": "Method cannot be claimed to repair rank-level decay.",
    },
    {
        "pattern": re.compile(r"\b(?:submission|paper|method|result|evidence|narrative|framing|work)\b.{0,40}\b(?:is|are|as|now|becomes?|clearly|only)\s+AAAI[- ]safe\b|\bAAAI[- ]safe\s+(?:to\s+submit|submission|result|evidence|claim)\b", re.IGNORECASE),
        "message": "AAAI-safe wording is blocked unless the final decision report explicitly allows it; use bounded recommendation language instead.",
    },
    {
        "pattern": re.compile(r"\bpost[- ]hoc\s+graph[- ]harm\s+predictor\s+is\s+pre[- ]training\b", re.IGNORECASE),
        "message": "Post-hoc graph-harm predictors must not be described as pre-training predictors.",
    },
    # --- RB15 GraphSafe-TTA claim gates ---
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+(?:solves?|eliminates?|fully\s+fixes?)\s+protocol\s+shift\b", re.IGNORECASE),
        "message": "GraphSafe-TTA is a decision-level deployment wrapper, not a protocol-shift solution.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+(?:fixes?|repairs?|solves?)\s+(?:graph[- ]?)?structure[- ]decay\b", re.IGNORECASE),
        "message": "GraphSafe-TTA must not be claimed to repair rank-level graph-structure decay.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+improves?\s+rank[- ]level\s+metrics\b", re.IGNORECASE),
        "message": "GraphSafe-TTA rank-level improvement language is blocked unless an exact rank-level table supports it.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+(?:universally\s+)?improves?\s+all\s+(?:datasets|models|datasets/models)\b", re.IGNORECASE),
        "message": "Universal GraphSafe-TTA improvement language is unsupported.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+is\s+causal\b", re.IGNORECASE),
        "message": "GraphSafe-TTA is not causal evidence.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+beats?\s+all\s+baselines\b", re.IGNORECASE),
        "message": "GraphSafe-TTA beats-all-baselines language is blocked without corrected evidence.",
    },
    {
        "pattern": re.compile(r"\bgating\s+improves?\s+robustness\b", re.IGNORECASE),
        "message": "Gating-improves-robustness language requires worst-block/regret support.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+consistently\s+improves?\s+across\s+datasets\b", re.IGNORECASE),
        "message": "Consistent-across-datasets GraphSafe-TTA improvement language is blocked unless the acceptance table supports that exact claim.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+beats?\s+(?:the\s+)?best\s+branch\b", re.IGNORECASE),
        "message": "Unqualified GraphSafe-TTA beats-best-branch language is blocked unless corrected acceptance evidence supports it.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+reduces?\s+worst[- ]block\s+regret\s+across\s+datasets\b", re.IGNORECASE),
        "message": "Across-dataset worst-block-regret language must be scoped to acceptance-table support.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+improves?\s+review[- ]budget\s+metrics\s+across\s+datasets\b", re.IGNORECASE),
        "message": "Across-dataset review-budget improvement language must be scoped to RB17 corrected support.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+reduces?\s+deployment\s+risk\s+under\s+review[- ]budget\s+and\s+worst[- ]block\s+metrics\b", re.IGNORECASE),
        "message": "Deployment-risk reduction language must cite exact RB17 dataset/protocol/model support.",
    },
    {
        "pattern": re.compile(r"\bGraphSafe[- ]TTA\s+(?:fixes?|repairs?|solves?)\s+rank[- ]level\s+decay\b", re.IGNORECASE),
        "message": "GraphSafe-TTA cannot be claimed to repair rank-level decay.",
    },
)

BLOCKING_CONTEXT_TOKENS = (
    "pending",
    "blocked",
    "not claimed",
    "not ",
    "no ",
    "no verified",
    "no dgraphfin",
    "no t-finance",
    "no tfinance",
    "do not",
    "does not",
    "cannot",
    "never",
    "future work",
    "scaffold",
    "unless",
    "until",
    "if ",
    "rank metrics are invariant",
)

REQUIRED_PENDING = (
    "validation_clean",
    "matched_gnn",
    "multi_dataset",
    "dgraphfin",
    "tfinance",
    "mitigation",
    "tpc",
)


@dataclass
class GateIssue:
    level: str
    message: str


@dataclass
class GateReport:
    issues: List[GateIssue] = field(default_factory=list)
    doc_gates: Dict[str, List[str]] = field(default_factory=dict)
    index_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)


def extract_gates(text: str) -> List[str]:
    return re.findall(r"\b(Verified|Partial|Pending|Unsafe)\b", text)


def load_index(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def check_docs(report: GateReport) -> None:
    for name, rel in DOC_PATHS.items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            report.issues.append(GateIssue("error", f"Missing doc: {rel}"))
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        gates = extract_gates(text)
        report.doc_gates[name] = gates
        lower = text.lower()
        for token in REQUIRED_PENDING:
            if token.replace("_", " ") not in lower and token not in lower:
                report.issues.append(
                    GateIssue("warning", f"{rel} does not mention pending topic: {token}")
                )


def check_index(report: GateReport) -> None:
    path = os.path.join(REPO_ROOT, INDEX_PATH)
    index = load_index(path)
    if index is None:
        report.issues.append(GateIssue("warning", f"Missing index: {INDEX_PATH}"))
        return
    report.index_summary = {
        "n_artifacts": len(index.get("artifacts", [])),
        "n_scanned_files": len(index.get("scanned_files", [])),
    }
    for art in index.get("artifacts", []):
        rel_path = art.get("path", "")
        gate = art.get("claim_gate", "")
        exists = art.get("exists", False)
        if not exists and gate.lower().startswith("verified") and "metadata" not in gate.lower():
            report.issues.append(
                GateIssue(
                    "error",
                    f"Index marks Verified without artifact: {rel_path} ({gate})",
                )
            )
        if "dgraphfin" in rel_path or "tfinance" in rel_path:
            if exists and "verified" in gate.lower() and "metadata" not in gate.lower():
                report.issues.append(
                    GateIssue("error", f"Blocked dataset marked verified in index: {rel_path}")
                )


def _iter_phrase_scan_files() -> List[str]:
    found: List[str] = []
    for rel in PHRASE_SCAN_PATHS:
        path = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(path):
            found.append(path)
            continue
        if not os.path.isdir(path):
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in {"__pycache__", "professor_phase1_packet", "phase1_professor_tables"}
            ]
            for name in filenames:
                if name.startswith(".") or not name.lower().endswith((".md", ".tex", ".txt")):
                    continue
                found.append(os.path.join(dirpath, name))
    return sorted(set(found))


def unsupported_second_dataset_phrase_hits(text: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(token in lower for token in BLOCKING_CONTEXT_TOKENS):
            continue
        for pattern in UNSUPPORTED_SECOND_DATASET_PATTERNS:
            match = pattern.search(line)
            if match:
                hits.append({"line": lineno, "text": line.strip(), "pattern": pattern.pattern})
                break
    return hits


def _tpc_tta_holm_supported(root: Path) -> bool:
    path = root / "results" / "runs" / "method_win_search" / "corrected_method_wins.csv"
    if not path.is_file():
        return False
    try:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("method", "")).strip() == "tpc_tta" and str(row.get("survives_holm", "")).lower() == "true":
                    return True
    except OSError:
        return False
    return False


def _graph_counterfactual_results_exist(root: Path) -> bool:
    result_dir = root / "results" / "runs" / "graph_counterfactuals"
    if not result_dir.is_dir():
        return False
    result_names = {
        "counterfactual_results.csv",
        "graph_counterfactual_results.csv",
        "intervention_results.csv",
        "results.csv",
    }
    return any((result_dir / name).is_file() for name in result_names)


def _dgraphfin_gat_completed(root: Path) -> bool:
    candidate_csvs = [
        root / "results" / "runs" / "RUN_DATABASE.csv",
        root / "results" / "runs" / "multi_dataset_protocol" / "runs.csv",
    ]
    for path in candidate_csvs:
        if not path.is_file():
            continue
        try:
            import csv

            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    dataset = str(row.get("dataset", "")).strip().lower()
                    model = str(row.get("model", "")).strip().lower()
                    status = str(row.get("status", "")).strip().lower()
                    f1 = str(row.get("f1", "")).strip()
                    if dataset == "dgraphfin" and model == "gat" and (f1 or status in {"computed", "verified"}):
                        return True
        except OSError:
            continue
    return False


def _aaai_gpu_family_verified(root: Path, family_key: str) -> bool:
    path = root / "results" / "runs" / "aaai_gpu_expansion_status.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    families = payload.get("families", {})
    if not isinstance(families, dict):
        return False
    item = families.get(family_key, {})
    if not isinstance(item, dict):
        return False
    return str(item.get("status", "")).strip().upper() == "VERIFIED"


def unsupported_general_phrase_hits(text: str, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    scan_root = root or Path(REPO_ROOT)
    tpc_supported = _tpc_tta_holm_supported(scan_root)
    counterfactual_supported = _graph_counterfactual_results_exist(scan_root)
    second_dataset_supported = second_dataset_results_exist(scan_root)
    dgraphfin_gat_supported = _dgraphfin_gat_completed(scan_root)
    aaai_gpu_verified = {
        key: _aaai_gpu_family_verified(scan_root, key)
        for key in ("RB11b", "RB12", "RB13", "RB14")
    }
    for lineno, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        blocking_tokens = [token for token in BLOCKING_CONTEXT_TOKENS if token != "if "]
        if any(token in lower for token in blocking_tokens):
            continue
        if "if " in lower and "if only oom evidence exists" not in lower:
            continue
        for spec in UNSUPPORTED_GENERAL_CLAIM_PATTERNS:
            if spec.get("requires_tpc_tta_holm") and tpc_supported:
                continue
            if spec.get("requires_counterfactual_results") and counterfactual_supported:
                continue
            if spec.get("requires_second_dataset_results") and second_dataset_supported:
                continue
            if spec.get("requires_dgraphfin_gat_completed") and dgraphfin_gat_supported:
                continue
            family_key = spec.get("requires_aaai_gpu_family_verified")
            if family_key and aaai_gpu_verified.get(str(family_key), False):
                continue
            pattern = spec["pattern"]
            match = pattern.search(line)
            if match:
                hits.append(
                    {
                        "line": lineno,
                        "text": line.strip(),
                        "pattern": pattern.pattern,
                        "message": spec["message"],
                    }
                )
                break
    return hits


def check_second_dataset_phrases(report: GateReport) -> None:
    if second_dataset_results_exist(Path(REPO_ROOT)):
        return
    for path in _iter_phrase_scan_files():
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        for hit in unsupported_second_dataset_phrase_hits(text):
            rel_path = os.path.relpath(path, REPO_ROOT)
            report.issues.append(
                GateIssue(
                    "error",
                    f"Unsupported second-dataset result phrase without RB09 evidence: {rel_path}:{hit['line']}: {hit['text']}",
                )
            )


def check_general_phrases(report: GateReport) -> None:
    root = Path(REPO_ROOT)
    for path in _iter_phrase_scan_files():
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        for hit in unsupported_general_phrase_hits(text, root):
            rel_path = os.path.relpath(path, REPO_ROOT)
            report.issues.append(
                GateIssue(
                    "error",
                    f"{hit['message']} {rel_path}:{hit['line']}: {hit['text']}",
                )
            )


def render_markdown(report: GateReport) -> str:
    lines = [
        "# Claim gate cross-check",
        "",
        f"Generated: {utc_now()}",
        f"Status: {'PASS' if report.ok else 'FAIL'}",
        "",
        "## Document gate tokens",
        "",
    ]
    for name, gates in report.doc_gates.items():
        lines.append(f"- `{DOC_PATHS[name]}`: {', '.join(sorted(set(gates))) or 'none'}")
    lines.extend(["", "## Index summary", ""])
    if report.index_summary:
        for k, v in report.index_summary.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- index missing")
    lines.extend(["", "## Issues", ""])
    if report.issues:
        for issue in report.issues:
            lines.append(f"- [{issue.level}] {issue.message}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Claim gate consistency check.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "results", "runs"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = GateReport()
    check_docs(report)
    check_index(report)
    check_second_dataset_phrases(report)
    check_general_phrases(report)
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "created_at_utc": utc_now(),
        "ok": report.ok,
        "doc_gates": report.doc_gates,
        "index_summary": report.index_summary,
        "issues": [i.__dict__ for i in report.issues],
    }
    write_json(os.path.join(args.output_dir, "claim_gate_check.json"), payload)
    write_text(os.path.join(args.output_dir, "claim_gate_check.md"), render_markdown(report))
    print(f"[claim-gates] ok={report.ok} issues={len(report.issues)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
