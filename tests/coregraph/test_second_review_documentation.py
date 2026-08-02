from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT_V3_SPEC = (
    ROOT / "results/coregraph_build/PILOT_V3_SPECIFICATION.md"
)
if not PILOT_V3_SPEC.is_file():
    PILOT_V3_SPEC = ROOT / "specifications/PILOT_V3_SPECIFICATION.md"


def test_pilot_semantics_are_aligned_across_public_documentation() -> None:
    paths = (
        ROOT / "docs/coregraph/METHOD_CARD.md",
        ROOT / "docs/coregraph/OBJECTIVE_SPECIFICATION.md",
        ROOT / "docs/coregraph/STATISTICAL_ANALYSIS_PLAN.md",
        ROOT / "docs/coregraph/EXPERIMENT_PROTOCOL.md",
        ROOT / "docs/coregraph/REPRODUCIBILITY.md",
        PILOT_V3_SPEC,
        ROOT / "paper_iclr/sections/04_problem.tex",
        ROOT / "paper_iclr/sections/05_method.tex",
        ROOT / "paper_iclr/sections/07_experiments.tex",
        ROOT / "paper_iclr/sections/08_results_placeholder.tex",
        ROOT / "paper_iclr/sections/10_conclusion.tex",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    required_phrases = (
        "contract-level feasible oracle",
        "instance-clairvoyant oracle",
        "source-contract budgets",
        "source-contract abstention capacities",
        "expert-prediction seed",
        "router-training seed",
        "graphsafe_confidence_abstention_component",
    )
    for phrase in required_phrases:
        assert phrase in combined.lower()
    assert "graphsafe v2 compatibility implementation" not in combined.lower()
    assert "ready for the declared runs" not in combined.lower()
    assert "instance oracle is the headline" not in combined.lower()
