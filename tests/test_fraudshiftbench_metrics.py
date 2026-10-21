"""Synthetic tests for the FraudShiftBench metrics API."""

from __future__ import annotations

import unittest

from fraudshiftbench.badges import badge_catalog, badge_for_evidence
from fraudshiftbench.metrics import (
    evidence_bound_claim_status,
    false_positive_workload,
    fraud_recall_at_budget,
    graph_harm_rate,
    graph_help_rate,
    high_confidence_harm_rate,
    leaderboard_instability_score,
    protocol_regret,
    protocol_risk_index,
    protocol_robust_selection_regret,
    rank_reversal_score,
)


class FraudShiftBenchMetricsTest(unittest.TestCase):
    def test_rank_reversal_score(self) -> None:
        a = {"gcn": 0.9, "sage": 0.7, "gat": 0.1}
        b = {"gcn": 0.1, "sage": 0.7, "gat": 0.9}
        self.assertEqual(rank_reversal_score(a, b), 1.0)
        self.assertEqual(leaderboard_instability_score(a, b), 1.0)
        self.assertIsNone(rank_reversal_score({"gcn": 1.0}, {"gcn": 0.1}))

    def test_protocol_risk_index_clips_and_ignores_missing(self) -> None:
        score = protocol_risk_index(
            leaderboard_flip_probability=1.5,
            rank_instability=0.5,
            temporal_prior_drift=None,
            protocol_metric_gap=-0.2,
        )
        self.assertAlmostEqual(score, (1.0 + 0.5 + 0.0) / 3.0)
        self.assertIsNone(protocol_risk_index())

    def test_graph_harm_help_rates(self) -> None:
        labels = ["graph_harm", "graph_help", "neutral", "graph_harm"]
        self.assertEqual(graph_harm_rate(labels), 0.5)
        self.assertEqual(graph_help_rate(labels), 0.25)

    def test_high_confidence_harm_rate(self) -> None:
        rows = [
            {"category": "graph_harm", "gnn_margin": "0.45"},
            {"category": "graph_harm", "gnn_margin": "0.1"},
            {"category": "graph_help", "gnn_margin": "0.9"},
        ]
        self.assertAlmostEqual(high_confidence_harm_rate(rows, threshold=0.4), 1 / 3)

    def test_fraud_review_metrics(self) -> None:
        labels = [1, 2, 1, 2]
        scores = [0.9, 0.8, 0.7, 0.1]
        self.assertEqual(fraud_recall_at_budget(labels, scores, 2), 0.5)
        self.assertEqual(false_positive_workload(labels, scores, 2), 1)

    def test_protocol_regret(self) -> None:
        scores = {"gcn": 0.4, "sage": 0.7}
        self.assertAlmostEqual(protocol_regret(scores, "gcn"), 0.3)
        robust = protocol_robust_selection_regret(
            {"p1": {"gcn": 0.4, "sage": 0.7}, "p2": {"gcn": 0.8, "sage": 0.6}},
            "sage",
        )
        self.assertAlmostEqual(robust["p2"], 0.2)
        self.assertAlmostEqual(robust["worst_case_regret"], 0.2)

    def test_evidence_status_and_badges(self) -> None:
        self.assertEqual(
            evidence_bound_claim_status("supported", has_real_results=True),
            "SUPPORTED_SINGLE_DATASET",
        )
        self.assertEqual(evidence_bound_claim_status("diagnostic"), "SUPPORTED_DIAGNOSTIC")
        self.assertEqual(evidence_bound_claim_status("sensitivity"), "SENSITIVITY_ONLY")
        self.assertEqual(evidence_bound_claim_status("scaffold"), "SCAFFOLD_ONLY")
        self.assertEqual(evidence_bound_claim_status("blocked"), "BLOCKED_NOT_CLAIMED")
        self.assertIn("SUPPORTED_10SEED", badge_catalog())
        self.assertEqual(badge_for_evidence("sensitivity"), "FDR_SENSITIVITY_ONLY")


if __name__ == "__main__":
    unittest.main()
