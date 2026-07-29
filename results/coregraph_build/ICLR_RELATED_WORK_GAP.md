# ICLR related-work and policy gap

The closest verified competitors are GraphMETRO for mixtures of graph shifts,
Mowst for feature/graph expert collaboration, GMoE for message-passing experts,
and concurrent STEM-GNN for routing stability under corruptions. CoReGraph's
intended distinction is the joint typed treatment of time, visibility,
construction, selection, review budget, and resource feasibility, evaluated on
unseen combinations and contract regret. That distinction is a hypothesis until
direct baselines run.

GraphMETRO's official repository has no licence file. This blocks code
integration, not citation or conceptual comparison. EERM is task-compatible
with temporal node prediction but has the same upstream licence problem. CIGA
has an MIT licence but its official task surface is primarily graph
classification and cannot be relabelled as a fraud-node baseline.

STEM-GNN (arXiv:2602.09258) is a 2026 concurrent preprint and must be rechecked
before submission. `FG-EGCN` was requested in the governing prompt, but no
unambiguous primary source was identified in the audit; it is recorded as an
identity blocker rather than guessed.

## Official ICLR policy check

No ICLR 2027 author guide was available on 2026-07-29. The most recent official
guide was ICLR 2026: double blind, a nine-page initial main-text limit, the
official year-specific template, and recommended reproducibility/ethics
statements. It also required disclosure of significant LLM use. These are not
assumed to remain unchanged. Before submission, replace the article fallback
with the current official style and rerun anonymity, page, policy, and LLM-use
checks.

Official sources:

- https://iclr.cc/Conferences/2026/AuthorGuide
- https://iclr.cc/public/CodeOfEthics
- https://iclr.cc/FAQ/LLM
