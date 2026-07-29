# Repository map

| Path | Role | Status |
| --- | --- | --- |
| `fraudshiftbench/` | Protocol contracts, metrics, evidence units, claim gates, model contracts | Canonical reusable framework |
| `data/` | Dataset interfaces and graph/feature construction | Canonical code; raw data excluded |
| `models/` | Legacy, modern, temporal, theory, mitigation, and GraphSafe components | Canonical code with legacy compatibility |
| `experiments/` | Legacy and multi-dataset runners | Code only; not all scaffolds have checked-in evidence |
| `scripts/tkde_rebuild/` | Evidence inventory, claim ledger, statistics, figures, bibliography, build | Canonical final analysis |
| `scripts/tkde_visual_rebuild/` | V2 tables, layout audits, delta gate, release hygiene | Canonical final publication design |
| `paper_tkde/` | Authoritative main/supplement LaTeX sources | Canonical manuscript |
| `paper/pdf/` | Final main and supplement PDFs | Canonical deliverables |
| `results/tkde_rebuild/` | Frozen aggregates, evidence inventory, claim ledger, provenance | Canonical evidence summary |
| `results/tkde_visual_rebuild/` | Final visual/editorial reports and provenance | Canonical readiness record |
| `results/github_publish/` | Public-repository audit, baseline, validation, push, and handoff | Publication metadata |
| `figures/` | Earlier `origin/main` Elliptic figures | Preserved legacy remote content |
| root legacy scripts | Earlier Elliptic reproduction | Preserved legacy remote content |

The exact source-to-public path map is
[`SOURCE_TO_GITHUB_PATH_MAP.csv`](SOURCE_TO_GITHUB_PATH_MAP.csv). Only the two
final PDFs are relocated, from `output/pdf/` to `paper/pdf/`.

