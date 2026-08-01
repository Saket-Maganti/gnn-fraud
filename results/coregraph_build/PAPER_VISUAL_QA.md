# CoReGraph Level-4 paper visual QA

Status: `PASS`

The final locally compiled, anonymous paper artifacts were rendered with
Poppler and inspected page by page at full detail on 2026-08-01.

| Artifact | Pages | Page size | Bytes | SHA-256 | Result |
|---|---:|---|---:|---|---|
| `paper_iclr/main.pdf` | 13 | US letter | 416,915 | `c3f082a3c278f807d425629f0c324ad5aa93fe811eaa3cfde89b41fb7729e993` | PASS |
| `paper_iclr/supplement.pdf` | 5 | US letter | 313,308 | `404aca64dd09bb449435dd39687cd63bf007f4b674c2d5b12efcda0cc22a2c03` | PASS |

All 18 pages passed checks for clipped or overlapping text, cropped figures,
missing glyphs, broken references, unreadable captions, stray identity data,
and unintended numerical results. The eight non-empirical figures are legible.
The seven empirical figure templates visibly say `PENDING VALIDATED RUNS` and
contain no curves, points, or values. Red `RESULT_PENDING[...]` tokens remain
intentional and distinguish unexecuted empirical cells from build evidence.

During review, stale archive-availability wording on main-paper pages 7, 8,
and 12 was corrected. Those pages were rebuilt, rerendered, and reinspected.
The final text now records the six locally verified RB09v3 archives without
claiming that the empirical pilot was executed.

The paper uses the documented fallback article layout because no official
target-year ICLR style was available at build time. A target-style rebuild and
fresh visual check remain a pre-submission gate.
