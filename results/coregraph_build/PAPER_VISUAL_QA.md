# CoReGraph Level-4 paper visual QA

Status: `PASS`

The final locally compiled, anonymous paper artifacts were rendered with
Poppler and inspected page by page at full detail on 2026-08-01.

| Artifact | Pages | Page size | Bytes | SHA-256 | Result |
|---|---:|---|---:|---|---|
| `paper_iclr/main.pdf` | 13 | US letter | 416,901 | `c02464208bed0fc7ecac6af68fce9f1568485aa8320cc0110448bc5305eb33e0` | PASS |
| `paper_iclr/supplement.pdf` | 5 | US letter | 313,298 | `73bd71b2d22850c8afc988236f628c00894e02ff5e7a0cfb8b4140111c1aa6e7` | PASS |

All 18 pages passed checks for clipped or overlapping text, cropped figures,
missing glyphs, broken references, unreadable captions, stray identity data,
and unintended numerical results. The eight non-empirical figures are legible.
The seven empirical figure templates visibly say `PENDING VALIDATED RUNS` and
contain no curves, points, or values. Red `RESULT_PENDING[...]` tokens remain
intentional and distinguish unexecuted empirical cells from build evidence.

The final build fixes the PDF source epoch. Two consecutive compilations
produced identical SHA-256 values, and all 18 Poppler-rendered page PNGs were
byte-identical to the previously inspected committed PDFs. The deterministic
metadata change therefore introduced no visual delta.

During review, stale archive-availability wording on main-paper pages 7, 8,
and 12 was corrected. Those pages were rebuilt, rerendered, and reinspected.
The final text now records the six locally verified RB09v3 archives without
claiming that the empirical pilot was executed.

The paper uses the documented fallback article layout because no official
target-year ICLR style was available at build time. A target-style rebuild and
fresh visual check remain a pre-submission gate.
