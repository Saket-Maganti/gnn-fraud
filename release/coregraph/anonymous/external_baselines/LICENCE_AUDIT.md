# Official baseline licence audit

Verified 2026-07-29 against the pinned upstream repositories.

| Baseline | Pinned commit | Licence finding | Build disposition |
|---|---|---|---|
| Mowst | `2e3569962d2388bfda4535cdd1fc0b6eaec88a28` | MIT file present | Adapter prepared; upstream checkout not installed |
| GraphMETRO | `e2b6ab62c6d7a3d72b6508db9bfce49336a9b129` | No licence file or repository licence declaration found | `UNAVAILABLE_LICENSE`; do not clone, vendor, or count as ready |
| CIGA | `454801108737ff8855ac2be947201dd9338dff37` | MIT file present | Adapter prepared; task scope is graph classification |
| EERM | `ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9` | No licence file or repository licence declaration found | `UNAVAILABLE_LICENSE`; do not clone, vendor, or count as ready |
| TGN | `d55bbe678acabb9fc3879c408fd1f2e15919667c` | Apache-2.0 file present | Adapter prepared; event/transaction tasks only |
| GOOD | `b53566c9297bc65b90a7f2213fb9ffa930f5b6e5` | GPL-3.0 file present | External-process boundary; do not mix source into an unresolved project licence |

The repository root still contains `LICENSE_REVIEW_REQUIRED.md`. This build
does not infer a project-wide licence. Paper citations, dataset terms, and code
licences are distinct. GraphMETRO and EERM require an explicit upstream licence
or written permission before official-code integration.

The GitHub pages and raw licence files were inspected directly. Commit pins were
resolved with `git ls-remote` and must not float during later runs.
