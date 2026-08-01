# Evidence cache build report

Verdict: `PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS`.

- Canonical archives present and whole-file SHA-256 verified: 6/6.
- ZIP CRC tests: 6/6 pass.
- Expected prediction-member identities: 180/180 exact; each archive also contains its single run-summary CSV.
- Streamed prediction-member SHA-256 values, schemas, coordinates, ordering, chronology, and label-known semantics: 180/180 pass.
- Cross-expert provider-label and row alignment groups: 60/60 pass.
- Compressed local cache: 1004185299 bytes.
- Streamed uncompressed prediction payload: 16952120128 bytes across 114494850 rows.
- Permanent prediction CSV extractions: 0.

The six candidates were already at the authoritative external cache location. Their destination hashes equal the frozen canonical hashes, so no repository copy or SSD access was needed. Archives were marked read-only. Member digests are observations derived from bytes inside a whole-archive-hash-verified canonical ZIP; they are not inherited or fabricated values. No model fitting, target metric, target oracle, or threshold selection occurred.
