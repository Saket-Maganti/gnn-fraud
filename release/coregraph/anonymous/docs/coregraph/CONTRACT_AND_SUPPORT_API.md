# Contract and support API

The contract coordinates are `time`, `visibility`, `construction`,
`selection`, `budget`, and `resource`. Every coordinate has a declared
`UNKNOWN` representation. Contracts serialize canonically to JSON/YAML and
produce stable SHA-256 identifiers. Dataset/task, access regime, environment
role, and schema version are also explicit.

`EvidenceUnitV2` identifies prediction artifacts, metrics, seeds, provenance,
validation, integrity, construct validity, feasibility, import state, hashes,
and scope. `TypedClaim` records the claim quantifier, comparison, direction,
uncertainty, pairing, prediction requirement, permitted wording, and a machine
predicate.

`SupportEngine` is deliberately conservative. It can report supported,
theoretical-only, resource-bounded, incomplete scope or seeds, missing
predictions, integrity/construct exclusion, diagnostic-only, refuted, or not
applicable. It returns matched and excluded artifact IDs, missing requirements,
pairing diagnostics, contradictions, and scope widening. It never substitutes
for curator review of scientific meaning.
