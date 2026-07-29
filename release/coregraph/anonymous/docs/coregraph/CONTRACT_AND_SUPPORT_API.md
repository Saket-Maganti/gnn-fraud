# Contract and support API

The V3 contract coordinates are `time`, `visibility`, `construction`,
`selection`, `budget`, and `resource`. Visibility records node and edge
access separately. Construction composes history, orientation, edge-feature,
and topology policies. Budget composes review capacity, costs, abstention, and
latency. Resource records device, memory, latency, explicit expert blocks, and
measurement status. A one-way adapter migrates V2 payloads.

Contracts serialize canonically to JSON/YAML. The coordinate SHA-256 excludes
environment ID, role, dataset, and task so matched scientific coordinates can
be compared across environments. The separate artifact/environment SHA-256
includes the complete contract identity.

`EvidenceUnitV2` identifies prediction artifacts, metrics, seeds, provenance,
validation, integrity, construct validity, feasibility, import state, hashes,
and scope. `TypedClaim` records the claim quantifier, comparison, direction,
uncertainty, pairing, prediction requirement, permitted wording, and a machine
predicate.

`SupportEngine` is deliberately conservative. Theoretical support requires a
verified proof artifact and a matching theorem-status hash; a boolean flag is
not evidence. Resource boundaries arise only from claim-relevant cells. Scope
comparison distinguishes exact, narrower, wider, and incompatible requests, so
a narrower request is not misreported as widening. The engine returns matched
and excluded artifact IDs, missing requirements, pairing diagnostics,
contradictions, and scope relations. It never substitutes for curator review
of scientific meaning.
