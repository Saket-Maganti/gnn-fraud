# Data and tasks

Adapters expose node, edge, and transaction classification with typed unique
identifiers. Unknown labels never enter supervised masks. Metrics and exports
declare their prediction unit, and cross-expert alignment is by identifier,
never row order.

V2 adapters correct temporal semantics: DGraphFin uses first incident event
time without changing direction or type; T-Finance rejects temporal claims when
only pseudo-time exists; Elliptic-family adapters preserve its unknown/fraud/
normal convention. IBM AML Small and Medium are transaction tasks; Large is
resource-blocked until an execution envelope is approved. The GOOD adapter
exposes only selected, declared shift settings.

Raw provider data is never downloaded by a test or replaced silently. Every
real run must bind a dataset manifest and checksum.
