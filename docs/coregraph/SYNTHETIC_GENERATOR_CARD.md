# Synthetic ContractShift generator card

The generator has eight named regimes: `graph_best`, `feature_best`,
`ordering_crosses`, `fixed_mixture_regret`, `factorised_generalisation`,
`interaction_breaks_factorisation`, `resource_mask`, and
`budget_changes_expert`. Controls set feature and graph signal, homophily,
heterophily, class prior, topology, temporal/concept/covariate drift,
missingness, review budget, label delay, availability, and seed.

Generation is deterministic for a fixed configuration. Each artifact includes
the true mechanism and source/target contracts, enabling falsifiable recovery,
router stability, regret, and interaction tests. Synthetic evidence validates
mechanisms and theory; it does not establish real-fraud performance.

The generator is the only sanctioned synthetic path. Real dataset adapters
must fail loudly when provider files are absent.
