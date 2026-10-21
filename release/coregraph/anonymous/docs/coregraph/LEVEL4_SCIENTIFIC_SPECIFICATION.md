# CoReGraph Level-4 scientific specification

Status: `PRE_RUN_RESULTS_BLOCKED`.

## Scientific question

CoReGraph studies compositional deployment-contract generalisation: given heterogeneous experts and labelled source contracts, learn a policy that can route under an unseen combination of graph, temporal, selection, review-budget, and resource conditions without target-label access. Fraud is one demanding evaluation domain, not the definition of the method.

The contract is

`c = (time, visibility, construction, selection, budget, resource)`.

Every axis may be categorical or continuous and may be observed, missing, uncertain, out of source range, unseen, or part of an unseen combination. Contract metadata is distinct from label-free expert and graph diagnostics. A hybrid experimental extension may infer latent factors, but it cannot be promoted to the main contribution without its frozen gate.

## Method family

The main model uses a source-normalized factorised axis encoder, bounded pairwise interactions, per-expert label-free diagnostics, a contract-level prior, a bounded instance correction, pre-softmax resource masks, a source-frozen abstention policy, and a multi-term source objective. Every component has an ablation: protocol one-hot; flat MLP; no interactions; no diagnostics; contract-only; instance-only; no regret/CVaR; no budget; no stability; no abstention; and no resource intervention.

The deployable router cannot consume target outcomes, target-derived thresholds, a target oracle, or label-dependent diagnostics. The whole-contract feasible oracle and instance-clairvoyant oracle are offline diagnostics with different meanings.

## Benchmark layers

1. FraudShiftBench integration: Elliptic and DGraphFin under strict inductive, isolated inductive, and transductive-structure contracts, plus resource and budget interventions.
2. Controlled mechanisms: fifteen deterministic mechanisms with tiny fixtures and future scalable modes.
3. Non-fraud graph OOD: GOOD is primary; an OGB molecular shift is fallback. Neither official repository or dataset is installed in this phase.

## Evidence and claim boundary

The six RB09v3 archives match their frozen SHA-256 values. All 180 prediction-member identities, streamed hashes, schemas, coordinates, chronology, label-known semantics, 60 cross-expert alignment groups, and 20 cross-protocol dataset-seed row scopes pass from the local cache without SSD access or permanent extraction. The registry contains 180 byte-verified role-neutral coordinates, 60 held-out-protocol scenarios, and 540 scenario-local bindings. This closes the pilot-input integrity gate, not any empirical claim.

All seven empirical hypotheses start `BLOCKED_PENDING_RESULTS` or a stricter resource/licence status. Theory statements do not imply empirical performance. Paper result cells use typed blocked macros, never plausible numbers.

## Intended build-phase outcome

This repository can reach a validated, maximum-ceiling pre-run state. It cannot be called submission-ready before saved-output pilots, strong official baselines, real experiments, paired statistics, result-driven paper population, and independent review.
