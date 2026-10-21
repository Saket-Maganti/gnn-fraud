# Finite counterexamples

- Protocol one-hot: one basis vector per observed protocol fits arbitrary source risks but maps an unseen combination to no identified coefficient.
- Fixed mixture: crossing expert risks force positive worst-contract regret.
- Confidence only: an overconfident wrong expert beats a better calibrated expert by confidence while having higher Brier risk.
- Hidden mask: masking after softmax can retain or lose mass incorrectly; mask before normalization.
- Target-label leakage: choosing a gate or threshold on target outcomes converts evaluation information into fitting information.
- Wrong pairing: `(elliptic, seed=1)` and `(dgraphfin, seed=1)` are distinct experimental units.
- Runtime substitution: end-to-end training seconds do not measure warmed inference latency.
- Noisy contract: two latent contracts can generate the same noisy metadata without diagnostics, so the latent cause is not identifiable.
- Arbitrary interaction: XOR risk has zero marginal axis effects but nonzero composition risk.

Every finite numeric construction is mirrored in `executable_checks.py`; passing a check verifies the construction, not the general theorem's deployment assumptions.
