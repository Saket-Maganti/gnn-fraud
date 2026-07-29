# Licence and reuse audit

Status: `BLOCKED_FOR_FULL_HEADLINE_BASELINE_SET`

Pinned and licence-file verified:

- Mowst: MIT; pending isolated integration and parity.
- CIGA: MIT; pending integration and graph-classification task parity.
- official TGN: Apache-2.0; pending event-adapter parity.
- GOOD: GPL-3.0; pending isolated integration and redistribution-boundary
  review.

Blocked:

- GraphMETRO: no licence file verified at pinned commit
  `e2b6ab62c6d7a3d72b6508db9bfce49336a9b129`.
- EERM: no licence file verified at pinned commit
  `ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9`.
- Provider fraud datasets retain their own manual acquisition and terms.
- Repository-wide public reuse remains subject to the existing root licence
  review; this build does not invent a licence.

Internal GroupDRO, VREx, IRM, feature/tree experts, CoReGraph, and sampled
GCN/SAGE adapters are labelled validated reimplementations, not official
upstream code. Legacy Transformer/GPS/PCGNN/SnapshotTGN aliases remain
diagnostic. A blocked or pending adapter cannot be promoted by configuration.

Therefore baseline/licence gate 4 does not pass, heavy headline comparisons are
not authorised, and no GitHub push is made by this build session.
