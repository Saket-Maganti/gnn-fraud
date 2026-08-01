# Resource-mask guarantee

Status: `PROVED_AND_REVIEWED` for the algebraic mask semantics.

Given logits \(z_e\) and feasible mask \(m_e\), define \(\tilde z_e=z_e\) when \(m_e=1\) and \(-\infty\) otherwise, then apply softmax. Every unavailable expert has exactly zero probability, while feasible probabilities sum to one when the feasible set is nonempty. Memory, latency, and invocation constraints are composed into the Boolean mask before normalization, so selected experts satisfy every declared constraint.

When no expert is feasible, softmax is not interpreted as a routing distribution. The implementation emits the zero vector, selected-expert sentinel \(-1\), and forced abstention/fallback status. This explicit branch avoids NaNs and prevents hidden use of an unavailable expert. Feasibility is only as trustworthy as the resource measurements and their units.
