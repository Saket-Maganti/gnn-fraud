"""Selective-risk transfer bound under explicit density-ratio assumptions."""

from __future__ import annotations


def selective_risk_transfer_bound(
    *,
    source_selective_risk: float,
    source_coverage: float,
    target_coverage_lower_bound: float,
    density_ratio_bound: float,
    loss_upper_bound: float = 1.0,
) -> float:
    """Bound target selective risk when target joint mass is source-dominated.

    If ``dP_t/dP_s <= density_ratio_bound`` on the accepted set and target
    coverage is at least ``target_coverage_lower_bound``, then
    ``R_t^sel <= kappa * cov_s * R_s^sel / cov_t``. The bounded loss supplies
    the trivial cap. This is not a distribution-free target guarantee.
    """

    if not 0 <= source_selective_risk <= loss_upper_bound:
        raise ValueError("source selective risk is outside the declared loss range")
    if not 0 <= source_coverage <= 1 or not 0 < target_coverage_lower_bound <= 1:
        raise ValueError("coverage values are invalid")
    if density_ratio_bound < 0 or loss_upper_bound <= 0:
        raise ValueError("density and loss bounds must be positive")
    bound = (
        density_ratio_bound
        * source_coverage
        * source_selective_risk
        / target_coverage_lower_bound
    )
    return min(float(loss_upper_bound), float(bound))
