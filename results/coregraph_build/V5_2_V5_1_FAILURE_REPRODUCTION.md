# V5.2 Reproduction of the V5.1 Numerical Invariant Failure

The V5.1 failed coordinate stored both routed scores and routing weights as float32. Its routing-weight sums ranged from `0.99999994` to `1.0000001`; 34 rows exceeded one.

A deterministic one-row fixture reproduces the defect independently of empirical outcomes:

```text
weights(float32) = [0.33333337, 0.33333337, 0.33333337]
sum(float32)     = 1.0000001
expert scores    = [0.9, 0.9, 0.9]
label            = 1
old routed score = 0.9000001
old row regret   = approximately -2.38e-08
```

The real V5.1 failure was `-1.677436056723991e-08`, the same ordinary float32 accumulation scale. This is `NUMERICAL_IMPLEMENTATION_INVARIANT_FAILURE`, not a scientific gate result.
