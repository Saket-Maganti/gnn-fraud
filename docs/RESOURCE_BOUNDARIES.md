# Resource boundaries

Resource status is not predictive performance. Unmeasured cells remain outside
rankings and Pareto sets.

| Cell | Recorded boundary | Interpretation |
| --- | --- | --- |
| IBM AML HI-Large | safe resource guard | unmeasured |
| IBM AML LI-Large | safe resource guard | unmeasured |
| IBM AML HI-Medium GINE h64 | single Tesla T4 CUDA OOM | unmeasured |
| IBM AML LI-Medium GINE h64 | single Tesla T4 CUDA OOM | unmeasured |
| DGraphFin GAT h64/l2 | Tesla T4 CUDA OOM | memory-reduced diagnostic is not a replacement |
| DGraphFin GraphSAGE max-pool rerun | larger GPU required | no imported result CSV |

The machine-readable authority is
`results/tkde_rebuild/RESOURCE_BOUNDARIES.csv`.

