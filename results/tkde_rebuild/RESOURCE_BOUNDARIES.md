# Resource Boundaries

Resource status is not predictive performance. Rows with zero outputs remain outside rankings.

cell,resource_envelope_or_reason,result_outputs,prediction_exports,status,interpretation
IBM AML HI-Large,safe resource guard,0,0,SAFE_RESOURCE_BLOCKED,not in predictive ranking
IBM AML LI-Large,safe resource guard,0,0,SAFE_RESOURCE_BLOCKED,not in predictive ranking
IBM AML HI-Medium GINE h64,single Tesla T4 CUDA OOM,0 of 20 planned,0,RESOURCE_BLOCKED_T4_CUDA_OOM,not in predictive ranking
IBM AML LI-Medium GINE h64,single Tesla T4 CUDA OOM,0 of 20 planned,0,RESOURCE_BLOCKED_T4_CUDA_OOM,not in predictive ranking
DGraphFin GAT h64/l2,Tesla T4 CUDA OOM,0 of 20 planned,0,BLOCKED_T4_OOM,memory-reduced h32/l1 diagnostic is not a replacement
DGraphFin GraphSAGE max-pool rerun,larger GPU required,0,0,BLOCKED_WAITING_FOR_GPU,no imported result CSV

