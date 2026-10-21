# Protocol Definitions

Instantiated protocol contracts; no hierarchy is implied among separately recorded coordinates.

protocol,temporal_masks,graph_visibility,label_availability,selection_or_threshold,instantiated_on
strict-inductive,chronological,training nodes/edges only during training; held-out structure available at evaluation,test labels hidden,validation F1,"Elliptic, DGraphFin"
isolated-inductive,same chronological masks,held-out nodes isolated from training-period cross-split edges at evaluation,test labels hidden,validation F1,"Elliptic, DGraphFin"
transductive,same masks,full graph structure visible,test labels hidden,validation F1,"Elliptic, DGraphFin"
late-window holdout,60/20/20 chronological,"shared node-history map uses first 60%, which is this protocol's training interval",test transaction labels hidden,fixed 0.5 threshold for saved F1,IBM AML-Data
early-to-late transfer,50/20/30 chronological,"classifier labels use first 50%; shared label-free node-history map uses first 60%, including covariates from 50-60%",test transaction labels hidden,fixed 0.5 threshold for saved F1,IBM AML-Data

