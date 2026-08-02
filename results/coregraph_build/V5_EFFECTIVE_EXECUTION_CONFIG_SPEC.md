# V5 effective execution configuration

Schema: `coregraph_v5_effective_execution_config_v1`.

The canonical JSON hash binds base-config and preregistration hashes; configured and effective chunk rows; max workers; real/synthetic mode; synthetic-fixture flag; float dtype; deterministic algorithms; output, metric, and method-registry schemas; archive streaming; source sampling; target inference; dependency lock; and code SHA.

The hash is part of the coordinate key and identity hash, run/scenario manifests, checkpoints, policy freeze, evaluation, COMPLETE identity, aggregate, gate, package coordinate manifest, output checksum header, and final summary. A mismatch fails resume and package validation. The authoritative runner accepts only one worker and has no method, scenario, or dirty-tree execution override.
