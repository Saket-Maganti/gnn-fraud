# Reproducibility

Every run ID hashes canonical config, code commit, dataset manifest, and
dependency lock. Manifests use atomic writes and distinguish planned, running,
complete, smoke-pass, failed, and resource-blocked states. Resume validates
hashes, output schema, status, result checksum, and prediction checksum.

Predictions include typed ID, contract ID, split, label status, score, expert,
config hash, frozen abstention decision and provenance, execution status,
expert-prediction seed, and deterministic method-specific router-training
seed. The expert-prediction seed is the inferential block; router seeds are
secondary provenance only. Telemetry records wall time and process memory.
Random seeds cover Python, NumPy, Torch, CUDA, batching, and synthetic
generation.

Saved-prediction V5 manifests separate immutable scientific evidence from its
evaluation role. One base artifact is identified by dataset, task, protocol,
expert, expert-prediction seed, fold, bytes, and role-neutral contract
coordinates. A separate scenario binding assigns source train/validation or
known-label target-test scope for one held-out protocol. The same base artifact
may change roles across scenarios but can never occupy both roles inside one
scenario. Routing-cost proxies and measured compute evidence are recorded
separately and unresolved legacy provenance remains explicit.

Anonymous release tooling excludes git history, raw data, predictions, local
absolute paths, and identity. Frozen TKDE assets are rechecked by SHA-256 before
commit or release packaging.
