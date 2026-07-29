# Reproducibility

Every run ID hashes canonical config, code commit, dataset manifest, and
dependency lock. Manifests use atomic writes and distinguish planned, running,
complete, smoke-pass, failed, and resource-blocked states. Resume validates
hashes, output schema, status, result checksum, and prediction checksum.

Predictions include typed ID, contract ID, split, label status, score, expert,
and config hash. Telemetry records wall time and process memory. Random seeds
cover Python, NumPy, Torch, CUDA, batching, and synthetic generation.

Anonymous release tooling excludes git history, raw data, predictions, local
absolute paths, and identity. Frozen TKDE assets are rechecked by SHA-256 before
commit or release packaging.
