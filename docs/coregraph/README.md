# CoReGraph / ContractShift pre-run system

This directory documents the new ICLR-track research system. It is separate
from the frozen FraudShiftBench/TKDE scientific assets. No file under the
frozen boundary may be modified to support a CoReGraph claim.

The implementation is ready for deterministic smoke and synthetic validation.
The saved-output pilot is not execution-ready: manifest conversion, a dry-run
completeness audit, and a third independent review remain required. Provider
data and official external repositories remain explicit inputs; no loader
substitutes synthetic data for a missing dataset.

Start with `RESEARCH_IDENTITY.md`, then `EXPERIMENT_PROTOCOL.md`. Before any
heavy execution, run the local gates in `ICLR_GO_NO_GO_GATES.md`.
