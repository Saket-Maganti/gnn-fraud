# Developer and execution handoff

Use the dedicated worktree and branch. Create `.venv` from the lock, run
`make coregraph-local-gates`, then freeze the analysis plan and matrices. Stage
provider data manually and create manifests; do not edit loaders to bypass
missing files.

Run the saved-output pilot before training. Integrate each official baseline in
its isolated environment, execute its parity fixture, and update the baseline
registry only with evidence. Profile the resource grid, then execute five-seed
screening. Review the pilot and screening gate without opening confirmatory
results beyond the preregistered decision rule. Only then schedule ten-seed
final and ablation waves.

Import outputs by manifest/checksum, run statistics, rebuild paper figures and
tables from generated data, validate claims, build the anonymous package, and
rerun frozen-boundary verification. Exact commands are in
`runbooks/coregraph/EXECUTION_ORDER.md`.
