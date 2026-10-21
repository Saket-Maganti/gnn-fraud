# Result import runbook

Copy a completed wave archive into a new directory under
`results/coregraph_import/`. Do not merge files manually. Verify the outer
checksum, then every manifest/result/prediction checksum. Reject duplicate run
IDs with different hashes, stale schemas, incomplete seed blocks, and ID/label
misalignment. Preserve failed and resource-blocked manifests.

Only an imported, validated prediction artifact may become an
`EvidenceUnitV2`. Run the SupportEngine before a paper asset generator is
allowed to consume it.
