# Secret and privacy audit

Final pre-push status: **PASSED**.

## Scanner coverage

- `gitleaks`, `trufflehog`, and `detect-secrets` were not installed.
- A deterministic filename scan covered `.env`, `.netrc`, Kaggle credential
  names, SSH private-key names, PEM/key files, cookies, and credential JSON
  outside `.git` and the excluded virtual environment. No candidate credential
  file was found.
- Deterministic content regexes covered GitHub, OpenAI, Hugging Face, and AWS
  token formats plus private-key headers. The allowlist candidate surfaces had
  zero matches.
- Private-path scanning covered macOS, Linux, mounted-volume, and Windows
  user-home prefixes. Matches occurred only in excluded LaTeX logs,
  excluded starting-state/audit payloads, and the publication validator's own
  test patterns. None is selected as public content.
- Email scanning of the candidate source/paper/evidence surfaces returned zero
  matches.
- Raw-data and prediction-path enumeration confirmed that `data/raw/`, nested
  imported outputs, synthetic/raw predictions, and row-level graph-harm files
  exist locally. They are explicitly excluded. Only named aggregate summaries,
  evidence locks, and prediction manifests from the final source-analysis
  dependency closure may be copied.
- The large-file audit identified every file over 10 MiB outside `.git`.
  All raw data, archives, environment binaries, row-level prediction-adjacent
  payloads, and files above 100 MiB are excluded. The only selected file above
  10 MiB is the 18.6 MiB frozen scalar provenance map.

## Anonymity and identity

The final visual rebuild recorded `double_blind_ok=true`, zero high-severity
paper-source findings, and no author identity in PDF metadata. The identity-
bearing JSON configuration emitted by the earlier local scanner is excluded;
only the identity-free Markdown result is allowlisted. Repository identifiers
required by the GitHub baseline, push report, PR body, and analysis handoff are
not manuscript author metadata.

## Final clean-checkout and staged-tree gate

The exact staged tree contains 475 files and 36,198,761 logical file bytes.
The largest ordinary Git file is the 19,502,682-byte frozen scalar provenance
CSV. No staged object exceeds 100 MiB.

Both the clean working tree and a Git archive produced from the exact staged
index passed the deterministic public-tree validator with zero findings.
Full-index scans, including text larger than the validator's normal 4 MiB
limit, found:

- zero GitHub, OpenAI, Hugging Face, AWS, or private-key token patterns;
- zero private absolute paths or home aliases;
- zero email addresses;
- zero credential-bearing filenames;
- zero raw-data, model-payload, or archive suffixes;
- zero symlinks and zero Git LFS rules.

The staged archive also passed all 36 frozen scientific hashes, 14 support
cases, 27 pytest cases, and seven unittest-style cases. The public-branch gate
therefore passes.
