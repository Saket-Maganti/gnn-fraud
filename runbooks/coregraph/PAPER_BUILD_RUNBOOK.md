# Paper build runbook

Before empirical runs, the paper is expected to contain `RESULT_PENDING` and
`TABLE_PENDING_RUNS` markers. After validated imports:

```bash
.venv/bin/python scripts/coregraph/generate_paper_assets.py
bash scripts/coregraph/build_iclr_paper.sh
.venv/bin/python scripts/coregraph/validate_claims.py
```

Never type result numbers into TeX. All result tables and empirical figures are
generated from prediction-level artifacts that have an eligible evidence unit.
Reconfirm the target-year ICLR template, anonymity, page limits, disclosure,
ethics, and reproducibility policy before submission.
