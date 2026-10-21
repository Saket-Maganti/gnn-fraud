# Build and validation commands

All commands run from the repository root. They consume included frozen
aggregates and do not train models, download data, or launch provider jobs.

```bash
python3 -m pip install -r requirements-publication.txt
make compile
make test
make unittest
make support
make claims
make figures
make tables
make delta
```

With `pdflatex`, `bibtex`, and the IEEEtran class available:

```bash
make paper
```

The build writes temporary PDFs and auxiliaries under `paper_tkde/`; those files
are ignored. The two checked-in final PDFs remain under `paper/pdf/`.

Full model experiments are intentionally outside this lightweight build. See
`docs/REPRODUCIBILITY.md` and `docs/DATA_ACCESS.md`.

`make claims` runs the claim-language and no-heavy-default audits over the
curated surface. The final 22-claim shape/status gate is part of `make delta`.
