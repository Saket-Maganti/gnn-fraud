PY ?= python3
PYTESTS = \
	tests/test_fraudshiftbench_claim_gates.py \
	tests/test_fraudshiftbench_evidence_units.py \
	tests/test_fraudshiftbench_metrics.py \
	tests/test_fraudshiftbench_protocols.py \
	tests/test_fraudshiftbench_red_team_integrity.py \
	tests/test_tkde_visual_release_audits.py

.PHONY: help compile test unittest support claims figures tables paper delta public-audit \
	coregraph-compile coregraph-lint coregraph-typecheck coregraph-test \
	coregraph-local-gates

help:
	@echo "Dataset-free targets: compile test unittest support claims figures tables paper delta public-audit"

compile:
	$(PY) -m compileall -q fraudshiftbench data models experiments scripts tests utils

test:
	$(PY) -m pytest -q $(PYTESTS)

unittest:
	$(PY) -m unittest discover -s tests -t . -p "test_*.py"

support:
	$(PY) scripts/tkde_rebuild/validate_support_relation.py --frozen-only

claims:
	$(PY) scripts/audit_claim_language.py
	$(PY) scripts/safety_check_no_heavy_defaults.py --output-dir results/tkde_visual_rebuild/validation/safety

figures:
	$(PY) scripts/tkde_rebuild/make_figures.py

tables:
	$(PY) scripts/tkde_visual_rebuild/build_main_tables.py
	$(PY) scripts/tkde_visual_rebuild/build_curated_supplement_tables.py
	$(PY) scripts/tkde_rebuild/build_bibliography.py

paper:
	bash scripts/tkde_rebuild/compile_papers.sh

delta:
	$(PY) scripts/tkde_visual_rebuild/scientific_delta_gate.py --strict --skip-baseline-archives

public-audit:
	$(PY) scripts/github_publish/validate_public_tree.py .

coregraph-compile:
	$(PY) -m compileall -q coregraph scripts/coregraph tests/coregraph

coregraph-lint:
	$(PY) -m ruff check coregraph scripts/coregraph tests/coregraph --select F,I

coregraph-typecheck:
	$(PY) -m mypy coregraph/contracts coregraph/routing coregraph/objectives \
		coregraph/evidence.py --ignore-missing-imports --show-error-codes

coregraph-test:
	$(PY) -m pytest -q

coregraph-local-gates: coregraph-compile coregraph-lint coregraph-typecheck coregraph-test
	$(PY) scripts/coregraph/check_theory_numerically.py
	$(PY) scripts/coregraph/validate_theory_status.py
	$(PY) scripts/coregraph/validate_notebooks.py
	$(PY) scripts/coregraph/run_coregraph_smoke.py
	$(PY) scripts/coregraph/validate_paper_skeleton.py
	$(PY) scripts/coregraph/build_anonymous_release.py
	$(PY) scripts/coregraph/audit_anonymous_release.py
	$(PY) scripts/coregraph/hash_frozen_assets.py --verify
