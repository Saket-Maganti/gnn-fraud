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
	coregraph-coverage coregraph-local-gates coregraph-level4-validate \
	coregraph-evidence-cache-check coregraph-pilot-plan coregraph-pilot-validate \
	coregraph-full-plan coregraph-analysis coregraph-paper coregraph-release \
	coregraph-cleanroom

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
	$(PY) -m ruff check coregraph scripts/coregraph tests/coregraph

coregraph-typecheck:
	$(PY) -m mypy coregraph/contracts coregraph/routing coregraph/objectives \
		coregraph/experts coregraph/data/leakage.py \
		coregraph/experiments/pilot.py \
		coregraph/experiments/manifest_conversion.py \
		coregraph/experiments/scenario_manifests.py \
		coregraph/experiments/canonical_recovery.py \
		coregraph/experiments/protocol_registry.py \
		coregraph/experiments/contract_splits.py coregraph/evaluation/statistics.py \
		coregraph/evaluation/metrics.py coregraph/evaluation/selective.py \
		coregraph/evaluation/resources.py coregraph/evaluation/counterfactual.py \
		coregraph/evidence coregraph/diagnostics coregraph/resources \
		coregraph/baselines coregraph/benchmarks coregraph/theory coregraph/theory_checks \
		scripts/coregraph/run_statistical_analysis.py \
		scripts/coregraph/evaluate_pilot_gate.py \
		scripts/coregraph/run_saved_output_pilot.py \
		scripts/coregraph/convert_prediction_manifests_v4.py \
		scripts/coregraph/recover_canonical_manifests_v5.py \
		--ignore-missing-imports --show-error-codes

coregraph-test:
	$(PY) -m pytest -q

coregraph-coverage:
	$(PY) -m coverage erase
	$(PY) -m coverage run --source=coregraph,scripts/coregraph -m pytest -q
	$(PY) -m coverage report --include='coregraph/contracts/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/routing/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/objectives/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/experiments/pilot.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/experiments/manifest_conversion.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/experiments/scenario_manifests.py' --fail-under=80
	$(PY) -m coverage report --include='coregraph/experiments/canonical_recovery.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/experiments/protocol_registry.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/data/leakage.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/evaluation/statistics.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/evidence/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/experiments/contract_splits.py' --fail-under=85
	$(PY) -m coverage report --include='coregraph/diagnostics/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/benchmarks/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/baselines/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/resources/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/theory/*' --fail-under=85
	$(PY) -m coverage report --include='coregraph/evaluation/selective.py,coregraph/evaluation/resources.py,coregraph/evaluation/counterfactual.py' --fail-under=85
	$(PY) -m coverage report --include='scripts/coregraph/evaluate_pilot_gate.py' --fail-under=85
	# The runner file retains the separately authorised empirical execution
	# branch; this gate covers its exercised plan/validate-only surface.
	$(PY) -m coverage report --include='scripts/coregraph/run_saved_output_pilot.py' --fail-under=50

coregraph-local-gates: coregraph-compile coregraph-lint coregraph-typecheck coregraph-test coregraph-coverage
	$(PY) scripts/coregraph/check_theory_numerically.py
	$(PY) scripts/coregraph/validate_theory_status.py
	$(PY) scripts/coregraph/validate_notebooks.py
	$(PY) scripts/coregraph/run_synthetic_method_checks.py
	$(PY) scripts/coregraph/run_coregraph_smoke.py
	$(PY) scripts/coregraph/validate_paper_skeleton.py
	$(PY) scripts/coregraph/build_anonymous_release.py
	$(PY) scripts/coregraph/audit_anonymous_release.py
	$(PY) scripts/coregraph/hash_frozen_assets.py --verify

coregraph-evidence-cache-check:
	$(PY) scripts/coregraph/check_level4_evidence_cache.py
	$(PY) scripts/coregraph/validate_level4_row_scopes.py

coregraph-pilot-plan:
	$(PY) scripts/coregraph/orchestrate_level4.py pilot-plan

coregraph-pilot-validate:
	$(PY) scripts/coregraph/orchestrate_level4.py pilot-validate

coregraph-full-plan:
	$(PY) scripts/coregraph/orchestrate_level4.py full-plan

# This target intentionally exits non-zero until a validated run import exists.
coregraph-analysis:
	$(PY) scripts/coregraph/orchestrate_level4.py analysis

coregraph-paper:
	$(PY) scripts/coregraph/generate_level4_figures.py
	$(PY) scripts/coregraph/validate_paper_skeleton.py
	$(PY) scripts/audit_cross_paper_overlap.py
	$(PY) scripts/coregraph/build_level4_paper.py

coregraph-release:
	$(PY) scripts/coregraph/build_level4_release.py
	$(PY) scripts/coregraph/validate_level4_release.py

coregraph-cleanroom:
	$(PY) scripts/coregraph/validate_level4_release.py --cleanroom

coregraph-level4-validate: coregraph-compile coregraph-lint coregraph-typecheck coregraph-test
	$(PY) scripts/coregraph/build_level4_artifacts.py
	$(PY) scripts/coregraph/check_level4_evidence_cache.py
	$(PY) scripts/coregraph/orchestrate_level4.py pilot-validate
	$(PY) scripts/coregraph/check_theory_numerically.py
	$(PY) theory/coregraph_level4/executable_checks.py
	$(PY) scripts/coregraph/run_synthetic_method_checks.py
	$(PY) scripts/coregraph/run_coregraph_smoke.py
	$(PY) scripts/coregraph/validate_notebooks.py
	$(PY) scripts/coregraph/validate_paper_skeleton.py
	$(PY) scripts/audit_cross_paper_overlap.py
	$(PY) scripts/coregraph/build_level4_paper.py
	$(PY) scripts/coregraph/validate_level4_release.py
	$(PY) scripts/coregraph/hash_frozen_assets.py --verify
