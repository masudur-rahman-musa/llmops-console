.PHONY: help setup run test select-tests test-adjacent lint typecheck format verify eval infra-check

# The console arrives in a later milestone. Until then its checks are skipped
# rather than deleted, so the intended gate stays visible in this file. The
# `else` branch is what makes this a skip-when-absent and not a skip-on-failure:
# once the directory exists, a failing pnpm step fails the target.
define console
@if [ -d services/console ]; then pnpm -C services/console $(1); \
else echo "skip: no services/console yet - $(1)"; fi
endef
help:       ## Show all targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
setup:      ## Install dependencies and pre-commit hooks
	uv sync && uv run pre-commit install
	$(call console,install)
run:        ## Run the project locally
	docker compose up -d && uv run python -c 'from services.gateway.main import serve; serve(reload=True)'
# The packages that carry a guarantee rather than a feature. They are gated at
# 100% on top of the 80% the whole tree has to clear. This reads the coverage
# data pytest just wrote instead of re-running with `--cov=<path>`, because that
# flag overrides `[tool.coverage.run] source` and would silently unmeasure
# everything outside the path.
COVERED_100 = packages/opsconsole/opsconsole/routing/*
# The per-test coverage map /adjacent-regression selects from. Written by the
# same run that proves the suite green, so it can never be staler than it.
COV_DB = .coverage-baseline
test:       ## Run tests with the coverage gate
	COVERAGE_FILE=$(COV_DB) uv run pytest -q --cov --cov-context=test --cov-fail-under=80
	COVERAGE_FILE=$(COV_DB) uv run coverage report --include='$(COVERED_100)' --fail-under=100
select-tests: ## Print the tests whose recorded coverage touches this change
	@uv run python scripts/select_tests.py --explain
test-adjacent: ## Run only the tests whose coverage touches this change
	@uv run python scripts/select_tests.py | tr ' ' '\n' | xargs uv run pytest -q
lint:       ## Lint
	uv run ruff check .
	$(call console,lint)
typecheck:  ## Static types
	uv run mypy
	$(call console,typecheck)
format:     ## Auto-format
	uv run ruff format .
verify: format lint typecheck test   ## The full gate, in order
eval:       ## Project evals: golden set vs project-evals/baseline.json
	uv run ops-eval run --target http://localhost:8080 --set "$$GOLDEN_SET" --tenant evals --out project-evals/reports/local
infra-check: ## Validate infrastructure as code (no apply)
	terraform -chdir=infra/terraform/hetzner fmt -check && terraform -chdir=infra/terraform/hetzner validate && kustomize build infra/k8s | kubeconform
