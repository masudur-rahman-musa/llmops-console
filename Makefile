.PHONY: help setup run test lint typecheck format verify eval infra-check
help:       ## Show all targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
setup:      ## Install dependencies and pre-commit hooks
	uv sync && uv run pre-commit install && pnpm -C services/console install
run:        ## Run the project locally
	docker compose up -d && uv run uvicorn services.gateway.main:app --reload --port 8080
test:       ## Run tests with the coverage gate
	uv run pytest -q --cov=packages/opsconsole --cov-fail-under=80
lint:       ## Lint
	uv run ruff check . && pnpm -C services/console lint
typecheck:  ## Static types
	uv run mypy && pnpm -C services/console typecheck
format:     ## Auto-format
	uv run ruff format .
verify: format lint typecheck test   ## The full gate, in order
eval:       ## Project evals: golden set vs project-evals/baseline.json
	uv run ops-eval run --target http://localhost:8080 --set "$$GOLDEN_SET" --tenant evals --out project-evals/reports/local
infra-check: ## Validate infrastructure as code (no apply)
	terraform -chdir=infra/terraform/hetzner fmt -check && terraform -chdir=infra/terraform/hetzner validate && kustomize build infra/k8s | kubeconform
