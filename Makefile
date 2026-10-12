.PHONY: help setup run test select-tests test-adjacent lint typecheck format verify eval infra-check k8s-check k8s-aws-check k8s-server-check server-secrets server-helpdesk-data k8s-gpu-check aws-evidence aws-evidence-cost tradeoff tradeoff-option tradeoff-gpu tradeoff-gpu-cost migrate purge-residency refresh-rollup secrets-encrypt secrets-decrypt secrets-check load

# The console's checks run through pnpm, and are skipped only in a checkout that
# has no services/console, so the intended gate stays visible in this file. The
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
	$(call console,test)
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
# The ledger's schema is applied by a deploy STEP, never on app start: replicas
# coming up together would otherwise race each other over the same DDL, and
# nobody would get to read the migration before it ran. Both targets answer a
# non-zero code when OPSCONSOLE_DATABASE_URL is unset, so a pipeline cannot
# mistake "nothing was configured" for "nothing needed doing".
migrate:    ## Apply the ledger migrations (safe to run twice)
	uv run ops-migrate
purge-residency: ## Drop residency log partitions past the retention
	uv run ops-purge-residency
refresh-rollup: ## Refresh the daily cost rollup the console reads
	uv run ops-refresh-rollup
eval:       ## Project evals: golden set vs project-evals/baseline.json
	uv run ops-eval run --target http://localhost:8080 --set "$$GOLDEN_SET" --tenant evals --out project-evals/reports/local
# The published comparison (the design). Pure: it reads a manifest and the
# report.json each option names, so it needs no stack and no key. MANIFEST and
# OUT are overridable because the measured run builds its table from its own
# evidence directory.
MANIFEST ?= project-evals/tradeoff.json
OUT ?= docs
tradeoff:   ## Build docs/tradeoff.md and docs/tradeoff.json from a manifest of runs
	uv run ops-eval tradeoff --manifest $(MANIFEST) --out $(OUT)
# The three measurement steps behind that table, each a thin call into
# `scripts/tradeoff_run.sh`. No `||` in any recipe: a measurement run that failed
# must fail its target, or the table is built from whatever the last run left on
# disk. No workflow names them either: the GPU half applies
# real infrastructure, which a project rule keeps in the owner's own shell.
#
# OPTION, SHA, START and END have no defaults on purpose. A defaulted option id
# would measure the wrong row; a defaulted sha would deploy the wrong image.
tradeoff-option: ## Measure one option (OPTION=<id> TARGET=<eval target url>)
	scripts/tradeoff_run.sh option $(OPTION) --target $(TARGET)
tradeoff-gpu: ## Apply the GPU host, measure gpu-vllm, capture, destroy (SHA=<commit>)
	scripts/tradeoff_run.sh gpu $(SHA)
tradeoff-gpu-cost: ## Write docs/tradeoff-run/cost.json from Cost Explorer (START=, END=)
	scripts/tradeoff_run.sh cost $(START) $(END)
# The k6 load run: 50 RPS for ten minutes against the real gateway over the
# fixed-delay mock provider (`docker-compose.load.yml`).
#
# One logical shell line, so the throwaway key lives in one shell's environment
# and nowhere else: it is generated here, its digest is handed to the gateway and
# the key itself to k6, and neither is written to a file. The stack comes up and
# is migrated before k6 starts, since every served request writes the ledger.
# k6's exit code is captured straight after it runs and is the target's own, so
# a breached threshold fails `make load` and the workflow that calls it.
# The results directory is opened to every uid because the k6 image runs as its
# own non-root user, which on a Linux runner does not own the bind mount.
LOAD_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.load.yml --profile load
load:       ## k6 load test, 50 RPS for 10 minutes, against a mock provider
	@mkdir -p project-evals/load/results
	@chmod a+rwx project-evals/load/results
	@key=$$(openssl rand -hex 32); \
	digest=$$(printf '%s' "$$key" | openssl dgst -sha256 -r | cut -d' ' -f1); \
	export LOAD_API_KEY="$$key"; \
	export OPSCONSOLE_API_KEYS="load:$$digest"; \
	$(LOAD_COMPOSE) up -d --build --wait gateway mock-provider || exit $$?; \
	$(LOAD_COMPOSE) exec -T gateway ops-migrate || exit $$?; \
	$(LOAD_COMPOSE) run --rm k6; \
	status=$$?; \
	exit $$status
# Two halves, each on its own line so a failure in either is the target's exit
# code. The terraform half runs for real; the kustomize half is a delegation to
# `k8s-check`, still written as a conditional rather than `A && B || C`, which
# would report success the moment terraform failed. The conditional is kept now
# that infra/k8s does carry a kustomization, so a checkout that has not got one
# skips rather than failing.
# `-backend=false` keeps `validate` off Terraform Cloud: the same init CI runs.
infra-check: ## Validate infrastructure as code (no apply)
	terraform -chdir=infra/terraform/hetzner fmt -check -recursive
	terraform -chdir=infra/terraform/hetzner init -backend=false -input=false
	terraform -chdir=infra/terraform/hetzner validate
	terraform -chdir=infra/terraform/aws fmt -check -recursive
	terraform -chdir=infra/terraform/aws init -backend=false -input=false
	terraform -chdir=infra/terraform/aws validate
	terraform -chdir=infra/terraform/aws-gpu fmt -check -recursive
	terraform -chdir=infra/terraform/aws-gpu init -backend=false -input=false
	terraform -chdir=infra/terraform/aws-gpu validate
	@if [ -f infra/k8s/kustomization.yaml ]; then $(MAKE) k8s-check; \
	else echo "skip: no infra/k8s/kustomization.yaml yet - kustomize build"; fi
# The kustomize half on its own target, so CI can run it without installing a
# terraform toolchain it has no use for, and so the pipeline is written in
# exactly one place for the two callers to share.
# `-schema-location default` keeps the stock schemas for core kinds alongside
# the CRD catalogue, which is where cert-manager's and Traefik's own kinds are
# resolved from once they land. `-strict` refuses an unknown field. There is
# deliberately no `-ignore-missing-schemas`: it would turn a kind kubeconform
# cannot resolve into a green tick, which is the one failure this step exists
# to produce.
k8s-check:  ## Build the kustomization and validate it against the schemas (no apply)
	kustomize build infra/k8s | kubeconform -strict -summary -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
# The AWS evidence overlay, with the same kubeconform flags as `k8s-check`. The
# relaxed load restrictor is needed because the overlay reuses files outside its
# own directory, the kind llama patch among them.
k8s-aws-check: ## Build the AWS evidence overlay and validate it (no apply)
	kustomize build --load-restrictor=LoadRestrictionsNone infra/k8s/aws | kubeconform -strict -summary -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
# the node's build, with the same kubeconform flags as
# `k8s-aws-check`. Applied only from the owner's shell, so CI is its one check.
k8s-server-check: ## Build the node overlay and validate it (no apply)
	kustomize build --load-restrictor=LoadRestrictionsNone infra/k8s/server | kubeconform -strict -summary -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
# Builds both images on this machine, loads them into the node's k3s and
# applies the overlay. The owner runs it; CI never does.
server-secrets: ## Write the node's six infra/secrets/*.dec.yaml to encrypt, generating the node-only values once
	bash scripts/server_secrets.sh
server-helpdesk-data: ## Load the Helpdesk's corpus, demo tenants and vectors onto the node (once)
	bash scripts/server_helpdesk_data.sh
# The GPU measurement build, with the same kubeconform flags as `k8s-check`. It
# is standalone (the root build does not include it), so nothing else checks it.
k8s-gpu-check: ## Build the GPU measurement kustomization and validate it (no apply)
	kustomize build infra/k8s/gpu | kubeconform -strict -summary -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
# The one-off AWS evidence run and its cost capture, from the owner's shell only:
# `run` applies real infrastructure, so no workflow calls either target.
# docs/aws-evidence/README.md lists what has to be in place first.
aws-evidence: ## Apply, deploy, capture and destroy the AWS stack once (SHA=<commit>)
	scripts/aws_evidence.sh run "$(SHA)"
aws-evidence-cost: ## Write docs/aws-evidence/cost.json (START=YYYY-MM-DD END=YYYY-MM-DD)
	scripts/aws_evidence.sh cost "$(START)" "$(END)"

# The three secrets targets. `sops` is NOT a build dependency of this repository
# and the age key is never in it, so each of the two that shell out to `sops`
# starts by refusing when the binary is absent, with a message naming it - a bare
# `sops --encrypt` inside a loop exits 127 with `command not found`, and a loop
# over an empty glob reports success over a no-op, which is an operator believing
# four files were written when none were. Written as `if ! command -v` and not as
# `A && B || C`, which reports success the moment B fails.
#
# `--output` rather than a shell redirect: `> out` creates the file before sops
# runs, so a failed encrypt would leave an empty `.enc.yaml` behind that looks
# like a result.
#
# Pinned by `tests/infra/test_secrets.py`
# (`test_the_sops_targets_refuse_to_run_when_sops_is_missing` and
# `test_the_sops_targets_refuse_to_report_success_over_an_empty_glob`).
SECRETS_DIR = infra/secrets
secrets-encrypt: ## Encrypt every infra/secrets/*.dec.yaml to its *.enc.yaml
	@if ! command -v sops >/dev/null 2>&1; then \
	  echo "sops is not on PATH: install sops before encrypting - $(SECRETS_DIR)/README.md" >&2; \
	  exit 1; \
	fi
	@found=0; \
	for plain in $(SECRETS_DIR)/*.dec.yaml; do \
	  [ -e "$$plain" ] || continue; \
	  found=1; \
	  sops --encrypt --filename-override "$${plain%.dec.yaml}.enc.yaml" \
	    --output "$${plain%.dec.yaml}.enc.yaml" "$$plain" || exit 1; \
	  echo "encrypted $$plain"; \
	done; \
	if [ "$$found" = 0 ]; then \
	  echo "nothing to encrypt: no $(SECRETS_DIR)/*.dec.yaml - copy a *.example.yaml and fill it in" >&2; \
	  exit 1; \
	fi
secrets-decrypt: ## Decrypt every infra/secrets/*.enc.yaml to its *.dec.yaml
	@if ! command -v sops >/dev/null 2>&1; then \
	  echo "sops is not on PATH: install sops before decrypting - $(SECRETS_DIR)/README.md" >&2; \
	  exit 1; \
	fi
	@found=0; \
	for sealed in $(SECRETS_DIR)/*.enc.yaml; do \
	  [ -e "$$sealed" ] || continue; \
	  found=1; \
	  sops --decrypt --output "$${sealed%.enc.yaml}.dec.yaml" "$$sealed" || exit 1; \
	  echo "decrypted $$sealed"; \
	done; \
	if [ "$$found" = 0 ]; then \
	  echo "nothing to decrypt: no $(SECRETS_DIR)/*.enc.yaml yet" >&2; \
	  exit 1; \
	fi
# The guard, which needs neither sops nor age: it asserts on file structure, on
# `git ls-files` and on the kustomize build, so it is the one secrets target that
# runs on any machine and in CI.
secrets-check: ## Check the secrets policy and contracts (no sops, no age needed)
	uv run pytest -q tests/infra/test_secrets.py
