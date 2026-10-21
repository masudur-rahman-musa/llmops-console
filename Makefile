.PHONY: help setup run up down rebuild all gateway console test fast test-slowest lint typecheck format verify security eval infra-check k8s-check k8s-aws-check k8s-server-check server-secrets server-helpdesk-data k8s-gpu-check aws-evidence aws-evidence-cost tradeoff tradeoff-option tradeoff-gpu tradeoff-gpu-cost migrate purge-residency refresh-rollup secrets-encrypt secrets-decrypt secrets-check load

# The console's checks run through pnpm, and are skipped only in a checkout that
# has no web, so the intended gate stays visible in this file. The
# `else` branch is what makes this a skip-when-absent and not a skip-on-failure:
# once the directory exists, a failing pnpm step fails the target.
define console
@if [ -d web ]; then pnpm -C web $(1); \
else echo "skip: no web yet - $(1)"; fi
endef
# The Python project lives in `backend/` (its `pyproject.toml`, venv and tests).
# `UV_ROOT` keeps the shell in the repo root - for targets that also name root
# paths like `infra/tests/`, `infra/load/` or `backend/evals/` - while using
# backend's project and venv. `UV_BK` moves the shell into `backend/` for the
# tools whose config is read relative to the working directory (mypy's `files`,
# pytest's `testpaths`). Ruff runs from the root over the whole repo and names
# `backend/pyproject.toml` with `--config`, so its paths resolve from the root.
UV_ROOT = uv run --project backend
UV_BK = uv run --directory backend
help:       ## Show all targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
setup:      ## Install dependencies and pre-commit hooks
	uv sync --project backend && $(UV_ROOT) pre-commit install
	$(call console,install)
up:         ## Start everything in Docker: backing services, gateway and the console on 3001
	docker compose --profile app up -d --build
down:       ## Stop the whole Docker stack, keeping its data
	docker compose --profile app down
# `make rebuild` or `make rebuild all`: rebuild and restart the gateway and the console,
# never the databases. `make rebuild gateway` (or console) does just that one.
# The words after `rebuild` are empty targets so make accepts them.
REBUILD_WHAT = $(filter-out rebuild,$(MAKECMDGOALS))
rebuild:    ## Rebuild gateway and console, never the databases: make rebuild [all|gateway|console]
	docker compose --profile app up -d --build --no-deps $(if $(filter-out all,$(REBUILD_WHAT)),$(filter-out all,$(REBUILD_WHAT)),gateway console)
all gateway console:
	@:
run:        ## Run the project locally
	docker compose up -d && $(UV_ROOT) python -c 'from app.main import serve; serve(reload=True)'
# The packages that carry a guarantee rather than a feature. They are gated at
# 100% on top of the 80% the whole tree has to clear. This reads the coverage
# data pytest just wrote instead of re-running with `--cov=<path>`, because that
# flag overrides `[tool.coverage.run] source` and would silently unmeasure
# everything outside the path.
COVERED_100 = backend/app/model_routing/*,backend/app/core/validation.py,backend/app/limits/*
# `make test` (and so `make verify`) points the three OPSCONSOLE_TEST_* URLs at
# the compose host ports (Postgres 15432, Redis 16379 db 15, Qdrant 16333), so
# the full suite really runs the 116 backing-service tests instead of skipping
# them. This needs `docker compose up -d` first; without a reachable service
# those tests FAIL rather than skip, which is the point - a green `make verify`
# now means the backing tests ran. The names are OPSCONSOLE_TEST_*, never the
# OPSCONSOLE_* ones Settings reads, so they reach only the test fixtures
#. The password matches compose's default (POSTGRES_PASSWORD
# defaults to opsconsole in docker-compose.yml and backend/.env.demo). The three
# names are written out in each recipe, not hidden behind a make variable, so a
# static reader (and its guard test) sees the suite is pointed at them. Each
# keeps a value already set (CI's service ports) and falls back to compose's.
test:       ## Run tests with the coverage gate (backing services must be up)
	OPSCONSOLE_TEST_DATABASE_URL=$${OPSCONSOLE_TEST_DATABASE_URL:-postgresql://opsconsole:opsconsole@localhost:15432/opsconsole} \
	OPSCONSOLE_TEST_REDIS_URL=$${OPSCONSOLE_TEST_REDIS_URL:-redis://localhost:16379/15} \
	OPSCONSOLE_TEST_QDRANT_URL=$${OPSCONSOLE_TEST_QDRANT_URL:-http://localhost:16333} \
	$(UV_ROOT) pytest backend/tests -n auto --cov --cov-fail-under=80
	$(UV_ROOT) coverage report --include='$(COVERED_100)' --fail-under=100
	$(call console,test)
# The everyday run: every test not marked `slow`, across one
# xdist worker per core (`-n auto`), with NO OPSCONSOLE_TEST_* URL set, so the
# Postgres, Redis and Qdrant tests skip and nothing waits on a service. This is
# the fast loop while coding; CI and `make verify` still run every test. It
# carries no coverage gate on purpose - the floors belong to the full `test`.
fast:       ## Everyday fast run: non-slow tests in parallel, no backing services
	$(UV_ROOT) pytest backend/tests -n auto -m "not slow"
# Prints the slowest tests and fixtures (setup/call/teardown each counted) so a
# new slow spot is seen and marked `slow` or fixed. Reads the same tree `test`
# does, with the backing URLs set so the backing tests are timed too, not skipped.
test-slowest: ## Print the 30 slowest tests and fixtures
	OPSCONSOLE_TEST_DATABASE_URL=postgresql://opsconsole:opsconsole@localhost:15432/opsconsole \
	OPSCONSOLE_TEST_REDIS_URL=redis://localhost:16379/15 \
	OPSCONSOLE_TEST_QDRANT_URL=http://localhost:16333 \
	$(UV_ROOT) pytest backend/tests --durations=30
lint:       ## Lint
	$(UV_ROOT) ruff check --config backend/pyproject.toml .
	$(call console,lint)
typecheck:  ## Static types
	$(UV_BK) mypy
	$(call console,typecheck)
format:     ## Auto-format
	$(UV_ROOT) ruff format --config backend/pyproject.toml .
# The three security checks CI runs, with the same tool versions, so a green
# laptop run means a green CI `security` job (it runs only this target).
# - gitleaks scans every commit of the mounted repo. It runs through docker
#   because nobody installs it locally. The image is pinned by tag AND digest,
#   and the tag is written out literally: `infra/tests/ci/test_ci_workflow.py`
#   compares it to the pre-commit hook's `rev:`. No
#   staging-area flag: those belong to the local hook.
# - pip-audit reads the locked Python set. `--locked` audits what the repo pins,
#   `--all-groups` takes the dev and test trees too, `--no-emit-project` keeps
#   the unhashed editable entry out of a `--require-hashes` read. The file lands
#   in the gitignored `backend/scratch/`, so a run leaves nothing at the root.
# - pnpm audit covers the console.
security:   ## Secret scan over all history, Python and console dependency audits
	docker run --rm -v "$(CURDIR):/repo" -w /repo zricethezav/gitleaks:v8.24.3@sha256:e1b35e12a8c6fa8901f060459cfb6b2fc4c484d3afbe3b029733a3bbfab07055 git --redact --exit-code 1 .
	mkdir -p backend/scratch
	uv export --project backend --format requirements-txt --locked --no-emit-project --all-groups -o backend/scratch/requirements-audit.txt
	uvx --from pip-audit==2.9.0 pip-audit --strict --require-hashes --disable-pip -r backend/scratch/requirements-audit.txt
	$(call console,audit)
verify: format lint typecheck test security   ## The full gate, in order
# The ledger's schema is applied by a deploy STEP, never on app start: replicas
# coming up together would otherwise race each other over the same DDL, and
# nobody would get to read the migration before it ran. Both targets answer a
# non-zero code when OPSCONSOLE_DATABASE_URL is unset, so a pipeline cannot
# mistake "nothing was configured" for "nothing needed doing".
migrate:    ## Apply the ledger migrations (safe to run twice)
	$(UV_ROOT) ops-migrate
purge-residency: ## Drop residency log partitions past the retention
	$(UV_ROOT) ops-purge-residency
refresh-rollup: ## Refresh the daily cost rollup the console reads
	$(UV_ROOT) ops-refresh-rollup
# The eval gate: the help desk's 33 conversation cases against the
# real help desk agent, with this gateway beside it and the real cheap model
# (`agent-default`) answering, so `ANTHROPIC_API_KEY` must be set (in the shell,
# as CI does, or in `backend/.env`). One run, then one gate check against
# `backend/evals/conversation-baseline.json`; either failing fails the target.
# HELPDESK_DIR is the help desk checkout (CI checks it out to `helpdesk/`).
# Each run makes two throwaway gateway keys, one per help desk tenant: the
# digests go to the gateway, the keys to the help desk, and nothing is stored.
# The judge uses the first key. A run's own exit code is not the verdict: failed
# cases are the gate's input, and a run that wrote no report fails its gate
# check instead. The gateway's response cache is cleared first, so no answer
# from an older prompt is served again. A help desk scheduler left running on a
# laptop is stopped first: its review job writes the same rows as the reset.
# RUN_ID=<id> continues a stopped run: cases already saved under that id are
# skipped, and the reports folder is kept. Without it the folder starts empty.
# The judge is held to the labels only once `calibration.json` exists.
HELPDESK_DIR ?= ../ai-help-desk-governed-access
EVAL_HELPDESK = $(abspath $(HELPDESK_DIR))
EVAL_CASES = $(EVAL_HELPDESK)/backend/evals/conversations/cases.jsonl
EVAL_REPORTS = backend/evals/reports/local
EVAL_CALIBRATION = backend/evals/labels/calibration.json
EVAL_HELPDESK_COMPOSE = docker compose --project-directory $(EVAL_HELPDESK) -f $(EVAL_HELPDESK)/docker-compose.yml -f infra/evals/helpdesk.compose.yml --profile app
eval:       ## The eval gate: the help desk's 33 conversations on the real cheap model, then the gate
	@nord=$$(openssl rand -hex 32); cuan=$$(openssl rand -hex 32); judge=$$(openssl rand -hex 32); \
	nord_digest=$$(printf '%s' "$$nord" | openssl dgst -sha256 -r | cut -d' ' -f1); \
	cuan_digest=$$(printf '%s' "$$cuan" | openssl dgst -sha256 -r | cut -d' ' -f1); \
	judge_digest=$$(printf '%s' "$$judge" | openssl dgst -sha256 -r | cut -d' ' -f1); \
	export OPSCONSOLE_API_KEYS="nordlicht:$$nord_digest,cuan:$$cuan_digest,evals:$$judge_digest"; \
	export HELPDESK_GATEWAY_API_KEYS="nordlicht:$$nord,cuan:$$cuan"; \
	export OPSCONSOLE_EVAL_JUDGE_KEY="$$judge"; \
	export HELPDESK_DIR=$(EVAL_HELPDESK); \
	stack="docker compose -f docker-compose.yml -f infra/evals/compose.yml$${ANTHROPIC_API_KEY:+ -f infra/evals/compose.key.yml}"; \
	$$stack up -d --build --wait gateway || exit $$?; \
	$$stack exec -T gateway ops-migrate || exit $$?; \
	$$stack exec -T redis sh -c 'redis-cli --scan --pattern "response:*" | xargs -r redis-cli del' >/dev/null || exit $$?; \
	(cd $(EVAL_HELPDESK) && docker compose up -d --wait postgres redis qdrant) || exit $$?; \
	$(EVAL_HELPDESK_COMPOSE) stop helpdesk-scheduler || exit $$?; \
	infra/evals/prepare_helpdesk.sh $(EVAL_HELPDESK) || exit $$?; \
	$(EVAL_HELPDESK_COMPOSE) up -d --build --wait helpdesk-mcp-server helpdesk-agent-api || exit $$?; \
	[ -n "$(RUN_ID)" ] || rm -rf $(EVAL_REPORTS); \
	status=0; \
	$(UV_ROOT) ops-eval conversations --target http://localhost:8000 --cases $(EVAL_CASES) \
	  --run-kind ci --judge-url http://localhost:8080 --out $(EVAL_REPORTS) \
	  $(if $(RUN_ID),--run-id $(RUN_ID)) || status=1; \
	$(UV_ROOT) ops-eval gate --report $(EVAL_REPORTS)/report.json \
	  --baseline backend/evals/conversation-baseline.json \
	  $(if $(wildcard $(EVAL_CALIBRATION)),--calibration $(EVAL_CALIBRATION)) \
	  --comment $(EVAL_REPORTS)/comment.md || status=1; \
	exit $$status
# The published comparison (the design). Pure: it reads a manifest and the
# report.json each option names, so it needs no stack and no key. MANIFEST and
# OUT are overridable because the measured run builds its table from its own
# evidence directory.
MANIFEST ?= infra/tradeoff/tradeoff.json
OUT ?= docs
tradeoff:   ## Build docs/tradeoff.md and docs/tradeoff.json from a manifest of runs
	$(UV_ROOT) ops-eval tradeoff --manifest $(MANIFEST) --out $(OUT)
# The three measurement steps behind that table, each a thin call into
# `infra/scripts/tradeoff_run.sh`. No `||` in any recipe: a measurement run that failed
# must fail its target, or the table is built from whatever the last run left on
# disk. No workflow names them either: the GPU half applies
# real infrastructure, which a project rule keeps in the owner's own shell.
#
# OPTION, SHA, START and END have no defaults on purpose. A defaulted option id
# would measure the wrong row; a defaulted sha would deploy the wrong image.
tradeoff-option: ## Measure one option (OPTION=<id> TARGET=<eval target url>)
	infra/scripts/tradeoff_run.sh option $(OPTION) --target $(TARGET)
tradeoff-gpu: ## Apply the GPU host, measure gpu-vllm, capture, destroy (SHA=<commit>)
	infra/scripts/tradeoff_run.sh gpu $(SHA)
tradeoff-gpu-cost: ## Write docs/tradeoff-run/cost.json from Cost Explorer (START=, END=)
	infra/scripts/tradeoff_run.sh cost $(START) $(END)
# The k6 load run: 50 RPS for ten minutes against the real gateway over the
# fixed-delay mock provider (`infra/load/compose.yml`).
#
# One logical shell line, so the throwaway key lives in one shell's environment
# and nowhere else: it is generated here, its digest is handed to the gateway and
# the key itself to k6, and neither is written to a file. The stack comes up and
# is migrated before k6 starts, since every served request writes the ledger.
# k6's exit code is captured straight after it runs and is the target's own, so
# a breached threshold fails `make load` and the workflow that calls it.
# The results directory is opened to every uid because the k6 image runs as its
# own non-root user, which on a Linux runner does not own the bind mount.
LOAD_COMPOSE = docker compose -f docker-compose.yml -f infra/load/compose.yml --profile load
load:       ## k6 load test, 50 RPS for 10 minutes, against a mock provider
	@mkdir -p infra/load/results
	@chmod a+rwx infra/load/results
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
	bash infra/scripts/server_secrets.sh
server-helpdesk-data: ## Load the Helpdesk's corpus, demo tenants and vectors onto the node (once)
	bash infra/scripts/server_helpdesk_data.sh
# The GPU measurement build, with the same kubeconform flags as `k8s-check`. It
# is standalone (the root build does not include it), so nothing else checks it.
k8s-gpu-check: ## Build the GPU measurement kustomization and validate it (no apply)
	kustomize build infra/k8s/gpu | kubeconform -strict -summary -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
# The one-off AWS evidence run and its cost capture, from the owner's shell only:
# `run` applies real infrastructure, so no workflow calls either target.
# docs/aws-evidence/README.md lists what has to be in place first.
aws-evidence: ## Apply, deploy, capture and destroy the AWS stack once (SHA=<commit>)
	infra/scripts/aws_evidence.sh run "$(SHA)"
aws-evidence-cost: ## Write docs/aws-evidence/cost.json (START=YYYY-MM-DD END=YYYY-MM-DD)
	infra/scripts/aws_evidence.sh cost "$(START)" "$(END)"

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
# Pinned by `infra/tests/infra/test_secrets.py`
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
	$(UV_ROOT) pytest -q infra/tests/infra/test_secrets.py
