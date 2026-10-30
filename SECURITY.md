# Security

## Reporting a problem

Please report a security issue privately, through GitHub's "Report a
vulnerability" form on this repository's Security tab. Do not open a public issue
for it. Say what you found, how to reproduce it, and what it lets someone do. This
is a portfolio project kept by one person, so replies are best effort.

## What the code does

**Keys.**

- Tenant keys and the admin key are stored only as sha256 digests
  (`OPSCONSOLE_API_KEYS`, `OPSCONSOLE_ADMIN_KEY_SHA256`). Comparison uses
  `hmac.compare_digest`.
- A key is bound to one tenant. A request whose `X-Tenant` does not match its key
  is refused, and the tenant is never read from the request body.
- The admin key is a separate key space. A tenant key gets a 401 on `/admin/*` and
  `/console/*`. With no admin digest set, nobody gets in.
- A budget override is an HMAC token signed with its own secret and bound to one
  tenant and one month. With no secret set, no override is honoured.

**Personal data.**

- When a tenant's rule says redact, ten kinds of personal data are replaced with
  placeholders before a request reaches a provider. Requests with no data class
  count as `confidential`, which the shipped tenants redact.
- Redaction maps live for one hour in Redis, readable only by the tenant that made
  them. The compose Redis keeps no disk copy.
- Traces carry the tenant and session, and at most a random user id, never a name
  or an email. Prompt text reaches Langfuse only for a tenant that turns
  `trace_content` on, and only after redaction.
- Redaction is not perfect. Its measured misses are published in the README and
  listed one by one in [docs/redaction-misses.md](docs/redaction-misses.md).

**Secrets.**

- No real secret is committed. `backend/.env.example` and `web/.env.example`
  list names only. Real values go in env files git ignores.
- For a cluster, secrets are SOPS files encrypted with age (`.sops.yaml`,
  `infra/secrets/`). Only the encrypted files may be committed.
- `make security` runs gitleaks over the whole git history, `pip-audit` over the
  locked Python packages and `pnpm audit` over the console. A gitleaks hook also
  runs before each commit.
- The build workflow scans each pushed image with Trivy and publishes its SBOM.
  Images run as non-root users.

**The console.**

- Login is one shared password, stored as a sha256 digest and checked in constant
  time. With no digest set, every login is refused.
- The session cookie is signed and set `HttpOnly`, `Secure` and `SameSite=Strict`.
- The admin key stays on the server. The browser never sees it, and never talks to
  the database or Prometheus.

## What is not covered

- There is no per-user login for the console, only the shared password.
- `/metrics` needs no key. In a cluster the ingress keeps it off the public port.
  Locally it is open on the gateway port.
- Local Grafana allows anonymous read-only viewers and listens on 127.0.0.1 only.
