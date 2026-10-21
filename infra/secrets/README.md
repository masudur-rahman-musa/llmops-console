# Secrets

Nine Secrets the cluster needs and this repository does not contain. What is
here is the contract for each one, the policy that encrypts them, and the
workflow. What is never here is a private key or a filled-in value.

Three kinds of file live in this directory:

| pattern | contents | committed |
|---|---|---|
| `*.example.yaml` | the contract: name, namespace, key names, every value `REPLACE_ME` | yes |
| `*.dec.yaml` | your filled-in copy, plaintext | **no** - gitignored |
| `*.enc.yaml` | the same file after `sops`, values as `ENC[AES256_GCM,data:...]` | yes |

`../../.sops.yaml` is what decides the difference. Its one creation rule matches
`infra/secrets/*.enc.yaml` and encrypts exactly `^(data|stringData)$`, so a
Secret's name, namespace and kind stay readable and only its values become
ciphertext. The rule deliberately does not match `*.example.yaml`.

Nothing in this directory is a kustomize input. `infra/k8s` lists no path under
it and the build emits no `Secret` at all; these files are applied by hand (and
by CD, in `.github/workflows/deploy.yml`) before the manifests that name them. Until they exist an
apply fails on the missing Secret, which is the intended direction - a
placeholder with a working password in git would come up running and never
mention itself again.

## The nine contracts

| Secret | namespace | template |
|---|---|---|
| `postgres-credentials` | `platform` | `postgres-credentials.example.yaml` |
| `traefik-basic-auth` | `platform`, `apps` | `traefik-basic-auth.example.yaml` |
| `gateway-secrets` | `apps` | `gateway-secrets.example.yaml` |
| `helpdesk-secrets` | `apps` | `helpdesk-secrets.example.yaml` |
| `ghcr-pull` | `apps` | `ghcr-pull.example.yaml` |
| `console-secrets` | `apps` | `console-secrets.example.yaml` |
| `grafana-admin` | `monitoring` | `grafana-admin.example.yaml` |
| `backup-secrets` | `platform` | `backup-secrets.example.yaml` |
| `alertmanager-secrets` | `monitoring` | `alertmanager-secrets.example.yaml` |

The key names and who reads them are in `../k8s/README.md`'s secrets table; the
templates carry the same sets, and `infra/tests/infra/test_secrets.py` fails if the two
ever disagree or if either drifts from the `secretRef` names the build emits.

`traefik-basic-auth` is one file of two documents because a Traefik `Middleware`
reads its Secret from its own namespace, so each routing namespace needs its own
copy of the same credentials.

Two key names in `gateway-secrets` and one in `helpdesk-secrets` are endpoints
rather than credentials (`OPSCONSOLE_OTLP_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`,
`LANGFUSE_HOST`). They sit here because a pod reads exactly two sources
- its ConfigMap and its Secret - and each of these travels with a credential it
would otherwise be separated from.

## Generating the key

Once, on the machine that will encrypt:

```sh
age-keygen -o ~/.config/sops/age/keys.txt
```

That prints the PUBLIC key (`age1...`) and writes the private key to the file.
Put the public key into `../../.sops.yaml` in place of `REPLACE_ME` and commit
that; a recipient is not a secret. The private key goes into a password manager
and nowhere else.

`keys.txt`, `*.agekey` and `*.dec.yaml` are gitignored repo-wide, and a test
asserts `git ls-files` matches none of them. That is a backstop, not the plan:
the private key should not be in the working tree at all.

For CD, the same private key is stored as the `SOPS_AGE_KEY` GitHub secret. The
deploy workflow (`.github/workflows/deploy.yml`) exports it and runs
`sops --decrypt` on each `*.enc.yaml` immediately before `kubectl apply`, so the
decrypted form exists only in the runner's memory for the length of one job. An
empty glob fails that step rather than applying the manifests without them.

## Encrypting

```sh
cp gateway-secrets.example.yaml gateway-secrets.dec.yaml
$EDITOR gateway-secrets.dec.yaml          # replace every REPLACE_ME
make secrets-encrypt                      # writes gateway-secrets.enc.yaml
rm gateway-secrets.dec.yaml
```

`make secrets-encrypt` encrypts every `*.dec.yaml` in this directory and refuses
to run at all when `sops` is not on `PATH`, rather than reporting success over a
no-op. `make secrets-decrypt` is the reverse, for editing an existing secret.
`make secrets-check` runs the guard, and needs neither `sops` nor `age`.

Deleting the `.dec.yaml` afterwards is the habit worth keeping. It is gitignored,
so leaving it is not a leak into history, but it is a plaintext credential on
disk with no expiry.

## the node

The live node does not fill templates by hand. `make server-secrets` writes six
`*.dec.yaml` straight from the owner's state file outside the repo, so the live
passwords and keys never change: `postgres-credentials`, `gateway-secrets`,
`console-secrets`, `helpdesk-secrets`, and `traefik-basic-auth` as two files
(`traefik-basic-auth-platform` and `traefik-basic-auth-apps`), because each
`*.enc.yaml` is read as one document. Then `make secrets-encrypt`, delete the
`*.dec.yaml`, and merge the six `*.enc.yaml` through a pull request. The deploy
workflow applies them. the project notes has the whole procedure, and
`infra/tests/infra/test_server_secrets.py` pins the script.

## Rotating one value

Decrypt, edit, re-encrypt, apply, then restart the workloads that read it -
`envFrom` is resolved at pod start, so a rolled Secret reaches a running pod
never.

```sh
make secrets-decrypt
$EDITOR gateway-secrets.dec.yaml
make secrets-encrypt
rm gateway-secrets.dec.yaml
kubectl apply -f gateway-secrets.enc.yaml   # human-only; see the repo's hard rules
kubectl -n apps rollout restart deployment/gateway
```

Two rotations are not one command. `postgres-credentials`' user and password are
read by the `postgres` image on FIRST START only, so changing them there is a new
database rather than a new password: rotate the role inside Postgres with
`ALTER ROLE`, then update this Secret and `gateway-secrets`'
`OPSCONSOLE_DATABASE_URL` to match. And the Helpdesk's gateway keys are pairs -
`helpdesk-secrets`' `GATEWAY_API_KEYS` holds one issued key per tenant
(`tenant:key,tenant:key`), `gateway-secrets`' `OPSCONSOLE_API_KEYS` holds each
one's sha256 digest - so a tenant's two halves move together or every Helpdesk
model call for that tenant is a 401.

## Rotating the age key

Generate the new key, add the new recipient to `.sops.yaml` beside the old one,
run `sops updatekeys` on each `*.enc.yaml`, verify a decrypt with the new key
alone, then remove the old recipient and `updatekeys` again. Removing the old
recipient first locks you out of every file.
