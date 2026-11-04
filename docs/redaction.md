# Redaction

The gateway finds personal data in a request and swaps it for a placeholder before
any provider sees it. This page says what it finds, how it is scored and where it
runs.

## What it finds

It finds ten kinds of personal data. They are email addresses, phone numbers, IBANs,
card numbers, German tax IDs, Irish PPS numbers, IPv4 and IPv6 addresses, dates of
birth, and the names a tenant's operator has added to its own directory.

Sometimes the shape of a value is not enough. Then the value must also pass its own
checksum. Cards use Luhn. IBANs use mod-97. Tax IDs use ISO 7064 MOD 11,10. PPS
numbers use mod-23. So a sixteen-digit order number is not taken for a card.

A date counts as a date of birth only when a word such as "born" or "DOB" sits
close in front of it.

## How it is scored

The numbers in the README come from a held-out set of 600 generated cases. That is
50 values of each kind, written the ways people really write them. It also holds
100 look-alikes that must not be found, such as version numbers, MAC addresses and
file references.

The set is built with a fixed seed from the templates in
`backend/evals/redaction/heldout-templates.json`. Those templates were never used
to tune a detector.

The table reports F2. F2 counts recall more than precision. A missed value reaches
a model. A false find only hides a harmless string.

`backend/scripts/redaction_report.py` writes the README table and the list of
misses. A test fails if either one is out of date.

## Where it is weak

Most misses are phone numbers written with a `(0)`, German tax IDs, PPS numbers and
dates of birth. Every miss is listed in [redaction-misses.md](redaction-misses.md).

## Where it runs

Redaction runs in two places.

- **On each call.** When a tenant's rule says redact, every chat message and every
  embedding input is redacted before the request leaves. The answer carries
  `X-Gateway-Redacted`, which says whether a value was replaced.
- **When documents are loaded.** `POST /v1/redact` returns the redacted text and a
  `redaction_map_id`. The map is kept for one hour, for that tenant only. Send the
  id back as `X-Gateway-Rehydrate` on a chat call, and the real values are put back
  in the answer. This works for streamed answers too. An unknown or expired id is
  refused before any provider is called.

Within one request, the same value always gets the same placeholder. So `<EMAIL_1>`
means the same person in every message.
