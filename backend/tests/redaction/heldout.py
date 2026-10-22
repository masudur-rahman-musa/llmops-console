r"""The held-out redaction set: 600 cases built in memory from seed 82.

The 57-case fixture is the set the detectors were tuned against, so a score on it is a
test passing, not a measure. This set comes from sentence templates in
`backend/evals/redaction/heldout-templates.json` that no detector was written against, filled
with values made here: 50 positives for each of the 10 kinds and 100 look-alikes that must
not be found. No generated case is committed; every build with `HELDOUT_SEED` gives the same
600.

The values come in the forms people really write them, including forms the detectors may
not know (a spaced tax id, a German birth date cue, a grouped card). That is the point: a
miss here is a real gap, listed in `docs/redaction-misses.md`. There is no floor on this
set, because a floor would invite tuning to it.

Checksummed values are made with their own check digit worked out here, never by asking
the detectors' checks, so the generator cannot share a bug with what it measures.
`test_every_generated_checksummed_value_passes_its_checksum` holds them to the engine's
checks.
"""

from __future__ import annotations

import json
import random
import string
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from app.redaction import Kind

from .conftest import REPO_ROOT, Case, ExpectedSpan

HELDOUT_SEED: Final[int] = 82
TEMPLATES_PATH: Final[Path] = (
    REPO_ROOT / "backend" / "evals" / "redaction" / "heldout-templates.json"
)

POSITIVES_PER_KIND: Final[int] = 50
NEGATIVES: Final[int] = 100
SLOT: Final[str] = "{v}"

_PPSN_ALPHABET: Final[str] = "WABCDEFGHIJKLMNOPQRSTUV"
_MONTHS: Final[tuple[str, ...]] = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)  # fmt: skip

Value = tuple[str, tuple[str, ...]]
"""A generated value and the directory names its case is scored with."""

Framed = tuple[str, str, str]
"""Text written before the value, the value, and text after it. Only the value is a span:
the port after an address is not the address."""


def _templates() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    return document


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def _group(value: str, sizes: tuple[int, ...], separator: str = " ") -> str:
    parts: list[str] = []
    position = 0
    for size in sizes:
        parts.append(value[position : position + size])
        position += size
    if position < len(value):
        parts.append(value[position:])
    return separator.join(part for part in parts if part)


def _luhn_digit(body: str) -> str:
    total = 0
    for position, character in enumerate(reversed(body)):
        value = int(character)
        if position % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - total % 10) % 10)


def _email(rng: random.Random) -> str:
    first = rng.choice(("lena", "ciaran", "jo", "maximilian", "orla", "felix", "saoirse"))
    last = rng.choice(("krause", "doyle", "berger", "nolan", "vogt", "quinn"))
    domain = rng.choice(("contoso.de", "fabrikam.ie", "mail.northwind.eu", "tailspin-air.com"))
    style = rng.randrange(5)
    if style == 0:
        return f"{first}.{last}@{domain}"
    if style == 1:
        return f"{first}_{last}{rng.randrange(10, 99)}@{domain}"
    if style == 2:
        return f"{first}+billing@{domain}"
    if style == 3:
        return f"{first[0].upper()}.{last.capitalize()}@{domain.upper()}"
    return f"{last}-{first}@it.{domain}"


def _phone(rng: random.Random) -> str:
    style = rng.randrange(6)
    if style == 0:
        return f"+49{rng.choice(('151', '170', '30', '89'))}{_digits(rng, 8)}"
    if style == 1:
        return f"+353 {rng.choice(('85', '87', '1'))} {_digits(rng, 3)} {_digits(rng, 4)}"
    if style == 2:
        return f"0{rng.choice(('40', '221', '711'))} {_digits(rng, 7)}"
    if style == 3:
        return f"08{rng.choice('3567')} {_digits(rng, 3)} {_digits(rng, 4)}"
    if style == 4:
        return f"+49 (0){rng.choice(('30', '69'))} {_digits(rng, 4)} {_digits(rng, 4)}"
    return f"0043 1 {_digits(rng, 3)} {_digits(rng, 4)}"


_IBAN_SHAPES: Final[dict[str, tuple[int, int]]] = {
    "DE": (0, 18),
    "AT": (0, 16),
    "IE": (4, 14),
    "NL": (4, 10),
    "ES": (0, 20),
}
"""Each country's account part: leading bank letters, then digits."""


def _iban_check(country: str, bban: str) -> str:
    rearranged = bban + country + "00"
    number = int("".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged))
    return f"{98 - number % 97:02d}"


def _iban(rng: random.Random) -> str:
    country = rng.choice(sorted(_IBAN_SHAPES))
    letters, digits = _IBAN_SHAPES[country]
    bban = "".join(rng.choice(string.ascii_uppercase) for _ in range(letters)) + _digits(
        rng, digits
    )
    compact = f"{country}{_iban_check(country, bban)}{bban}"
    style = rng.randrange(3)
    if style == 0:
        return compact
    if style == 1:
        return _group(compact, (4,) * 8)
    return _group(compact.lower(), (4,) * 8) if country == "DE" else compact


def _card(rng: random.Random) -> str:
    style = rng.randrange(5)
    if style == 4:
        body = rng.choice(("34", "37")) + _digits(rng, 12)
        return _group(body + _luhn_digit(body), (4, 6, 5))
    prefix = rng.choice(("4", "51", "53", "55", "2221"))
    body = prefix + _digits(rng, 15 - len(prefix))
    number = body + _luhn_digit(body)
    if style == 0:
        return number
    if style == 1:
        return _group(number, (4, 4, 4, 4))
    if style == 2:
        return _group(number, (4, 4, 4, 4), "-")
    return f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"


def _steuer_check(first_ten: str) -> str:
    product = 10
    for character in first_ten:
        total = (int(character) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    return str((11 - product) % 10)


def _steuer_id(rng: random.Random) -> str:
    while True:
        digits = list(string.digits)
        rng.shuffle(digits)
        repeated = digits[0]
        first_ten = [*digits[:9], repeated]
        rng.shuffle(first_ten)
        if first_ten[0] != "0":
            break
    number = "".join(first_ten) + _steuer_check("".join(first_ten))
    if rng.randrange(4) == 0:
        return _group(number, (2, 3, 3, 3))
    return number


def _ppsn(rng: random.Random) -> str:
    digits = _digits(rng, 7)
    second = rng.choice(("", "", "A", "B", "W"))
    total = sum(int(c) * w for c, w in zip(digits, range(8, 1, -1), strict=True))
    if second:
        total += _PPSN_ALPHABET.index(second) * 9
    value = f"{digits}{_PPSN_ALPHABET[total % 23]}{second}"
    style = rng.randrange(4)
    if style == 0:
        return value.lower()
    if style == 1:
        return f"{digits} {value[7:]}"
    return value


def _ipv4(rng: random.Random) -> str | Framed:
    octets = [rng.randrange(1, 224), rng.randrange(256), rng.randrange(256), rng.randrange(1, 255)]
    address = ".".join(str(octet) for octet in octets)
    style = rng.randrange(4)
    if style == 0:
        return "", address, f":{rng.choice((22, 443, 8080))}"
    if style == 1:
        return "", f"{'.'.join(str(o) for o in octets[:3])}.0", "/24"
    return address


def _ipv6(rng: random.Random) -> str | Framed:
    groups = [f"{rng.randrange(1, 0x10000):x}" for _ in range(8)]
    style = rng.randrange(5)
    if style == 0:
        return ":".join(groups)
    if style == 1:
        return f"2a02:{groups[1]}:{groups[2]}::{groups[7]}"
    if style == 2:
        return f"fe80::{groups[4]}:{groups[5]}:{groups[6]}:{groups[7]}"
    if style == 3:
        return "[", f"2001:db8:{groups[2]}::{groups[7]}", "]:443"
    return ":".join(groups).upper()


def _dob(rng: random.Random) -> str:
    day = date(1940, 1, 1) + timedelta(days=rng.randrange(65 * 365))
    style = rng.randrange(5)
    if style == 0:
        return day.isoformat()
    if style == 1:
        return f"{day.day:02d}/{day.month:02d}/{day.year}"
    if style == 2:
        return f"{day.day}.{day.month}.{day.year}"
    if style == 3:
        return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"
    return f"{day.day:02d}.{day.month:02d}.{day.year}"


def _positive_value(
    kind: Kind, rng: random.Random, names: list[str]
) -> tuple[Framed, tuple[str, ...]]:
    if kind is Kind.PERSON:
        name = rng.choice(names)
        if rng.randrange(3) == 0:
            given = name.split(" ")[0]
            return ("", given, ""), (given,)
        return ("", name, ""), (name,)
    makers: dict[Kind, Callable[[random.Random], str | Framed]] = {
        Kind.EMAIL: _email,
        Kind.PHONE: _phone,
        Kind.IBAN: _iban,
        Kind.CARD: _card,
        Kind.STEUER_ID: _steuer_id,
        Kind.PPSN: _ppsn,
        Kind.IPV4: _ipv4,
        Kind.IPV6: _ipv6,
        Kind.DOB: _dob,
    }
    made = makers[kind](rng)
    return (("", made, "") if isinstance(made, str) else made), ()


def _negative_value(shape: str, rng: random.Random, words: list[list[str]]) -> Value:
    if shape == "luhn_fail":
        body = "4" + _digits(rng, 14)
        bad = (int(_luhn_digit(body)) + rng.randrange(1, 10)) % 10
        return body + str(bad), ()
    if shape == "version":
        parts = (rng.randrange(1, 9), rng.randrange(20), rng.randrange(10), rng.randrange(1, 400))
        return ".".join(str(part) for part in parts), ()
    if shape == "date_no_cue":
        day = date(2024, 1, 1) + timedelta(days=rng.randrange(900))
        return (
            day.isoformat() if rng.randrange(2) else f"{day.day:02d}.{day.month:02d}.{day.year}"
        ), ()
    if shape == "eleven_digits":
        first_ten = str(rng.randrange(1, 10)) + _digits(rng, 9)
        last = str((int(_steuer_check(first_ten)) + rng.randrange(1, 10)) % 10)
        return first_ten + last, ()
    if shape == "clock_time":
        return f"{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}", ()
    if shape == "mac_address":
        return ":".join(f"{rng.randrange(256):02x}" for _ in range(6)), ()
    if shape == "zero_reference":
        return f"0{rng.randrange(10)}-{_digits(rng, 4)}-{_digits(rng, 4)}", ()
    if shape == "ppsn_fail":
        digits = _digits(rng, 7)
        total = sum(int(c) * w for c, w in zip(digits, range(8, 1, -1), strict=True))
        wrong = _PPSN_ALPHABET[(total + rng.randrange(1, 23)) % 23]
        return f"{digits}{wrong}", ()
    if shape == "amount":
        return f"{rng.randrange(1, 99)},{_digits(rng, 3)}.{_digits(rng, 2)} EUR", ()
    if shape == "name_inside_word":
        word, name = rng.choice(words)
        return word, (name,)
    raise ValueError(f"no look-alike shape named {shape!r}")


def _case(
    case_id: str, template: str, framed: Framed, directory: tuple[str, ...], kind: Kind | None
) -> Case:
    before, value, after = framed
    start = template.index(SLOT) + len(before)
    text = template.replace(SLOT, before + value + after)
    spans = () if kind is None else (ExpectedSpan(kind, start, start + len(value), value),)
    return Case(id=case_id, text=text, spans=spans, directory=directory)


def build_heldout_cases() -> tuple[Case, ...]:
    """The 600 held-out cases: 50 per kind in `Kind` order, then the 100 look-alikes."""
    document = _templates()
    rng = random.Random(HELDOUT_SEED)
    names: list[str] = document["names"]
    words: list[list[str]] = document["words_with_names"]
    cases: list[Case] = []
    for kind in Kind:
        templates: list[str] = document["positive"][kind.value]
        for number in range(POSITIVES_PER_KIND):
            template = templates[number % len(templates)]
            framed, directory = _positive_value(kind, rng, names)
            case_id = f"heldout-{kind.value.lower()}-{number + 1:02d}"
            cases.append(_case(case_id, template, framed, directory, kind))
    shapes: dict[str, list[str]] = document["negative"]
    per_shape = NEGATIVES // len(shapes)
    for shape, templates in shapes.items():
        for number in range(per_shape):
            template = templates[number % len(templates)]
            value, directory = _negative_value(shape, rng, words)
            case_id = f"heldout-neg-{shape.replace('_', '-')}-{number + 1:02d}"
            cases.append(_case(case_id, template, ("", value, ""), directory, None))
    return tuple(cases)


def template_fixed_texts() -> list[str]:
    """Each template's longest fixed piece, the text around its slot, stripped."""
    document = _templates()
    templates = [t for group in document["positive"].values() for t in group]
    templates += [t for group in document["negative"].values() for t in group]
    return [max((piece.strip() for piece in t.split(SLOT)), key=len) for t in templates]
