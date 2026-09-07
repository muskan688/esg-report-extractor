"""Unit normalization for figures pulled out of sustainability reports.

Reports are wildly inconsistent about units: one company reports Scope 1 in
"t CO2e", another in "kt CO2e", another in "million t CO2e". Energy shows up
as MWh, GWh or TWh. Headcounts show up as raw numbers or "in thousands".
This module maps whatever string the LLM read next to a number onto a
canonical unit (see ``METRIC_FIELDS`` in schema/metrics.py) so metrics are
comparable across the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# multiplier -> canonical base value, keyed by loosely-matched tokens found
# in the unit string (checked longest-token-first so "million tonnes" beats
# a bare "tonnes").
_MAGNITUDE_TOKENS: dict[str, float] = {
    "billion": 1e9,
    "bn": 1e9,
    "mrd": 1e9,  # German "Milliarden"
    "million": 1e6,
    "mio": 1e6,  # German "Millionen"
    "mn": 1e6,
    "thousand": 1e3,
    "tsd": 1e3,  # German "Tausend"
    "kilo": 1e3,
    "k": 1e3,
}

# base-unit family -> {token: multiplier relative to the canonical unit}
_MASS_TO_TONNES = {
    "t": 1.0,
    "tonne": 1.0,
    "tonnes": 1.0,
    "ton": 1.0,
    "tons": 1.0,
    "kt": 1e3,
    "mt": 1e6,
    "kg": 1e-3,
    "g": 1e-6,
}

_ENERGY_TO_MWH = {
    "kwh": 1e-3,
    "mwh": 1.0,
    "gwh": 1e3,
    "twh": 1e6,
    "gj": 0.2778,  # 1 GJ = 0.2778 MWh
    "tj": 277.8,
    "pj": 277_800.0,
}

_VOLUME_TO_M3 = {
    "l": 1e-3,
    "liter": 1e-3,
    "litre": 1e-3,
    "m3": 1.0,
    "m³": 1.0,
    "cubic meter": 1.0,
    "cubic metre": 1.0,
    "ml": 1e3,  # megaliters
}

_FAMILY_BY_CANONICAL = {
    "t CO2e": ("mass_co2e", _MASS_TO_TONNES),
    "t": ("mass", _MASS_TO_TONNES),
    "MWh": ("energy", _ENERGY_TO_MWH),
    "m3": ("volume", _VOLUME_TO_M3),
    "%": ("percent", {"%": 1.0, "percent": 1.0, "pct": 1.0}),
    "headcount": ("count", {"": 1.0, "employees": 1.0, "fte": 1.0, "headcount": 1.0}),
}


@dataclass
class NormalizationResult:
    value: Optional[float]
    matched: bool
    note: str = ""


def _clean(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower())


def normalize_unit(
    raw_value: Optional[float], raw_unit: Optional[str], canonical_unit: str
) -> NormalizationResult:
    """Convert ``raw_value raw_unit`` into ``canonical_unit``.

    Returns ``matched=False`` (value passed through unchanged) when the unit
    text can't be confidently parsed, so callers can flag it for review
    rather than silently trusting a wrong conversion.
    """
    if raw_value is None:
        return NormalizationResult(value=None, matched=False, note="no raw value")

    if not raw_unit:
        # No unit given at all: assume it's already in the canonical unit.
        return NormalizationResult(value=raw_value, matched=True, note="no unit; assumed canonical")

    unit = _clean(raw_unit)
    family_name, table = _FAMILY_BY_CANONICAL.get(canonical_unit, (None, None))
    if table is None:
        return NormalizationResult(value=raw_value, matched=False, note=f"unknown canonical unit {canonical_unit}")

    # magnitude prefix, e.g. "million tonnes CO2e" or "thousand employees"
    magnitude = 1.0
    remainder = unit
    for token, mult in sorted(_MAGNITUDE_TOKENS.items(), key=lambda kv: -len(kv[0])):
        pattern = rf"\b{re.escape(token)}\b"
        if re.search(pattern, remainder):
            magnitude = mult
            remainder = re.sub(pattern, " ", remainder)
            break

    remainder = remainder.replace("co2e", "").replace("co2-e", "").replace("co2eq", "")
    remainder = re.sub(r"\s+", " ", remainder).strip()

    base_mult = None
    for token, mult in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if token == "":
            continue
        if re.fullmatch(re.escape(token), remainder) or remainder == token:
            base_mult = mult
            break
    if base_mult is None:
        # fall back to substring match (handles trailing noise like "p.a.")
        for token, mult in sorted(table.items(), key=lambda kv: -len(kv[0])):
            if token and token in remainder:
                base_mult = mult
                break

    if base_mult is None:
        if family_name == "count" and remainder == "":
            base_mult = 1.0
        else:
            return NormalizationResult(
                value=raw_value, matched=False, note=f"could not parse unit '{raw_unit}'"
            )

    return NormalizationResult(value=raw_value * magnitude * base_mult, matched=True)


_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(%|[A-Za-z³./\- ]{0,20})"
)


def parse_number_and_unit(text: str) -> tuple[Optional[float], Optional[str]]:
    """Best-effort parse of a value+unit out of a short text fragment.

    Used as a fallback / sanity check when validating LLM output against the
    source quote (e.g. "1,234.5 kt CO2e" -> (1234.5, "kt CO2e")).
    """
    match = _NUMBER_RE.search(text)
    if not match:
        return None, None
    number_str, unit = match.group(1), match.group(2).strip() or None
    # German/EU formatting uses "." as thousands sep and "," as decimal;
    # disambiguate by whichever separator appears last.
    if "," in number_str and "." in number_str:
        if number_str.rfind(",") > number_str.rfind("."):
            number_str = number_str.replace(".", "").replace(",", ".")
        else:
            number_str = number_str.replace(",", "")
    elif "," in number_str:
        # ambiguous: "1,234" (thousands) vs "12,5" (decimal) - use length of
        # the trailing group to decide.
        head, tail = number_str.rsplit(",", 1)
        number_str = number_str.replace(",", "") if len(tail) == 3 else f"{head}.{tail}"
    try:
        return float(number_str), unit
    except ValueError:
        return None, unit
