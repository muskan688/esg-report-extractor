import math

import pytest

from esg_extractor.extraction.units import normalize_unit, parse_number_and_unit


@pytest.mark.parametrize(
    "raw_value,raw_unit,canonical_unit,expected",
    [
        (1234.5, "t CO2e", "t CO2e", 1234.5),
        (12.5, "kt CO2e", "t CO2e", 12_500.0),
        (1.2, "million t CO2e", "t CO2e", 1_200_000.0),
        (500, "MWh", "MWh", 500.0),
        (2.5, "GWh", "MWh", 2500.0),
        (36.5, "%", "%", 36.5),
        (12000, "employees", "headcount", 12000.0),
        (12, "thousand employees", "headcount", 12000.0),
        (100, "m3", "m3", 100.0),
        (50, "thousand m3", "m3", 50000.0),
    ],
)
def test_normalize_unit_conversions(raw_value, raw_unit, canonical_unit, expected):
    result = normalize_unit(raw_value, raw_unit, canonical_unit)
    assert result.matched
    assert math.isclose(result.value, expected, rel_tol=1e-6)


def test_normalize_unit_no_unit_assumes_canonical():
    result = normalize_unit(42.0, None, "MWh")
    assert result.matched
    assert result.value == 42.0


def test_normalize_unit_unparseable_flags_unmatched():
    result = normalize_unit(10.0, "furlongs", "MWh")
    assert result.matched is False
    assert result.value == 10.0  # passthrough, not silently dropped


def test_normalize_unit_no_value_returns_none():
    result = normalize_unit(None, "t CO2e", "t CO2e")
    assert result.value is None
    assert result.matched is False


@pytest.mark.parametrize(
    "text,expected_value",
    [
        ("1,234.5 kt CO2e", 1234.5),
        ("1.234,5 kt CO2e", 1234.5),  # German/EU formatting
        ("36,5 %", 36.5),
        ("42", 42.0),
    ],
)
def test_parse_number_and_unit(text, expected_value):
    value, _unit = parse_number_and_unit(text)
    assert math.isclose(value, expected_value, rel_tol=1e-6)
