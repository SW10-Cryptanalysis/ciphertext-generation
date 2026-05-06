import pytest
from dataclasses import dataclass
from typing import Any

from conversion.monoalphabetic_converter import MonoalphabeticConverter
from encipherment.cipher import MonoalphabeticCipher


@dataclass
class ConverterTestCase:
    """Defines parameters for validating the MonoalphabeticConverter logic."""

    input_stream: list[dict[str, Any]]
    expected_yield_count: int


converter_test_cases = [
    ConverterTestCase(
        input_stream=[
            {
                "plaintext": "thisisatest",
                "plaintext_with_boundaries": "this_is_a_test",
                "genres": ["test"],
                "source_id": "1",
                "source_name": "test_source",
            }
        ],
        expected_yield_count=1,
    ),
    ConverterTestCase(
        input_stream=[],
        expected_yield_count=0,
    ),
]


@pytest.mark.parametrize("test_case", converter_test_cases)
def test_monoalphabetic_converter_yields(mocker, test_case: ConverterTestCase):
    """Validates the converter correctly transforms stream dicts into cipher objects."""
    mocker.patch("encipherment.cipher.MonoalphabeticCipher.__init__", return_value=None)

    converter = MonoalphabeticConverter(stream=iter(test_case.input_stream))
    results = list(converter.process_stream())

    assert len(results) == test_case.expected_yield_count

    if test_case.expected_yield_count > 0:
        assert isinstance(results[0], MonoalphabeticCipher)
