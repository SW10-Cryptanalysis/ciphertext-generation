import pytest
from dataclasses import dataclass
from typing import Any

from truncation.cipher_truncator import CipherTruncator
from utils.text_sampling import TextStream
from encipherment.cipher import HomophonicCipher, SubstitutionCipher


@dataclass
class TruncaterTestCase:
    """Defines parameters for validating the CipherTruncater boundary logic."""

    input_stream: list[dict[str, Any]]
    max_length: int
    sampled_lengths: list[int]
    expected_yield_types: list[type]
    expected_texts: list[str]


truncater_test_cases = [
    TruncaterTestCase(
        input_stream=[{"plaintext_with_boundaries": "short_text_here"}],
        max_length=20,
        sampled_lengths=[],
        expected_yield_types=[SubstitutionCipher],
        expected_texts=[],
    ),
    TruncaterTestCase(
        input_stream=[
            {
                "plaintext_with_boundaries": "this_is_a_very_long_plaintext_that_requires_chunking_and_then_dropping_the_tail",
                "genres": ["test"],
                "source_id": "1",
                "source_name": "test_source",
            }
        ],
        max_length=20,
        sampled_lengths=[15, 20, 50],
        expected_yield_types=[TextStream, TextStream],
        expected_texts=["this_is_a_very_long", "plaintext_that"],
    ),
]


@pytest.mark.parametrize("test_case", truncater_test_cases)
def test_cipher_truncater_yields(mocker, test_case: TruncaterTestCase):
    """Validates target length sampling, chunk boundary logic, and tail dropping."""

    mocker.patch(
        "encipherment.cipher.HomophonicCipher.from_json",
        return_value=HomophonicCipher,
    )

    mock_sampler = mocker.Mock()
    mock_sampler.side_effect = test_case.sampled_lengths

    truncater = CipherTruncator(
        stream=iter(test_case.input_stream),
        max_length=test_case.max_length,
        length_sampler=mock_sampler,
    )

    results = list(truncater.process_stream())

    assert len(results) == len(test_case.expected_yield_types)
    for idx, result in enumerate(results):
        if type(result) is dict:
            assert result["text_with_boundaries"] == test_case.expected_texts[idx]
            assert result["target_length"] == test_case.sampled_lengths[idx]
        elif isinstance(result, SubstitutionCipher):
            assert result.plaintext_with_boundaries == test_case.expected_texts[idx]
