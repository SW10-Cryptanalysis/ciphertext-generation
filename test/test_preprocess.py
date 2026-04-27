import pytest
from dataclasses import dataclass
from typing import Any


from preprocess import RawToArrowConverter, Config


@dataclass
class TokenizeTestCase:
    """Defines a strict scenario for the tokenization logic."""

    name: str
    task: str
    use_spaces: bool
    payload: dict[str, Any]
    expected_input_ids: list[int]
    expected_labels: list[int]
    expected_plain_length: int


def get_tokenize_cases() -> list[TokenizeTestCase]:
    """Generates all permutations of task, spacing, and symbol arrangements."""

    payload_simple = {
        "ciphertext": "1 2",
        "plaintext": "ab",
        "ciphertext_with_boundaries": "1 _ 2",
        "plaintext_with_boundaries": "a _ b",
    }

    payload_duplicates = {
        "ciphertext": "1 2 1",
        "plaintext": "aba",
        "ciphertext_with_boundaries": "1 _ 2 _ 1",
        "plaintext_with_boundaries": "a _ b _ a",
    }

    return [
        TokenizeTestCase(
            name="causal_no_spaces",
            task="causal",
            use_spaces=False,
            payload=payload_simple,
            expected_input_ids=[13, 1, 2, 11, 15, 16, 14],
            expected_labels=[13, 1, 2, 11, 15, 16, 14],
            expected_plain_length=2,
        ),
        TokenizeTestCase(
            name="causal_with_spaces",
            task="causal",
            use_spaces=True,
            payload=payload_simple,
            expected_input_ids=[13, 1, 12, 2, 11, 15, 12, 16, 14],
            expected_labels=[13, 1, 12, 2, 11, 15, 12, 16, 14],
            expected_plain_length=3,
        ),
        TokenizeTestCase(
            name="mapping_no_spaces",
            task="mapping",
            use_spaces=False,
            payload=payload_simple,
            expected_input_ids=[13, 1, 2, 14],
            expected_labels=[0, 1, -100, -100],
            expected_plain_length=0,
        ),
        TokenizeTestCase(
            name="mapping_with_spaces",
            task="mapping",
            use_spaces=True,
            payload=payload_simple,
            expected_input_ids=[13, 1, 12, 2, 14],
            expected_labels=[0, 1, -100, -100, -100],
            expected_plain_length=0,
        ),
        TokenizeTestCase(
            name="mapping_with_duplicate_symbols",
            task="mapping",
            use_spaces=False,
            payload=payload_duplicates,
            expected_input_ids=[13, 1, 2, 1, 14],
            expected_labels=[0, 1, -100, -100],
            expected_plain_length=0,
        ),
    ]


class TestRawToArrowConverter:
    """Thorough test suite for dataset tokenization logic."""

    @pytest.mark.parametrize("case", get_tokenize_cases(), ids=lambda c: c.name)
    def test_tokenize_logic(self, case: TokenizeTestCase) -> None:
        """Verifies input_ids generation, label alignment, and length calculation."""

        cfg = Config(task=case.task, use_spaces=case.use_spaces, unique_homophones=10)

        converter = RawToArrowConverter(cfg)
        result = converter.tokenize_fn(case.payload)

        assert result["input_ids"] == case.expected_input_ids
        assert result["labels"] == case.expected_labels
        assert result["plain_length"] == case.expected_plain_length

        expected_raw_key = (
            "plaintext_with_boundaries" if case.use_spaces else "plaintext"
        )
        assert result["raw_plaintext"] == case.payload[expected_raw_key]
