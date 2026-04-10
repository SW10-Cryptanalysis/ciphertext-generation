import pytest
from preprocess import RawToArrowConverter, Config


@pytest.fixture
def sample_payload():
    """Provide a sample dictionary as it would appear when loaded from JSON."""
    return {
        "ciphertext": "1 2",
        "plaintext": "ab",
        "ciphertext_with_boundaries": "1 _ 2",
        "plaintext_with_boundaries": "a _ b",
        "redundancy": 20,
    }


def test_mapping_logic_no_spaces(sample_payload):
    # Set unique_homophones=10 to lock the special tokens to the expected IDs
    cfg = Config(unique_homophones=10, use_spaces=False)
    converter = RawToArrowConverter(cfg)

    result = converter.tokenize_fn(sample_payload)
    ids = result["input_ids"]

    # [BOS] + [1, 2] + [SEP] + [15, 16] + [EOS]
    assert ids == [13, 1, 2, 11, 15, 16, 14]

    # Verify joint distribution labels match inputs exactly
    assert result["labels"] == ids


def test_mapping_logic_with_spaces(sample_payload):
    cfg = Config(unique_homophones=10, use_spaces=True)
    converter = RawToArrowConverter(cfg)

    result = converter.tokenize_fn(sample_payload)
    ids = result["input_ids"]

    # Expected: [BOS] + [1, SPACE, 2] + [SEP] + [15, SPACE, 16] + [EOS]
    assert ids == [13, 1, 12, 2, 11, 15, 12, 16, 14]
