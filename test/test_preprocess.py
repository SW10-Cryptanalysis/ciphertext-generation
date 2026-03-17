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
        "difficulty": 20,
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


def test_smart_truncation_limits_content(sample_payload):
    cfg = Config(unique_homophones=10, use_spaces=False)

    # Payload has 2 cipher tokens + 2 plain tokens = 4 content tokens.
    # We force max_context to 5.
    # With 3 special tokens (BOS, SEP, EOS), the content budget is only 2.
    cfg.max_context = 5

    converter = RawToArrowConverter(cfg)
    result = converter.tokenize_fn(sample_payload)
    ids = result["input_ids"]

    # Budget is 2. The 50/50 split means Cipher gets 1 token, Plain gets 1 token.
    # Cipher: "1" -> 1
    # Plain: "a" -> 15
    # Expected: [BOS] + [1] + [SEP] + [15] + [EOS]
    assert ids == [13, 1, 11, 15, 14]

    # Mathematically guarantee the truncation logic respects the hard ceiling
    assert len(ids) == cfg.max_context
