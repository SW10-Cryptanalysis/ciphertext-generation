import pytest
import json
from typing import Any
from dataclasses import dataclass
from encipherment.cipher import HomophonicCipher, MonoalphabeticCipher
from utils.constants import MIN_DIFFICULTY, MAX_DIFFICULTY


@pytest.fixture(scope="module")
def valid_text_stream():
    """Returns a long valid TextStream dictionary for full cipher generation."""
    text = "abcdefghijklmnopqrstuvwxyz" * 20
    return {
        "text": text,
        "text_with_boundaries": text.replace("a", "a_"),
        "source_id": "long_1",
        "source_name": "Alphabet Repeated",
        "length": len(text),
        "target_length": len(text),
        "genres": ["Sci-Fi & Fantasy"],
    }


@pytest.fixture(scope="module")
def sample_stream_short():
    """Returns a very short valid TextStream dictionary."""
    return {
        "text": "abcbc",
        "text_with_boundaries": "abc_bc",
        "source_id": "short_1",
        "source_name": "Alphabet",
        "length": len("abcbc"),
        "target_length": len("abcbc"),
        "genres": ["Sci-Fi & Fantasy"],
    }


@dataclass
class BadStreamTestCase:
    """Encapsulates parameters for testing invalid TextStream dictionaries."""

    desc: str
    stream: dict[str, Any]
    expected_exception: type[Exception]
    match: str


BAD_STREAM_CASES = [
    BadStreamTestCase(
        desc="Invalid: Empty plaintext string",
        stream={
            "text": "",
            "text_with_boundaries": "",
            "source_id": "123",
            "source_name": "Test",
            "length": 0,
            "target_length": 0,
            "genres": [],
        },
        expected_exception=ValueError,
        match="non-empty",
    ),
    BadStreamTestCase(
        desc="Invalid: Whitespace only plaintext string",
        stream={
            "text": " ",
            "text_with_boundaries": " ",
            "source_id": "123",
            "source_name": "Test",
            "length": 0,
            "target_length": 0,
            "genres": [],
        },
        expected_exception=ValueError,
        match="lowercase alphabetic string with no spaces",
    ),
    BadStreamTestCase(
        desc="Invalid: Contains uppercase and punctuation",
        stream={
            "text": "InvalidText!@#",
            "text_with_boundaries": "Invalid_Text!@#",
            "source_id": "123",
            "source_name": "Test",
            "length": 14,
            "target_length": 14,
            "genres": [],
        },
        expected_exception=ValueError,
        match="lowercase alphabetic string with no spaces",
    ),
]


class TestStreamValidation:
    """Test that stream validation is consistently applied across all cipher types."""

    @pytest.mark.parametrize("cipher_class", [HomophonicCipher, MonoalphabeticCipher])
    @pytest.mark.parametrize("case", BAD_STREAM_CASES, ids=lambda c: c.desc)
    def test_invalid_stream_initialization(self, cipher_class, case: BadStreamTestCase):
        """Test that invalid stream parameters raise the specific expected exceptions."""
        with pytest.raises(case.expected_exception, match=case.match):
            cipher_class(case.stream)


class TestHomophonicCipher:
    def test_legal_plaintext(self, valid_text_stream):
        cipher = HomophonicCipher(valid_text_stream)
        cipher.generate_key()
        cipher.encipher()

        assert cipher.plaintext == valid_text_stream["text"]
        assert cipher.source_id == valid_text_stream["source_id"]
        assert MIN_DIFFICULTY <= cipher.redundancy <= MAX_DIFFICULTY
        assert isinstance(cipher.key, dict)
        assert all(isinstance(v, list) for v in cipher.key.values())
        assert isinstance(cipher.ciphertext, str)

    def test_all_homophones_used(self, valid_text_stream):
        cipher = HomophonicCipher(valid_text_stream)
        cipher.generate_key()
        cipher.encipher()
        used_homophones = set(int(num) for num in cipher.ciphertext.split())
        all_homophones = {h for homophones in cipher.key.values() for h in homophones}
        assert used_homophones == all_homophones

    def test_defined_redundancy(self, valid_text_stream):
        # We assume MIN_DIFFICULTY <= 10. If validation allows it, it should stick.
        target_red = min(10, MAX_DIFFICULTY)
        if target_red >= MIN_DIFFICULTY:
            cipher = HomophonicCipher(valid_text_stream, redundancy=target_red)
            assert cipher.redundancy == target_red

    def test_dynamic_redundancy_generation_bounds(self, sample_stream_short):
        """Verify generate_redundancy safely clamps to max possible even if MIN_DIFFICULTY > max."""
        # 'abcbc' has 5 chars and 3 unique letters. Max redundancy = 5 // 3 = 1.
        # This will force the generator to choose 1, overriding standard difficulty bounds.
        cipher = HomophonicCipher(sample_stream_short)
        assert cipher.redundancy == 1

    def test_automatic_redundancy_clamping_logic(self, sample_stream_short):
        """Verify that explicitly requested redundancy is capped at len(text) // unique_chars."""
        target_red = max(MIN_DIFFICULTY, 5)
        # Passing an explicit redundancy higher than physical bounds (1) should clamp it down.
        cipher = HomophonicCipher(sample_stream_short, redundancy=target_red)
        assert cipher.redundancy == 1

    def test_max_possible_redundancy_zero_letters(self, valid_text_stream):
        """Verify the divide-by-zero safeguard in get_max_possible_redundancy."""
        cipher = HomophonicCipher(valid_text_stream)
        # Artificially clear plaintext to hit the condition (since __init__ blocks it)
        cipher.plaintext = ""
        assert cipher.get_max_possible_redundancy() == 1

    def test_invalid_redundancy_input(self, valid_text_stream):
        for invalid_val in [MIN_DIFFICULTY - 1, MAX_DIFFICULTY + 1]:
            with pytest.raises((ValueError, TypeError)):
                HomophonicCipher(valid_text_stream, redundancy=invalid_val)

    def test_ciphertext_length(self, valid_text_stream):
        cipher = HomophonicCipher(valid_text_stream)
        cipher.generate_key()
        cipher.encipher()
        assert len(cipher.ciphertext.split()) == len(valid_text_stream["text"])

    def test_json_serialization(self, valid_text_stream):
        cipher = HomophonicCipher(valid_text_stream)
        json_data = cipher.__json__()

        assert json_data["plaintext"] == valid_text_stream["text"]
        assert json_data["source_id"] == valid_text_stream["source_id"]
        assert json_data["redundancy"] == cipher.redundancy

    def test_recurrence_encoding_property(self, valid_text_stream):
        cipher = HomophonicCipher(valid_text_stream)
        cipher.generate_key()
        cipher.encipher()

        encoding_numbers = list(map(int, cipher.recurrence_encoding.split()))
        assert all(num >= 1 for num in encoding_numbers)
        assert set(encoding_numbers) == set(range(1, max(encoding_numbers) + 1))


class TestApplyRecurrenceAndRemapKey:
    def test_standard_remapping_logic(self, sample_stream_short):
        cipher = HomophonicCipher(sample_stream_short)
        cipher.key = {"a": [99], "b": [10], "c": [50]}
        cipher.ciphertext = "99 10 50 10 50"

        cipher._apply_recurrence_and_remap_key()

        assert cipher.ciphertext == "1 2 3 2 3"
        assert cipher.key["a"] == [1]
        assert cipher.key["b"] == [2]
        assert cipher.key["c"] == [3]

    def test_generate_bounded_ciphertext_logic(self, sample_stream_short):
        cipher = HomophonicCipher(sample_stream_short)
        cipher.key = {"a": [99], "b": [10], "c": [50]}
        cipher.ciphertext = "99 10 50 10 50"

        cipher._apply_recurrence_and_remap_key()
        assert cipher.ciphertext_with_boundaries == "1 2 3 _ 2 3"

    def test_first_seen_is_always_one(self, sample_stream_short):
        cipher = HomophonicCipher(sample_stream_short)
        cipher.key = {"a": [10], "b": [20], "c": [30]}

        cipher.ciphertext = "30 10 30 20 10"
        cipher._apply_recurrence_and_remap_key()

        assert cipher.ciphertext.startswith("1")
        assert set(cipher.key["c"]) == {1}
        assert set(cipher.key["a"]) == {2}
        assert set(cipher.key["b"]) == {3}


class TestMonoalphabeticCipher:
    def test_legal_plaintext(self, sample_stream_short):
        cipher = MonoalphabeticCipher(sample_stream_short)

        assert cipher.plaintext == sample_stream_short["text"]
        assert cipher.source_id == sample_stream_short["source_id"]
        assert isinstance(cipher.key, dict)
        assert all(isinstance(v, list) and len(v) == 1 for v in cipher.key.values())
        assert len(cipher.ciphertext.split()) == len(sample_stream_short["text"])

    def test_key_structure(self, sample_stream_short):
        cipher = MonoalphabeticCipher(sample_stream_short)
        # 'abcbc' has 3 unique letters
        assert len(cipher.key) == 3
        all_numbers = [cipher.key[letter][0] for letter in cipher.key]
        assert len(set(all_numbers)) == 3

    def test_json_serialization_success(self, sample_stream_short):
        cipher = MonoalphabeticCipher(sample_stream_short)
        json_data = cipher.__json__()

        assert json_data["plaintext"] == sample_stream_short["text"]
        assert json_data["source_id"] == sample_stream_short["source_id"]
        assert json_data["key"] == cipher.key


class TestCipherSharedMethods:
    """Tests covering methods inherited from SubstitutionCipher."""

    @pytest.mark.parametrize(
        "cipher_class, needs_manual_steps",
        [
            (HomophonicCipher, True),
            (MonoalphabeticCipher, False),
        ],
    )
    def test_str_representation(
        self, cipher_class, needs_manual_steps, valid_text_stream
    ):
        """Verify the string representation dynamically handles child classes."""
        cipher = cipher_class(valid_text_stream)

        if needs_manual_steps:
            cipher.generate_key()
            cipher.encipher()

        str_repr = str(cipher)

        assert f'{cipher_class.__name__}(Plaintext: "' in str_repr
        assert f'"{valid_text_stream["text"]}"' in str_repr
        assert "Redundancy:" in str_repr
        assert f"Key: {cipher.key}" in str_repr
        assert f'Ciphertext: "{cipher.ciphertext}"' in str_repr

    def test_from_json_deserialization(self, valid_text_stream):
        """Verify that a dumped cipher can be perfectly reconstructed."""
        cipher = HomophonicCipher(valid_text_stream)
        cipher.generate_key()
        cipher.encipher()

        json_str = json.dumps(cipher.__json__())

        reconstructed = HomophonicCipher.from_json(json_str)

        assert reconstructed.plaintext == cipher.plaintext
        assert (
            reconstructed.plaintext_with_boundaries == cipher.plaintext_with_boundaries
        )
        assert reconstructed.key == cipher.key
        assert reconstructed.ciphertext == cipher.ciphertext
        assert (
            reconstructed.ciphertext_with_boundaries
            == cipher.ciphertext_with_boundaries
        )
        assert reconstructed.num_symbols == cipher.num_symbols
        assert reconstructed.redundancy == cipher.redundancy
        assert reconstructed.genres == cipher.genres
        assert reconstructed.source_id == cipher.source_id
        assert reconstructed.source_name == cipher.source_name
