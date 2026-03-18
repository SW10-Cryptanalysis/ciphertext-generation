from .homophones import extract_homophones, get_homophones
from .frequency import frequencies
import random
from collections import Counter
from utils.constants import MIN_DIFFICULTY, MAX_DIFFICULTY
from abc import ABC, abstractmethod
from parameter_validator import parameter_validator, all_of
import json
from utils.text_sampling import TextStream

from utils.validators import (
    in_range,
    strongly_typed_optional,
    validate_typed_dict,
    is_alpha_lowercase_no_spaces,
)


class SubstitutionCipher(ABC):
    """A base class for substitution ciphers.

    Attributes:
        plaintext (str): The lowercase plaintext to be encrypted with no punctuation
            or spaces.
        plaintext_with_boundaries (str): The plaintext preserving word boundary
            underscores.
        redundancy (int | None): The redundancy level of the cipher.
        num_symbols (int): The total number of unique symbols in the generated key.
        key (dict[str, list[int]]): A dictionary mapping each letter to its cipher
            symbols.
        ciphertext (str): The resulting ciphertext as a string of recurrence encoded
            numbers separated by spaces.
        ciphertext_with_boundaries (str): The recurrence encoded ciphertext preserving
            word boundary underscores.
        genres (list[str]): A list of genres associated with the source text.
        source_id (str): The unique identifier for the source text.
        source_name (str): The title or name of the source text.

    Methods:
        generate_key(): Generate a substitution cipher key.
        encipher(): Encipher the plaintext using the generated key.
        __json__(): Return a JSON-serializable representation of the object.
        __str__(): Return a string representation of the object.
        _apply_recurrence_and_remap_key(): Apply recurrence encoding and remap the key.
        _generate_bounded_ciphertext(): Generate ciphertext preserving word boundaries.

    """

    def __init__(
        self,
        text_obj: TextStream,
        *,
        redundancy: int | None = None,
    ) -> None:
        """Initialize the Cipher object with common plaintext and metadata.

        Args:
            text_obj (TextStream): Text object containing the plaintext and metadata.
            redundancy (int | None, optional): The redundancy level to apply.
                Defaults to None.

        Raises:
            ValueError: If the plaintext is empty or contains invalid characters.

        """
        if not text_obj.get("text"):
            raise ValueError("Plaintext must be a non-empty string.")
        if not is_alpha_lowercase_no_spaces(text_obj["text"]):
            raise ValueError(
                "Plaintext must be a lowercase alphabetic string with no spaces.",
            )

        self.plaintext = text_obj["text"]
        self.plaintext_with_boundaries = text_obj["text_with_boundaries"]
        self.redundancy = redundancy
        self.num_symbols = 0
        self.key: dict[str, list[int]] = {}
        self.ciphertext = ""
        self.ciphertext_with_boundaries = ""
        self.genres = text_obj["genres"]
        self.source_id = text_obj["source_id"]
        self.source_name = text_obj["source_name"]

    def _apply_recurrence_and_remap_key(self) -> None:
        """Apply recurrence encoding and remap the key using character keys.

        Transforms the raw generated ciphertext numbers into a strict 1..N
        sequential order based on their first appearance in the text.
        """
        original_numbers = self.ciphertext.split()
        new_ciphertext = []
        symbol_map = {}

        for symbol in original_numbers:
            if symbol not in symbol_map:
                symbol_map[symbol] = str(len(symbol_map) + 1)
            new_ciphertext.append(symbol_map[symbol])

        self.ciphertext = " ".join(new_ciphertext)
        self.recurrence_encoding = self.ciphertext

        new_key = {}
        for char, homophones in self.key.items():
            remapped_homophones = [
                int(symbol_map[str(h)]) for h in homophones if str(h) in symbol_map
            ]

            new_key[char] = remapped_homophones

        self.ciphertext_with_boundaries = self._generate_bounded_ciphertext()

        self.key = new_key

    def _generate_bounded_ciphertext(self) -> str:
        """Generate a ciphertext based on the plaintext with word boundaries.

        This method maps the plaintext with word boundaries to the ciphertext
        with underscores left in place.

        Returns:
            str: The ciphertext with underscores left in place.

        """
        ciphertext = iter(self.ciphertext.split())
        bounded_ciphertext = []
        for char in self.plaintext_with_boundaries:
            if char == "_":
                bounded_ciphertext.append("_")
            else:
                bounded_ciphertext.append(next(ciphertext))

        return " ".join(bounded_ciphertext)

    def __json__(self) -> dict:
        """Return a JSON-serializable representation of the Cipher object.

        Returns:
            dict: A dictionary containing the Cipher object's attributes.

        """
        return {
            "plaintext": self.plaintext,
            "plaintext_with_boundaries": self.plaintext_with_boundaries,
            "length": len(self.plaintext),
            "num_symbols": self.num_symbols,
            "redundancy": self.redundancy,
            "key": self.key,
            "ciphertext": self.ciphertext,
            "ciphertext_with_boundaries": self.ciphertext_with_boundaries,
            "genres": self.genres,
            "source_id": self.source_id,
            "source_name": self.source_name,
        }

    def __str__(self) -> str:
        """Return a string representation of the Cipher object.

        Returns:
            str: A string representation of the Cipher object.

        """
        class_name = self.__class__.__name__
        return f'''{class_name}(Plaintext: "{self.plaintext}"
            Redundancy: {self.redundancy}
            Key: {self.key}
            Ciphertext: "{self.ciphertext}"
            '''

    @abstractmethod
    def generate_key(self) -> dict:  # pragma: no cover
        """Generate a substitution cipher key."""
        pass

    @abstractmethod
    def encipher(self) -> str:  # pragma: no cover
        """Encipher the plaintext using the generated key."""
        pass

    @classmethod
    def from_json(cls, json_data: str) -> "SubstitutionCipher":
        """Create a cipher object from a JSON string."""
        data = json.loads(json_data)

        text_obj: TextStream = {
            "text": data["plaintext"],
            "text_with_boundaries": data["plaintext_with_boundaries"],
            "length": data.get("length", len(data["plaintext"])),
            "target_length": data.get("length", len(data["plaintext"])),
            "genres": data["genres"],
            "source_id": data["source_id"],
            "source_name": data["source_name"],
        }

        cipher = cls(text_obj, redundancy=data["redundancy"])
        cipher.key = data["key"]
        cipher.ciphertext = data["ciphertext"]
        cipher.ciphertext_with_boundaries = data["ciphertext_with_boundaries"]
        cipher.num_symbols = data["num_symbols"]

        return cipher


class HomophonicCipher(SubstitutionCipher):
    """A class representing a homophonic substitution cipher.

    Inherits attributes from SubstitutionCipher. Automatically calculates a continuous
    uniform redundancy based on the text length if no specific redundancy is provided.

    Methods:
        generate_key(): Generate a homophonic substitution cipher key based on a
            redundancy level.
        encipher(): Encipher the plaintext using the generated key.
        generate_redundancy(): Generate a random redundancy level for the cipher.

    """

    @parameter_validator(
        text_obj=validate_typed_dict,
        redundancy=all_of(
            in_range(MIN_DIFFICULTY, MAX_DIFFICULTY),
            strongly_typed_optional,
        ),
    )
    def __init__(self, text_obj: TextStream, *, redundancy: int | None = None) -> None:
        """Initialize the HomophonicCipher object.

        Args:
            text_obj (TextStream): Text object containing the plaintext and metadata.
            redundancy (int | None, optional): The redundancy level of the cipher.
                If None, a random continuous redundancy will be calculated and assigned.

        Raises:
            ValueError: If the provided redundancy falls outside the allowed bounds.

        """
        super().__init__(text_obj, redundancy=redundancy)

        if self.redundancy is None:
            self.redundancy = self.generate_redundancy()

        self.redundancy = self._clamp_redundancy(self.redundancy)

    def get_max_possible_redundancy(self) -> int:
        """Calculate the maximum redundancy a plaintext can support.

        Redundancy cannot exceed (Length / Unique Letters).

        Returns:
            int: Maximum possible redundancy.

        """
        unique_letters = len(set(self.plaintext))
        if unique_letters == 0:
            return 1
        return len(self.plaintext) // unique_letters

    def generate_redundancy(self) -> int:
        """Sample a redundancy level that is physically possible for the text.

        Returns:
            int: Redundancy level bounded by MIN_DIFFICULTY and MAX_DIFFICULTY.

        """
        max_physically_possible = self.get_max_possible_redundancy()
        upper_bound = min(MAX_DIFFICULTY, max_physically_possible)

        lower_bound = min(MIN_DIFFICULTY, upper_bound)

        return random.randint(lower_bound, upper_bound)

    def _clamp_redundancy(self, value: int) -> int:
        """Clamps the requested redundancy to the physical limits."""
        return min(value, self.get_max_possible_redundancy())

    def generate_key(self) -> dict:
        """Generate a homophonic substitution cipher key based on a redundancy level.

        Key is a dictionary mapping each letter to a list of its homophones.
        Each letter is assigned a number of homophones proportional to its frequency in
        English text, with some random noise added to create variability.

        Each homophone is represented by a unique, randomly sampled number from 1
        to the total number of cipher symbols.

        Returns:
            dict: A dictionary mapping each letter to a list of its homophones.

        """
        symbols: int = round(len(self.plaintext) / self.redundancy)

        letter_frequencies = frequencies(self.plaintext)

        homophones_dict: dict[str, int] = extract_homophones(
            symbols,
            letter_frequencies,
        )

        self.num_symbols = sum(homophones_dict.values())

        homophone_numbers: list[int] = list(range(1, self.num_symbols + 1))
        random.shuffle(homophone_numbers)
        key: dict[str, list[int]] = {}
        for letter, count in homophones_dict.items():
            key[letter] = homophone_numbers[:count]
            homophone_numbers = homophone_numbers[count:]
        self.key = key
        return key

    def encipher(self) -> str:
        """Encipher the plaintext using generated homophonic substitution cipher key.

        Returns:
            str: The resulting ciphertext as a string of numbers separated by spaces.

        """
        counts = Counter(ch for ch in self.plaintext if ch in self.key)

        homophones: dict[str, list[int]] = {}
        ptr: dict[str, int] = {}
        for letter, count in counts.items():
            homophones[letter] = get_homophones(self.key[letter], count)
            ptr[letter] = 0

        ciphertext_numbers: list[str] = []
        for char in self.plaintext:
            ciphertext_numbers.append(str(homophones[char][ptr[char]]))
            ptr[char] += 1
        self.ciphertext = " ".join(ciphertext_numbers)

        self._apply_recurrence_and_remap_key()
        return self.ciphertext


class MonoalphabeticCipher(SubstitutionCipher):
    """A monoalphabetic substitution cipher.

    Inherits attributes from SubstitutionCipher. Hardcodes the redundancy to 1,
    as every letter maps to exactly one unique symbol.

    Methods:
        generate_key(): Generate a monoalphabetic substitution key.
        encipher(): Encipher the plaintext using the generated key.

    """

    @parameter_validator(text_obj=validate_typed_dict)
    def __init__(self, text_obj: TextStream) -> None:
        """Initialize the MonoalphabeticCipher object.

        Automatically generates the key and enciphers the text upon initialization.

        Args:
            text_obj (TextStream): Text object containing the plaintext and metadata.

        """
        super().__init__(text_obj, redundancy=1)
        self.key = self.generate_key()
        self.encipher()

    def generate_key(self) -> dict[str, list[int]]:
        """Generate a random monoalphabetic substitution key.

        Each letter maps to exactly one unique random number.

        Returns:
            dict: A dictionary mapping each letter to a list containing one random
            number.

        """
        cipher_numbers = list(range(1, 27))
        random.shuffle(cipher_numbers)

        key = {}
        for i, letter in enumerate(set(self.plaintext)):
            key[letter] = [cipher_numbers[i]]
            self.num_symbols += 1

        return key

    def encipher(self) -> str:
        """Encipher the plaintext using the monoalphabetic substitution key.

        Returns:
            str: The resulting ciphertext as a string of numbers separated by spaces.

        """
        ciphertext_numbers = []
        for char in self.plaintext:
            if char in self.key:
                ciphertext_numbers.append(
                    str(self.key[char][0]),
                )

        self.ciphertext = " ".join(ciphertext_numbers)
        self._apply_recurrence_and_remap_key()

        return self.ciphertext
