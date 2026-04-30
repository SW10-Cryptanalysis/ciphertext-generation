from typing import Any, Iterator, Callable
from utils.text_sampling import TextStream
from encipherment.cipher import SubstitutionCipher
import json


class CipherTruncator:
    """Class to truncate ciphers to a maximum length.

    Attributes:
        stream (Iterator[dict[str, Any]]): The continuous stream of cipher data.
        max_length (int): The maximum length of each cipher in characters.

    """

    def __init__(
        self,
        stream: Iterator[dict[str, Any]],
        max_length: int,
        length_sampler: Callable[[], int],
    ) -> None:
        """Initialize the CipherTruncater.

        Args:
            stream (Iterator[dict[str, Any]]): The continuous stream of cipher data.
            max_length (int): The maximum length of each cipher in characters.
            length_sampler (Callable[[], int]): A function to sample target lengths.

        """
        self.stream = stream
        self.max_length = max_length
        self.length_sampler = length_sampler

    def process_stream(self) -> Iterator[SubstitutionCipher | TextStream]:
        """Process the continuous generator of cipher dictionaries.

        Yields:
            SubstitutionCipher | TextStream: The original cipher object if it
            satisfies the max_length constraint, otherwise yields multiple
            TextStream chunks adhering to the sampled length distributions.

        """
        for cipher_obj in self.stream:
            plaintext = cipher_obj.get("plaintext", "")

            if self._calculate_true_length(plaintext) <= self.max_length:
                yield SubstitutionCipher.from_json(json.dumps(cipher_obj))
                continue

            yield from self._process_long_plaintext(plaintext, cipher_obj)

    def _process_long_plaintext(
        self,
        plaintext: str,
        cipher_obj: dict[str, Any],
    ) -> Iterator[TextStream]:
        """Extract length-compliant chunks from a long plaintext iteratively.

        Discards any trailing text that is shorter than the newly sampled
        target length to maintain strict distributional integrity.

        Args:
            plaintext (str): The long plaintext to split.
            cipher_obj (dict[str, Any]): The cipher object containing the plaintext.

        Yields:
            TextStream: A TextStream object containing the chunk of plaintext.

        """
        words = plaintext.split("_")
        current_index = 0
        total_words = len(words)

        while current_index < total_words:
            target_length = self.length_sampler()
            current_chunk = []
            current_length = 0

            while current_index < total_words:
                word = words[current_index]
                word_len = len(word)

                if current_length + word_len > target_length and current_length > 0:
                    break

                current_chunk.append(word)
                current_length += word_len
                current_index += 1

            chunk_text = "_".join(current_chunk)
            true_length = self._calculate_true_length(chunk_text)

            if current_index >= total_words and true_length < target_length:
                break

            if true_length > 0:
                yield TextStream(
                    text=chunk_text.replace("_", ""),
                    target_length=target_length,
                    genres=cipher_obj.get("genres", []),
                    source_id=cipher_obj.get("source_id", ""),
                    source_name=cipher_obj.get("source_name", ""),
                    length=true_length,
                    text_with_boundaries=chunk_text,
                )

    def _calculate_true_length(self, text: str) -> int:
        """Calculate the sequence length excluding word boundary markers."""
        return len(text.replace("_", ""))
