from typing import Any, Iterator
from utils.text_sampling import TextStream
from encipherment.cipher import MonoalphabeticCipher


class MonoalphabeticConverter:
    """Class to convert a stream of plaintexts into monoalphabetic ciphers.

    Attributes:
        stream (Iterator[dict[str, Any]]): The continuous stream of cipher data.

    """

    def __init__(self, stream: Iterator[dict[str, Any]]) -> None:
        """Initialize the MonoalphabeticConverter.

        Args:
            stream (Iterator[dict[str, Any]]): The continuous stream of cipher data.

        """
        self.stream = stream

    def process_stream(self) -> Iterator[MonoalphabeticCipher]:
        """Process the continuous generator of text dictionaries.

        Yields:
            MonoalphabeticCipher: The enciphered text.

        """
        for item in self.stream:
            text_stream = TextStream(
                text=item.get("plaintext", ""),
                text_with_boundaries=item.get("plaintext_with_boundaries", ""),
                length=item.get("length", len(item.get("plaintext", ""))),
                target_length=item.get("target_length", len(item.get("plaintext", ""))),
                genres=item.get("genres", []),
                source_id=item.get("source_id", ""),
                source_name=item.get("source_name", ""),
            )
            yield MonoalphabeticCipher(text_stream)
