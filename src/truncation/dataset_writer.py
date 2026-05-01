from pathlib import Path
from utils.text_sampling import TextStream
from encipherment.cipher import SubstitutionCipher, HomophonicCipher


class DatasetWriter:
    """A class for writing chunked files with ciphers to disk.

    Attributes:
        output_dir (Path): The output directory for the chunked files.
        chunk_size (int): The size of each chunk.
        prefix (str): The prefix for the chunked files.

    """

    def __init__(
        self,
        output_dir: Path,
        chunk_size: int = 10000,
        prefix: str = "chunk_",
    ) -> None:
        """Initialize the FileHandler.

        Args:
            output_dir (_type_): The output directory for the chunked files.
            chunk_size (int, optional): The size of each chunk. Defaults to 10000.
            prefix (str, optional): The prefix for the chunked files.
                Defaults to "chunk_".

        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.prefix = prefix
        self.current_file_index = 0
        self.current_file_lines = 0
        self.current_file = None

    def __enter__(self) -> "DatasetWriter":
        """Set up the output directory and initialize the first file stream."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_next_file()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: type) -> None:
        """Ensure the active file descriptor is safely closed upon exit."""
        if self.current_file is not None:
            self.current_file.close()

    def _open_next_file(self) -> None:
        """Safely rotate the file stream to the next indexed chunk."""
        if self.current_file is not None:
            self.current_file.close()

        file_path = (
            self.output_dir / f"{self.prefix}_{self.current_file_index:03d}.jsonl"
        )
        self.current_file = open(file_path, "w", encoding="utf-8")  # noqa: SIM115
        self.current_line_count = 0
        self.current_file_index += 1

    def _encipher_func(self, text_obj: TextStream) -> SubstitutionCipher:
        """Encipher the text object using a random redundancy level."""
        cipher = HomophonicCipher(text_obj)
        cipher.generate_key()
        cipher.encipher()
        return cipher

    def write(self, item: SubstitutionCipher | TextStream) -> None:
        """Encipher item if necessary, then serialize and save to disk."""
        if self.current_file is None:
            raise ValueError("File not initialized. Call __enter__ first.")
        cipher_obj: SubstitutionCipher | None = None

        if type(item) is dict:
            cipher_obj = self._encipher_func(item)
        elif isinstance(item, SubstitutionCipher):
            cipher_obj = item

        if not cipher_obj:
            raise ValueError("Invalid cipher object type: " + str(type(item)))

        if self.current_line_count >= self.chunk_size:
            self._open_next_file()

        self.current_file.write(str(cipher_obj.__json__()) + "\n")
        self.current_line_count += 1
